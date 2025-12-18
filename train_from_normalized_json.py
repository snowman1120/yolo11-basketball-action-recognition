import os
import json
import math
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Try to import your packer
# ----------------------------
def _import_packer():
    # If you're using the organized package I gave you:
    try:
        from normalize.pack_features import pack_from_normalized_json  # type: ignore
        return pack_from_normalized_json
    except Exception:
        pass

    # If you're using the standalone file you had earlier:
    try:
        from features_pack_and_standardize import pack_from_normalized_json  # type: ignore
        return pack_from_normalized_json
    except Exception:
        pass

    raise ImportError(
        "Could not import pack_from_normalized_json. "
        "Make sure you have either:\n"
        " - normalize/pack_features.py (from normalize_organized.zip)\n"
        "or\n"
        " - features_pack_and_standardize_v2.py on PYTHONPATH.\n"
    )


pack_from_normalized_json = _import_packer()


# ----------------------------
# Repro
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Standardizer for continuous dims only
# ----------------------------
@dataclass
class RunningStats:
    count: int
    sum: np.ndarray
    sumsq: np.ndarray

    @classmethod
    def create(cls, dim: int) -> "RunningStats":
        return cls(count=0, sum=np.zeros(dim, np.float64), sumsq=np.zeros(dim, np.float64))

    def update(self, x: np.ndarray):
        # x: (N, dim)
        self.count += x.shape[0]
        self.sum += x.sum(axis=0)
        self.sumsq += (x * x).sum(axis=0)

    def mean_std(self, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.sum / max(self.count, 1)
        var = (self.sumsq / max(self.count, 1)) - (mean * mean)
        var = np.maximum(var, eps)
        std = np.sqrt(var)
        return mean.astype(np.float32), std.astype(np.float32)


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, idx_cont: np.ndarray):
        self.mean = mean
        self.std = std
        self.idx_cont = idx_cont

    @classmethod
    def fit(cls, dataset: "NormalizedJsonDataset") -> "Standardizer":
        # accumulate stats over all time-steps of all train samples
        first = dataset[0]
        D = first[0].shape[1]
        idx_cont = dataset.idx_cont
        stats = RunningStats.create(dim=len(idx_cont))

        for i in range(len(dataset)):
            X, _, _ = dataset[i]  # X: (T,D)
            Xc = X[:, idx_cont]   # (T, Dc)
            stats.update(Xc.astype(np.float64))

        mean, std = stats.mean_std()
        return cls(mean=mean, std=std, idx_cont=idx_cont)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        X[:, self.idx_cont] = (X[:, self.idx_cont] - self.mean) / self.std
        return X


# ----------------------------
# Dataset
# ----------------------------
class NormalizedJsonDataset(Dataset):
    def __init__(
        self,
        json_paths: List[Path],
        label_to_id: Dict[str, int],
        target_fps: float = 30.0,
    ):
        self.json_paths = json_paths
        self.label_to_id = label_to_id
        self.target_fps = target_fps

        # Peek 1 file to get idx_cont/idx_bool from the packer
        with self.json_paths[0].open("r") as f:
            sample = json.load(f)
        X, names, idx_cont, idx_bool, t = pack_from_normalized_json(sample, target_fps=target_fps)
        self.feature_names = names
        self.idx_cont = idx_cont
        self.idx_bool = idx_bool

        # Sanity: labels in data
        self.labels = []
        for p in self.json_paths:
            with p.open("r") as f:
                d = json.load(f)
            lbl = str(d.get("metadata", {}).get("label", "")).strip().lower()
            if lbl not in self.label_to_id:
                raise ValueError(f"Unknown label '{lbl}' in {p}")
            self.labels.append(self.label_to_id[lbl])

    def __len__(self):
        return len(self.json_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, Path]:
        p = self.json_paths[idx]
        with p.open("r") as f:
            d = json.load(f)

        X, names, idx_cont, idx_bool, t = pack_from_normalized_json(d, target_fps=self.target_fps)
        y = self.labels[idx]
        return X.astype(np.float32), y, p


# ----------------------------
# Collate: pad variable-length sequences
# ----------------------------
def collate_pad(batch):
    # batch: list of (X (T,D), y, path)
    lengths = [b[0].shape[0] for b in batch]
    maxT = max(lengths)
    D = batch[0][0].shape[1]

    Xpad = np.zeros((len(batch), maxT, D), dtype=np.float32)
    mask = np.zeros((len(batch), maxT), dtype=np.bool_)  # True for valid
    y = np.zeros((len(batch),), dtype=np.int64)
    paths = []

    for i, (X, yi, p) in enumerate(batch):
        T = X.shape[0]
        Xpad[i, :T] = X
        mask[i, :T] = True
        y[i] = yi
        paths.append(str(p))

    return (
        torch.from_numpy(Xpad),
        torch.from_numpy(mask),
        torch.from_numpy(y),
        paths,
        torch.tensor(lengths, dtype=torch.int64),
    )


# ----------------------------
# Model: Tiny Transformer encoder
# ----------------------------
class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,d)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


class ActionTransformer(nn.Module):
    def __init__(self, d_in: int, num_classes: int, d_model: int = 192, nhead: int = 4,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = SinusoidalPosEnc(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D), valid_mask: (B,T) True for valid
        h = self.proj(x)
        h = self.pos(h)
        # transformer expects src_key_padding_mask=True for pads
        pad_mask = ~valid_mask
        h = self.enc(h, src_key_padding_mask=pad_mask)

        # masked mean-pool
        m = valid_mask.unsqueeze(-1).float()
        h_sum = (h * m).sum(dim=1)
        denom = m.sum(dim=1).clamp_min(1.0)
        pooled = h_sum / denom
        return self.head(pooled)


# ----------------------------
# Metrics
# ----------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))


# ----------------------------
# Train / Eval
# ----------------------------
@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    ys, preds = [], []
    total, correct = 0, 0
    for X, mask, y, paths, lengths in loader:
        X = X.to(device)
        mask = mask.to(device)
        y = y.to(device)

        logits = model(X, mask)
        p = logits.argmax(dim=-1)
        total += y.numel()
        correct += (p == y).sum().item()

        ys.append(y.cpu().numpy())
        preds.append(p.cpu().numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    acc = correct / max(total, 1)
    f1 = macro_f1(y_true, y_pred, num_classes)
    return acc, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Folder containing normalized *.json files")
    ap.add_argument("--target_fps", type=float, default=30.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="action_transformer.pt")
    args = ap.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    json_paths = sorted([p for p in data_dir.rglob("*.json") if p.is_file()])
    if not json_paths:
        raise FileNotFoundError(f"No json files found under: {data_dir}")

    # Label mapping (edit if your labels differ)
    label_to_id = {"dribble": 0, "pass": 1, "shoot": 2}
    num_classes = len(label_to_id)

    # Shuffle + split
    rng = np.random.default_rng(args.seed)
    idxs = np.arange(len(json_paths))
    rng.shuffle(idxs)
    n_val = int(round(args.val_ratio * len(json_paths)))
    val_idxs = idxs[:n_val]
    train_idxs = idxs[n_val:]

    train_paths = [json_paths[i] for i in train_idxs]
    val_paths = [json_paths[i] for i in val_idxs]

    train_ds = NormalizedJsonDataset(train_paths, label_to_id, target_fps=args.target_fps)
    val_ds = NormalizedJsonDataset(val_paths, label_to_id, target_fps=args.target_fps)

    # Fit standardizer on train only
    std = Standardizer.fit(train_ds)

    # Wrap datasets to apply transform
    class StdWrapper(Dataset):
        def __init__(self, ds: NormalizedJsonDataset, std: Standardizer):
            self.ds = ds
            self.std = std
        def __len__(self): return len(self.ds)
        def __getitem__(self, i):
            X, y, p = self.ds[i]
            X = self.std.transform(X)
            return X, y, p

    from torch.utils.data import WeightedRandomSampler

    train_wrapped = StdWrapper(train_ds, std)
    val_wrapped   = StdWrapper(val_ds, std)

    # --- balanced sampler (train only) ---
    labels = np.array(train_ds.labels, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    class_inv = 1.0 / np.maximum(class_counts, 1.0)   # inverse freq

    sample_weights = class_inv[labels]  # weight per sample
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_wrapped,
        batch_size=args.batch_size,
        sampler=sampler,      # <-- use sampler
        shuffle=False,        # <-- must be False when sampler is set
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_pad
    )

    val_loader = DataLoader(
        val_wrapped,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_pad
    )


    # Compute class weights from train labels
    train_labels = np.array(train_ds.labels, dtype=np.int64)
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    weights = (counts.sum() / np.maximum(counts, 1.0))
    weights = weights / weights.sum() * num_classes
    class_w = torch.tensor(weights, dtype=torch.float32)

    # Model
    # Determine input dim
    X0, y0, p0 = train_ds[0]
    d_in = X0.shape[1]
    model = ActionTransformer(d_in=d_in, num_classes=num_classes,
                              d_model=args.d_model, nhead=args.heads,
                              num_layers=args.layers, dropout=args.dropout)
    device = torch.device(args.device)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # label smoothing helps when classes overlap visually
    def loss_fn(logits, y):
        return F.cross_entropy(logits, y, weight=class_w.to(device), label_smoothing=0.1)

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for X, mask, y, paths, lengths in train_loader:
            X = X.to(device)
            mask = mask.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(X, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running += loss.item() * y.size(0)
            n += y.size(0)

        train_loss = running / max(n, 1)
        val_acc, val_f1 = evaluate(model, val_loader, device, num_classes)

        print(f"Epoch {epoch:03d} | train_loss {train_loss:.4f} | val_acc {val_acc:.4f} | val_macroF1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {
                "model": model.state_dict(),
                "standardizer": {
                    "mean": std.mean,
                    "std": std.std,
                    "idx_cont": std.idx_cont,
                    "feature_names": train_ds.feature_names,
                },
                "label_to_id": label_to_id,
                "config": vars(args),
            }

    if best_state is None:
        best_state = {"model": model.state_dict()}

    torch.save(best_state, args.out)
    print(f"Saved best checkpoint to: {args.out} (best val_macroF1={best_f1:.4f})")


if __name__ == "__main__":
    main()
