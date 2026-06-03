"""
Build a PyTorch EEG dataset from Epoches directory.

Steps:
1. Load matching epoch + event_code file pairs (matched by session ID)
2. Transpose each epoch: (time, ch) -> (ch, time)
3. Relabel: code <= 1 -> 1, code > 1 -> 0
4. Resample time axis from 250 -> 1024 via scipy.signal.resample
5. Shuffle and wrap in a torch Dataset
6. Save to disk with torch.save
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample

# ── Config ──────────────────────────────────────────────────────────────────
EPOCH_DIR   = "Epoches"
OUTPUT_PATH = "EEGPT_Pretrain_Dataset.pt"
DROP_N_CHANNELS = 3
TARGET_LEN  = 1024
SEED        = 42
# ────────────────────────────────────────────────────────────────────────────


def extract_session(filename: str) -> str:
    """Return the session token (e.g. 'S001') from a filename."""
    m = re.search(r"ses-(S\d+)", filename)
    return m.group(1) if m else None


def load_paired_data(epoch_dir: str):
    files = os.listdir(epoch_dir)
    epoch_files = {extract_session(f): f for f in files if "epochs" in f}
    code_files  = {extract_session(f): f for f in files if "event_codes" in f}

    common_sessions = sorted(set(epoch_files) & set(code_files))
    print(f"Sessions with both epoch and code files: {common_sessions}")

    all_X, all_y = [], []
    for ses in common_sessions:
        X = np.load(os.path.join(epoch_dir, epoch_files[ses]))   # (n, time, ch)
        y = np.load(os.path.join(epoch_dir, code_files[ses]))    # (n,)
        print(f"  {ses}: X={X.shape}, codes unique={np.unique(y)}")
        all_X.append(X)
        all_y.append(y)

    X_all = np.concatenate(all_X, axis=0)   # (N, time, ch)
    y_all = np.concatenate(all_y, axis=0)   # (N,)
    return X_all, y_all


def relabel(y: np.ndarray) -> np.ndarray:
    """Codes 0-1 -> 1 (target), codes > 1 (including above 4) -> 0 (non-target)."""
    return (y <= 1).astype(np.int64)


def process(X: np.ndarray, target_len: int) -> np.ndarray:
    """
    X: (N, time, ch)
    Returns: (N, ch, target_len) float32
    """
    # Transpose to (N, ch, time)
    X = X.transpose(0, 2, 1)                        # (N, ch, time)
    # Resample time axis
    if X.shape[-1] != target_len:
        X = resample(X, target_len, axis=-1)        # (N, ch, target_len)
    return X.astype(np.float32)


class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, ch, time)  float32
        y: (N,)           int64
        """
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def main():
    print("Loading data...")
    X_raw, y_raw = load_paired_data(EPOCH_DIR)
    print(f"Total samples loaded: {X_raw.shape[0]}")

    print("Relabeling...")
    y = relabel(y_raw)
    print(f"  Label distribution -> 1 (target): {y.sum()}, 0 (non-target): {(y == 0).sum()}")

    print(f"Processing: transpose + resample to {TARGET_LEN}...")
    X = process(X_raw, TARGET_LEN)
    print(f"  Processed shape: {X.shape}")    # (N, ch, 1024)

    print(f"Dropping {DROP_N_CHANNELS} random channels...")
    rng = np.random.default_rng(SEED)
    n_channels = X.shape[1]
    drop_idx = rng.choice(n_channels, size=DROP_N_CHANNELS, replace=False)
    keep_idx = np.array([i for i in range(n_channels) if i not in drop_idx])
    X = X[:, keep_idx, :]
    print(f"  Dropped channel indices: {sorted(drop_idx.tolist())}, remaining channels: {X.shape[1]}")

    print("Shuffling...")
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    print("Building dataset...")
    dataset = EEGDataset(X, y)

    print(f"Saving dataset to '{OUTPUT_PATH}'...")
    torch.save(dataset, OUTPUT_PATH)
    print(f"Done. Dataset has {len(dataset)} samples, X shape per sample: {dataset[0][0].shape}")

    # Quick sanity-check with a DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch_X, batch_y = next(iter(loader))
    print(f"DataLoader test batch: X={batch_X.shape}, y={batch_y.shape}")


if __name__ == "__main__":
    main()

