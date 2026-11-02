# ============================================================
# Residual TCN for multi-label micro-action prediction
# Inputs: Ax, Ay, Az, Gx, Gy, Gz
# Target: atomic actions parsed from "micro-action"
# ============================================================


import os
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, average_precision_score, roc_auc_score
)


# ------------------------- Configuration -------------------------
CSV_PATH = "preprocessed_data.csv"
TARGET_COL = "micro-action"
FEATURE_COLS = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
WINDOW_SIZE = 50
STRIDE = 10
DEV_FRAC = 0.70
TRAIN_FRAC_OF_DEV = 0.80  # Overall: 56% train, 14% validation, 30% test
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 12
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20
CHANNELS = 64
MAX_POS_WEIGHT = 10.0
THRESHOLDS = np.arange(0.10, 0.91, 0.05)
SEED = 42
NUM_WORKERS = 0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# ------------------------- Load and validate data -------------------------
df = pd.read_csv(CSV_PATH)
required = FEATURE_COLS + [TARGET_COL]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")


df = df[required].copy()
for col in FEATURE_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df[TARGET_COL] = df[TARGET_COL].fillna("").astype(str)
df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
df[FEATURE_COLS] = df[FEATURE_COLS].interpolate(limit_direction="both")
df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)


def parse_actions(value):
    if not value or value.strip().lower() in {"nan", "none"}:
        return set()
    return {part.strip().lower() for part in value.split(",") if part.strip()}


parsed_labels = df[TARGET_COL].apply(parse_actions)
ACTIONS = sorted(set().union(*parsed_labels.tolist()))
if not ACTIONS:
    raise ValueError("No action labels found in the target column.")


print("Rows:", len(df))
print("Atomic actions:", ACTIONS)
print("Raw row-level positive counts:")
for action in ACTIONS:
    print(f"  {action:15s}: {sum(action in labels for labels in parsed_labels)}")


# ------------------------- Chronological split -------------------------
n = len(df)
dev_end = int(n * DEV_FRAC)
train_end = int(dev_end * TRAIN_FRAC_OF_DEV)
row_splits = {
    "train": (0, train_end),
    "val": (train_end, dev_end),
    "test": (dev_end, n),
}
print("\nChronological row partitions:")
for name, (start, end) in row_splits.items():
    print(f"  {name:5s}: [{start}, {end}) ({end-start} rows)")


# Fit scaling only on training rows. Preserve chronological order.
scaler = StandardScaler()
scaler.fit(df.iloc[:train_end][FEATURE_COLS].to_numpy(dtype=np.float32))
X_all = scaler.transform(df[FEATURE_COLS].to_numpy(dtype=np.float32)).astype(np.float32)


action_to_idx = {a: i for i, a in enumerate(ACTIONS)}
Y_all = np.zeros((n, len(ACTIONS)), dtype=np.float32)
for i, labels in enumerate(parsed_labels):
    for label in labels:
        Y_all[i, action_to_idx[label]] = 1.0


# ------------------------- Windows: no window crosses a split boundary -------------------------
def make_windows(start, end):
    xs, ys, indices = [], [], []
    for target_idx in range(start + WINDOW_SIZE - 1, end, STRIDE):
        begin = target_idx - WINDOW_SIZE + 1
        xs.append(X_all[begin:target_idx + 1])
        ys.append(Y_all[target_idx])  # label at the final timestep
        indices.append(target_idx)
    if not xs:
        return (
            np.empty((0, WINDOW_SIZE, len(FEATURE_COLS)), dtype=np.float32),
            np.empty((0, len(ACTIONS)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), np.asarray(indices)


splits = {}
for name, (start, end) in row_splits.items():
    splits[name] = make_windows(start, end)
    Xs, Ys, _ = splits[name]
    print(f"{name}: X={Xs.shape}, Y={Ys.shape}, positive counts={Ys.sum(axis=0).astype(int).tolist()}")


X_train, Y_train, _ = splits["train"]
X_val, Y_val, _ = splits["val"]
X_test, Y_test, _ = splits["test"]
if min(len(X_train), len(X_val), len(X_test)) == 0:
    raise ValueError("At least one split has no windows; reduce WINDOW_SIZE or revise split proportions.")


class WindowDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


train_loader = DataLoader(WindowDataset(X_train, Y_train), batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS,
                          pin_memory=(DEVICE.type == "cuda"))
val_loader = DataLoader(WindowDataset(X_val, Y_val), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS,
                        pin_memory=(DEVICE.type == "cuda"))
test_loader = DataLoader(WindowDataset(X_test, Y_test), batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS,
                         pin_memory=(DEVICE.type == "cuda"))


# ------------------------- Residual TCN -------------------------
class Chomp1d(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
    def forward(self, x):
        return x if self.n == 0 else x[:, :, :-self.n].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)


        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)


        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.out_act = nn.GELU()
        for layer in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        if isinstance(self.skip, nn.Conv1d):
            nn.init.kaiming_normal_(self.skip.weight, nonlinearity="relu")
            nn.init.zeros_(self.skip.bias)


    def forward(self, x):
        y = self.drop1(self.act1(self.norm1(self.chomp1(self.conv1(x)))))
        y = self.drop2(self.act2(self.norm2(self.chomp2(self.conv2(y)))))
        return self.out_act(y + self.skip(x))


class ResidualTCN(nn.Module):
    def __init__(self, input_channels, num_labels, channels=64, dropout=0.2):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv1d(input_channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        self.blocks = nn.Sequential(*[
            TemporalBlock(channels, channels, kernel_size=3, dilation=d, dropout=dropout)
            for d in [1, 2, 4, 8]
        ])
        self.head = nn.Sequential(
            nn.Linear(channels, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_labels)
        )


    def forward(self, x):
        # [batch, time, channels] -> [batch, channels, time]
        x = x.transpose(1, 2)
        x = self.project(x)
        x = self.blocks(x)
        x = x[:, :, -1]  # causal representation at the final timestep
        return self.head(x)  # logits


model = ResidualTCN(len(FEATURE_COLS), len(ACTIONS), CHANNELS, DROPOUT).to(DEVICE)
print("\nTrainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))


# ------------------------- Imbalance handling -------------------------
pos = Y_train.sum(axis=0)
neg = len(Y_train) - pos
pos_weight = np.ones(len(ACTIONS), dtype=np.float32)
valid = pos > 0
pos_weight[valid] = neg[valid] / pos[valid]
pos_weight = np.clip(pos_weight, 1.0, MAX_POS_WEIGHT)
print("\nPositive-class weights (capped at", MAX_POS_WEIGHT, "):")
for a, p, w in zip(ACTIONS, pos, pos_weight):
    print(f"  {a:15s}: positives={int(p):5d}, pos_weight={w:.3f}")


criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=DEVICE)
)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4
)


# ------------------------- Training -------------------------
def run_epoch(loader, training=False):
    model.train(training)
    loss_sum = 0.0
    probs_list, targets_list = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        loss_sum += loss.item() * len(xb)
        probs_list.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets_list.append(yb.detach().cpu().numpy())
    return loss_sum / len(loader.dataset), np.concatenate(probs_list), np.concatenate(targets_list)


best_val_loss = float("inf")
best_state = None
wait = 0
for epoch in range(1, EPOCHS + 1):
    train_loss, _, _ = run_epoch(train_loader, training=True)
    val_loss, _, _ = run_epoch(val_loader, training=False)
    scheduler.step(val_loss)
    print(f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | lr={optimizer.param_groups[0]['lr']:.2e}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping.")
            break


if best_state is None:
    raise RuntimeError("No best checkpoint was produced.")
model.load_state_dict(best_state)


# ------------------------- Validation threshold tuning -------------------------
_, val_probs, val_targets = run_epoch(val_loader, training=False)
def tune_thresholds(y_true, y_prob):
    result = []
    for j, action in enumerate(ACTIONS):
        if y_true[:, j].sum() == 0:
            print(f"WARNING: no positive validation examples for '{action}'; threshold defaults to 0.50.")
            result.append(0.50)
            continue
        scores = [
            f1_score(y_true[:, j], (y_prob[:, j] >= t).astype(int), zero_division=0)
            for t in THRESHOLDS
        ]
        result.append(float(THRESHOLDS[int(np.argmax(scores))]))
    return np.asarray(result)


thresholds = tune_thresholds(val_targets, val_probs)
print("\nValidation-selected thresholds:")
for a, t in zip(ACTIONS, thresholds):
    print(f"  {a:15s}: {t:.2f}")


# ------------------------- Test evaluation (thresholds are frozen) -------------------------
_, test_probs, test_targets = run_epoch(test_loader, training=False)
test_pred = (test_probs >= thresholds.reshape(1, -1)).astype(int)


print("\n================ TEST RESULTS ================")
print(f"Micro precision : {precision_score(test_targets, test_pred, average='micro', zero_division=0):.4f}")
print(f"Micro recall    : {recall_score(test_targets, test_pred, average='micro', zero_division=0):.4f}")
print(f"Micro F1        : {f1_score(test_targets, test_pred, average='micro', zero_division=0):.4f}")
print(f"Macro F1        : {f1_score(test_targets, test_pred, average='macro', zero_division=0):.4f}")
print(f"Weighted F1     : {f1_score(test_targets, test_pred, average='weighted', zero_division=0):.4f}")
print(f"Exact-match acc : {accuracy_score(test_targets, test_pred):.4f}")


print("\nPer-action report:")
print(classification_report(test_targets, test_pred, target_names=ACTIONS, zero_division=0, digits=4))


print("\nPer-action confusion counts and ranking metrics:")
for j, action in enumerate(ACTIONS):
    yt = test_targets[:, j].astype(int)
    yp = test_pred[:, j]
    prob = test_probs[:, j]
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    if len(np.unique(yt)) == 2:
        ap = average_precision_score(yt, prob)
        auc = roc_auc_score(yt, prob)
    else:
        ap, auc = float("nan"), float("nan")
    print(f"{action:15s} | support={int(yt.sum()):5d} | threshold={thresholds[j]:.2f} "
          f"| TP={tp:5d} FP={fp:5d} FN={fn:5d} TN={tn:5d} "
          f"| AP={ap:.4f} ROC-AUC={auc:.4f}")
