# Multimodal multi-task Transformer for driving-cause and driver-motive prediction
#
# Inputs:
#   - Six-axis IMU: Ax, Ay, Az, Gx, Gy, Gz
#   - Categorical and numerical context
#   - Learned micro-action probabilities from an auxiliary IMU action head
#
# Outputs:
#   - Causes: multi-label prediction (static obstacle, dynamic obstacle, road signage)
#   - Motive: single-label driver-motive prediction
#   - Auxiliary output: multi-label micro-action prediction
#
# Each window predicts labels at its final timestep. Windows never cross
# chronological train/validation/test boundaries.

import os
import copy
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, average_precision_score, roc_auc_score,
    confusion_matrix
)

# --------------------------- Configuration ---------------------------
CSV_PATH = "preprocessed_data.csv"
OUTPUT_DIR = "multimodal_transformer_output"

IMU_COLS = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]

CATEGORICAL_CONTEXT_COLS = [
    "road texture",
    "elevation",
    "traffic congestion",
    "traffic density",
    "weather",
    "light condition",
    "structural constraints",
]
NUMERIC_CONTEXT_COLS = ["Speed", "Alt"]

ACTION_COL = "micro-action"
CAUSE_COLS = ["static obstacle", "dynamic obstacle", "road signage"]
MOTIVE_COL = "driver's motive"

WINDOW_SIZE = 50
STRIDE = 10
TRAIN_FRAC = 0.56
VAL_FRAC = 0.14  # Remaining 30% is test.

BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 12
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
MAX_CAUSE_POS_WEIGHT = 10.0
MAX_ACTION_POS_WEIGHT = 10.0
SEED = 42
NUM_WORKERS = 0

LAMBDA_ACTION = 0.30
LAMBDA_CAUSE = 1.00
LAMBDA_MOTIVE = 1.00


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Device:", DEVICE)

# --------------------------- Load and validate data ---------------------------
df = pd.read_csv(CSV_PATH)

required_cols = IMU_COLS + [ACTION_COL] + CAUSE_COLS + [MOTIVE_COL]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {df.columns.tolist()}"
    )

cat_cols = [c for c in CATEGORICAL_CONTEXT_COLS if c in df.columns]
num_cols = [c for c in NUMERIC_CONTEXT_COLS if c in df.columns]
missing_context = [
    c for c in CATEGORICAL_CONTEXT_COLS + NUMERIC_CONTEXT_COLS
    if c not in df.columns
]
if missing_context:
    print("Context columns not found and skipped:", missing_context)
if not cat_cols and not num_cols:
    raise ValueError("No configured context columns were found.")

use_cols = IMU_COLS + cat_cols + num_cols + [ACTION_COL] + CAUSE_COLS + [MOTIVE_COL]
df = df[use_cols].copy().reset_index(drop=True)

# Preserve chronological order. Do not drop rows based on target values.
for col in IMU_COLS + num_cols + CAUSE_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)

for col in cat_cols + [ACTION_COL, MOTIVE_COL]:
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    df.loc[df[col].eq(""), col] = "Unknown"

df[CAUSE_COLS] = df[CAUSE_COLS].fillna(0).clip(0, 1)

# --------------------------- Chronological row partitions ---------------------------
n = len(df)
train_end = int(n * TRAIN_FRAC)
val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

row_splits = {
    "train": (0, train_end),
    "val": (train_end, val_end),
    "test": (val_end, n),
}
if min(train_end, val_end - train_end, n - val_end) < WINDOW_SIZE:
    raise ValueError("A chronological partition is shorter than WINDOW_SIZE.")

print("\nChronological row partitions:")
for name, (start, end) in row_splits.items():
    print(f"{name:5s}: [{start}, {end}) ({end-start} rows)")

# Interpolate sensor/context values independently within each partition.
# This prevents interpolation from using observations across split boundaries.
continuous_cols = IMU_COLS + num_cols
for start, end in row_splits.values():
    part = df.loc[start:end-1, continuous_cols].copy()
    part = part.interpolate(method="linear", limit_direction="both")
    part = part.fillna(0.0)
    df.loc[start:end-1, continuous_cols] = part

# --------------------------- Target encoding ---------------------------
def parse_actions(value):
    if not value or value.strip().lower() in {"nan", "none", "unknown"}:
        return set()
    return {
        item.strip().lower()
        for item in value.split(",")
        if item.strip() and item.strip().lower() not in {"none", "unknown"}
    }

parsed_actions = df[ACTION_COL].apply(parse_actions)

# Build target vocabularies from training rows only.
train_actions = parsed_actions.iloc[:train_end]
ACTION_CLASSES = sorted(set().union(*train_actions.tolist()))
if not ACTION_CLASSES:
    raise ValueError("No micro-action labels found in the training partition.")

train_motives = df.iloc[:train_end][MOTIVE_COL].astype(str)
MOTIVE_CLASSES = sorted(train_motives.unique().tolist())
if not MOTIVE_CLASSES or MOTIVE_CLASSES == ["Unknown"]:
    raise ValueError("No valid driver-motive labels found in the training partition.")

unknown_motives = sorted(set(df[MOTIVE_COL].astype(str)) - set(MOTIVE_CLASSES))
if unknown_motives:
    raise ValueError(
        "Motive classes occur in validation/test but not training: "
        f"{unknown_motives}. Add training examples or revise the split."
    )

unknown_actions = sorted(
    set().union(*parsed_actions.iloc[train_end:].tolist()) - set(ACTION_CLASSES)
)
if unknown_actions:
    raise ValueError(
        "Micro-action classes occur outside training but not in training: "
        f"{unknown_actions}. Add training examples or revise the split."
    )

action_to_idx = {label: i for i, label in enumerate(ACTION_CLASSES)}
motive_to_idx = {label: i for i, label in enumerate(MOTIVE_CLASSES)}

Y_action_all = np.zeros((n, len(ACTION_CLASSES)), dtype=np.float32)
for row_idx, labels in enumerate(parsed_actions):
    for label in labels:
        Y_action_all[row_idx, action_to_idx[label]] = 1.0

Y_motive_all = df[MOTIVE_COL].map(motive_to_idx).to_numpy(dtype=np.int64)
Y_cause_all = df[CAUSE_COLS].to_numpy(dtype=np.float32)

print("\nAction classes:", ACTION_CLASSES)
print("Motive classes:", MOTIVE_CLASSES)
print("Cause labels:", CAUSE_COLS)

# --------------------------- Feature encoding ---------------------------
# Fit scalers and categorical vocabularies on training rows only.
imu_scaler = StandardScaler()
imu_scaler.fit(df.iloc[:train_end][IMU_COLS].to_numpy(dtype=np.float32))
X_imu_all = imu_scaler.transform(
    df[IMU_COLS].to_numpy(dtype=np.float32)
).astype(np.float32)

if num_cols:
    num_scaler = StandardScaler()
    num_scaler.fit(df.iloc[:train_end][num_cols].to_numpy(dtype=np.float32))
    X_num_all = num_scaler.transform(
        df[num_cols].to_numpy(dtype=np.float32)
    ).astype(np.float32)
else:
    X_num_all = np.zeros((n, 0), dtype=np.float32)

category_maps = {}
cat_arrays = []
for col in cat_cols:
    train_values = sorted(df.iloc[:train_end][col].astype(str).unique().tolist())
    mapping = {value: idx + 1 for idx, value in enumerate(train_values)}
    category_maps[col] = mapping
    cat_arrays.append(
        df[col].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64)
    )

if cat_arrays:
    X_cat_all = np.stack(cat_arrays, axis=1)
else:
    X_cat_all = np.zeros((n, 0), dtype=np.int64)

cat_cardinalities = [len(category_maps[col]) + 1 for col in cat_cols]

# --------------------------- Window generation ---------------------------
def make_windows(start, end):
    xs_imu, xs_cat, xs_num = [], [], []
    ys_action, ys_cause, ys_motive, end_indices = [], [], [], []

    for target_idx in range(start + WINDOW_SIZE - 1, end, STRIDE):
        begin = target_idx - WINDOW_SIZE + 1
        xs_imu.append(X_imu_all[begin:target_idx + 1])
        xs_cat.append(X_cat_all[begin:target_idx + 1])
        xs_num.append(X_num_all[begin:target_idx + 1])
        ys_action.append(Y_action_all[target_idx])
        ys_cause.append(Y_cause_all[target_idx])
        ys_motive.append(Y_motive_all[target_idx])
        end_indices.append(target_idx)

    if not xs_imu:
        return None

    return {
        "imu": np.asarray(xs_imu, dtype=np.float32),
        "cat": np.asarray(xs_cat, dtype=np.int64),
        "num": np.asarray(xs_num, dtype=np.float32),
        "action": np.asarray(ys_action, dtype=np.float32),
        "cause": np.asarray(ys_cause, dtype=np.float32),
        "motive": np.asarray(ys_motive, dtype=np.int64),
        "indices": np.asarray(end_indices, dtype=np.int64),
    }


splits = {
    name: make_windows(start, end)
    for name, (start, end) in row_splits.items()
}
for name, data in splits.items():
    if data is None:
        raise ValueError(f"No windows generated for {name}.")
    print(
        f"{name}: N={len(data['imu'])}, "
        f"action positives={data['action'].sum(axis=0).astype(int).tolist()}, "
        f"cause positives={data['cause'].sum(axis=0).astype(int).tolist()}, "
        f"motive counts={np.bincount(data['motive'], minlength=len(MOTIVE_CLASSES)).tolist()}"
    )


class MultiTaskDataset(Dataset):
    def __init__(self, data):
        self.imu = torch.from_numpy(data["imu"]).float()
        self.cat = torch.from_numpy(data["cat"]).long()
        self.num = torch.from_numpy(data["num"]).float()
        self.action = torch.from_numpy(data["action"]).float()
        self.cause = torch.from_numpy(data["cause"]).float()
        self.motive = torch.from_numpy(data["motive"]).long()

    def __len__(self):
        return len(self.imu)

    def __getitem__(self, idx):
        return (
            self.imu[idx], self.cat[idx], self.num[idx],
            self.action[idx], self.cause[idx], self.motive[idx]
        )


train_loader = DataLoader(
    MultiTaskDataset(splits["train"]), batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
)
val_loader = DataLoader(
    MultiTaskDataset(splits["val"]), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
)
test_loader = DataLoader(
    MultiTaskDataset(splits["test"]), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda")
)

# --------------------------- Model ---------------------------
class MultimodalTransformer(nn.Module):
    def __init__(
        self, imu_dim, num_numeric, cat_cardinalities,
        num_actions, num_causes, num_motives,
        d_model=64, n_heads=4, n_layers=3, dropout=0.2,
        max_len=WINDOW_SIZE
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.imu_projection = nn.Sequential(
            nn.Linear(imu_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Auxiliary action predictions are generated from IMU features only.
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_actions)
        )
        self.action_projection = nn.Sequential(
            nn.Linear(num_actions, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        embedding_dim = min(16, d_model // 2)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim, padding_idx=0)
            for cardinality in cat_cardinalities
        ])
        context_dim = len(cat_cardinalities) * embedding_dim + num_numeric
        self.context_projection = nn.Sequential(
            nn.Linear(max(1, context_dim), d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.position_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        self.fusion = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers,
            enable_nested_tensor=False
        )

        self.cause_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_causes)
        )
        self.motive_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_motives)
        )

    def forward(self, imu, cat, num, causal_mask):
        # IMU branch and auxiliary action predictions.
        imu_features = self.imu_projection(imu)
        action_logits_all = self.action_head(imu_features)
        action_probabilities = torch.sigmoid(action_logits_all)
        action_features = self.action_projection(action_probabilities)

        # Context branch.
        cat_features = []
        for col_idx, embedding in enumerate(self.cat_embeddings):
            cat_features.append(embedding(cat[:, :, col_idx]))

        if cat_features:
            context_parts = cat_features + ([num] if num.shape[-1] > 0 else [])
            context_input = torch.cat(context_parts, dim=-1)
        elif num.shape[-1] > 0:
            context_input = num
        else:
            context_input = torch.zeros(
                imu.shape[0], imu.shape[1], 1, device=imu.device
            )

        context_features = self.context_projection(context_input)

        # Three-way fusion: IMU, context, and learned action probabilities.
        fused = self.fusion(
            torch.cat([imu_features, context_features, action_features], dim=-1)
        )
        fused = fused + self.position_embedding[:, :fused.size(1), :]

        encoded = self.transformer(fused, mask=causal_mask)
        final_state = encoded[:, -1, :]

        cause_logits = self.cause_head(final_state)
        motive_logits = self.motive_head(final_state)
        action_logits = action_logits_all[:, -1, :]

        return cause_logits, motive_logits, action_logits


model = MultimodalTransformer(
    imu_dim=len(IMU_COLS),
    num_numeric=len(num_cols),
    cat_cardinalities=cat_cardinalities,
    num_actions=len(ACTION_CLASSES),
    num_causes=len(CAUSE_COLS),
    num_motives=len(MOTIVE_CLASSES),
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

# --------------------------- Training-only loss weights ---------------------------
def multilabel_pos_weight(targets, max_weight):
    positives = targets.sum(axis=0)
    negatives = len(targets) - positives
    weights = negatives / np.maximum(positives, 1.0)
    return np.clip(weights, 1.0, max_weight).astype(np.float32)


train_action_y = splits["train"]["action"]
train_cause_y = splits["train"]["cause"]
train_motive_y = splits["train"]["motive"]

action_pos_weight = multilabel_pos_weight(
    train_action_y, MAX_ACTION_POS_WEIGHT
)
cause_pos_weight = multilabel_pos_weight(
    train_cause_y, MAX_CAUSE_POS_WEIGHT
)

motive_counts = np.bincount(
    train_motive_y, minlength=len(MOTIVE_CLASSES)
).astype(np.float32)
motive_weights = len(train_motive_y) / (
    len(MOTIVE_CLASSES) * np.maximum(motive_counts, 1.0)
)
motive_weights = motive_weights / motive_weights.mean()

action_criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(action_pos_weight, dtype=torch.float32, device=DEVICE)
)
cause_criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(cause_pos_weight, dtype=torch.float32, device=DEVICE)
)
motive_criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(motive_weights, dtype=torch.float32, device=DEVICE)
)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4
)

print("\nCause positive weights:", dict(zip(CAUSE_COLS, cause_pos_weight.tolist())))
print("Action positive weights:", dict(zip(ACTION_CLASSES, action_pos_weight.tolist())))
print("Motive class weights:", dict(zip(MOTIVE_CLASSES, motive_weights.tolist())))

# Boolean True entries mask future positions in PyTorch TransformerEncoder.
causal_mask = torch.triu(
    torch.ones(WINDOW_SIZE, WINDOW_SIZE, dtype=torch.bool, device=DEVICE),
    diagonal=1
)

# --------------------------- Training / validation ---------------------------
def run_epoch(loader, training=False):
    model.train(training)
    total_loss = 0.0
    n_seen = 0
    outputs = {
        "cause_probs": [], "cause_y": [],
        "motive_probs": [], "motive_y": [],
        "action_probs": [], "action_y": []
    }

    for imu, cat, num, action_y, cause_y, motive_y in loader:
        imu = imu.to(DEVICE, non_blocking=True)
        cat = cat.to(DEVICE, non_blocking=True)
        num = num.to(DEVICE, non_blocking=True)
        action_y = action_y.to(DEVICE, non_blocking=True)
        cause_y = cause_y.to(DEVICE, non_blocking=True)
        motive_y = motive_y.to(DEVICE, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            cause_logits, motive_logits, action_logits = model(
                imu, cat, num, causal_mask
            )
            loss_cause = cause_criterion(cause_logits, cause_y)
            loss_motive = motive_criterion(motive_logits, motive_y)
            loss_action = action_criterion(action_logits, action_y)

            loss = (
                LAMBDA_CAUSE * loss_cause
                + LAMBDA_MOTIVE * loss_motive
                + LAMBDA_ACTION * loss_action
            )

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = len(imu)
        total_loss += loss.item() * batch_size
        n_seen += batch_size

        outputs["cause_probs"].append(torch.sigmoid(cause_logits).detach().cpu().numpy())
        outputs["cause_y"].append(cause_y.detach().cpu().numpy())
        outputs["motive_probs"].append(torch.softmax(motive_logits, dim=1).detach().cpu().numpy())
        outputs["motive_y"].append(motive_y.detach().cpu().numpy())
        outputs["action_probs"].append(torch.sigmoid(action_logits).detach().cpu().numpy())
        outputs["action_y"].append(action_y.detach().cpu().numpy())

    result = {"loss": total_loss / max(1, n_seen)}
    for key, values in outputs.items():
        result[key] = np.concatenate(values, axis=0)
    return result


best_val_loss = float("inf")
best_state = None
wait = 0

for epoch in range(1, EPOCHS + 1):
    train_metrics = run_epoch(train_loader, training=True)
    val_metrics = run_epoch(val_loader, training=False)
    scheduler.step(val_metrics["loss"])

    print(
        f"Epoch {epoch:03d}/{EPOCHS} | "
        f"train_loss={train_metrics['loss']:.5f} | "
        f"val_loss={val_metrics['loss']:.5f} | "
        f"lr={optimizer.param_groups[0]['lr']:.2e}"
    )

    if val_metrics["loss"] < best_val_loss:
        best_val_loss = val_metrics["loss"]
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping.")
            break

if best_state is None:
    raise RuntimeError("No best model checkpoint was produced.")

model.load_state_dict(best_state)

# --------------------------- Tune cause thresholds on validation only ---------------------------
val_metrics = run_epoch(val_loader, training=False)
threshold_grid = np.arange(0.10, 0.91, 0.05)
cause_thresholds = []

for j, cause in enumerate(CAUSE_COLS):
    y = val_metrics["cause_y"][:, j].astype(int)
    p = val_metrics["cause_probs"][:, j]

    if len(np.unique(y)) < 2:
        threshold = 0.50
        print(
            f"WARNING: validation labels for '{cause}' contain only one class; "
            "threshold set to 0.50."
        )
    else:
        scores = [
            f1_score(y, (p >= threshold).astype(int), zero_division=0)
            for threshold in threshold_grid
        ]
        threshold = float(threshold_grid[int(np.argmax(scores))])

    cause_thresholds.append(threshold)

cause_thresholds = np.asarray(cause_thresholds, dtype=np.float32)

# --------------------------- Final test evaluation ---------------------------
test_metrics = run_epoch(test_loader, training=False)
cause_pred = (
    test_metrics["cause_probs"] >= cause_thresholds[None, :]
).astype(int)
motive_pred = test_metrics["motive_probs"].argmax(axis=1)

print("\n================ CAUSE TEST RESULTS ================")
print("Thresholds:", dict(zip(CAUSE_COLS, cause_thresholds.tolist())))
print("Micro precision:", precision_score(
    test_metrics["cause_y"], cause_pred, average="micro", zero_division=0
))
print("Micro recall:", recall_score(
    test_metrics["cause_y"], cause_pred, average="micro", zero_division=0
))
print("Micro F1:", f1_score(
    test_metrics["cause_y"], cause_pred, average="micro", zero_division=0
))
print("Macro F1:", f1_score(
    test_metrics["cause_y"], cause_pred, average="macro", zero_division=0
))
print(classification_report(
    test_metrics["cause_y"], cause_pred,
    target_names=CAUSE_COLS, zero_division=0, digits=4
))

for j, cause in enumerate(CAUSE_COLS):
    yt = test_metrics["cause_y"][:, j].astype(int)
    yp = cause_pred[:, j]
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())

    if len(np.unique(yt)) == 2:
        ap = average_precision_score(yt, test_metrics["cause_probs"][:, j])
        auc = roc_auc_score(yt, test_metrics["cause_probs"][:, j])
    else:
        ap, auc = float("nan"), float("nan")

    print(
        f"{cause:20s}: support={int(yt.sum())}, TP={tp}, FP={fp}, "
        f"FN={fn}, TN={tn}, AP={ap:.4f}, AUC={auc:.4f}"
    )

print("\n================ MOTIVE TEST RESULTS ================")
print("Accuracy:", accuracy_score(test_metrics["motive_y"], motive_pred))
print("Macro F1:", f1_score(
    test_metrics["motive_y"], motive_pred, average="macro", zero_division=0
))
print("Weighted F1:", f1_score(
    test_metrics["motive_y"], motive_pred, average="weighted", zero_division=0
))
print(classification_report(
    test_metrics["motive_y"], motive_pred,
    labels=np.arange(len(MOTIVE_CLASSES)),
    target_names=MOTIVE_CLASSES,
    zero_division=0, digits=4
))
print("Motive confusion matrix (class order shown above):")
print(confusion_matrix(
    test_metrics["motive_y"], motive_pred,
    labels=np.arange(len(MOTIVE_CLASSES))
))

print("\n================ AUXILIARY ACTION TEST RESULTS ================")
action_pred = (test_metrics["action_probs"] >= 0.50).astype(int)
print("Micro F1:", f1_score(
    test_metrics["action_y"], action_pred, average="micro", zero_division=0
))
print("Macro F1:", f1_score(
    test_metrics["action_y"], action_pred, average="macro", zero_division=0
))
print(classification_report(
    test_metrics["action_y"], action_pred,
    target_names=ACTION_CLASSES, zero_division=0, digits=4
))

# --------------------------- Save model and preprocessing metadata ---------------------------
checkpoint = {
    "model_state_dict": model.state_dict(),
    "imu_columns": IMU_COLS,
    "categorical_context_columns": cat_cols,
    "numeric_context_columns": num_cols,
    "category_maps": category_maps,
    "action_classes": ACTION_CLASSES,
    "motive_classes": MOTIVE_CLASSES,
    "cause_columns": CAUSE_COLS,
    "imu_scaler_mean": imu_scaler.mean_.tolist(),
    "imu_scaler_scale": imu_scaler.scale_.tolist(),
    "numeric_scaler_mean": num_scaler.mean_.tolist() if num_cols else [],
    "numeric_scaler_scale": num_scaler.scale_.tolist() if num_cols else [],
    "cause_thresholds": cause_thresholds.tolist(),
    "window_size": WINDOW_SIZE,
    "stride": STRIDE,
    "model_config": {
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "dropout": DROPOUT,
    },
}
torch.save(checkpoint, os.path.join(OUTPUT_DIR, "best_model.pt"))

with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(
        {k: v for k, v in checkpoint.items() if k != "model_state_dict"},
        f, indent=2
    )

print(f"\nSaved model and metadata to: {OUTPUT_DIR}")
