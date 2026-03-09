# ======================================================================================
# ✅ COMPLETE WORKING CODE: BERTweet + AdamW + Swarm(PSO) (Colab + /content/venv)
# - Uses ONLY torch DataLoader (NO HuggingFace datasets formatting) -> avoids NumPy 2.0 error
# - Saves ONLY JSON-serializable values (no ndarray in JSON)
# - Saves predictions/probabilities to NPZ for ROC/PR later
# - Produces:
#    1) Best hyperparams found by Swarm (PSO) on a small subset
#    2) Final training on FULL train set with best hyperparams
#    3) Final metrics on VAL + TEST
#
# INPUT FILES expected:
#   /content/train_fixed.csv
#   /content/val_fixed.csv
#   /content/test_fixed.csv
# Columns supported: auto-detects text + label from (text/tweet) and (label/toxicity)
#
# OUTPUTS:
#   /content/bertweet_swarm_out/
#       best_params.json
#       final_metrics.json
#       val_preds.npz   (y_true, prob_pos)
#       test_preds.npz  (y_true, prob_pos)
#       final_model/    (HF model + tokenizer)
# ======================================================================================

import os, sys, json, time, math, random, subprocess
import numpy as np
import pandas as pd

# ---------- HARD GUARANTEE: venv ----------
print("RUNNING:", sys.executable)
assert "/content/venv" in sys.executable, "❌ Not running /content/venv python"

# ---------- Install missing deps inside venv ----------
def pip_install(spec: str):
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", spec])

try:
    import sklearn
except Exception:
    pip_install("scikit-learn==1.5.1")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)
print("numpy:", np.__version__)

# =========================
# 0) PATHS
# =========================
TRAIN_PATH = "/content/train_fixed.csv"
VAL_PATH   = "/content/val_fixed.csv"
TEST_PATH  = "/content/test_fixed.csv"

OUT_DIR = "/content/bertweet_swarm_out"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = "vinai/bertweet-base"

# =========================
# 1) REPRODUCIBILITY
# =========================
SEED = 42
def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(SEED)

# =========================
# 2) LOAD DATA (auto detect columns)
# =========================
TEXT_CANDIDATES  = ["text", "tweet", "content", "comment", "sentence", "post"]
LABEL_CANDIDATES = ["label", "toxicity", "toxic", "class", "target"]

def load_df(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)

train_df = load_df(TRAIN_PATH)
val_df   = load_df(VAL_PATH)
test_df  = load_df(TEST_PATH)

def detect_cols(df: pd.DataFrame):
    cols_lower = {c.lower(): c for c in df.columns}
    text_col, label_col = None, None
    for c in TEXT_CANDIDATES:
        if c in cols_lower:
            text_col = cols_lower[c]
            break
    for c in LABEL_CANDIDATES:
        if c in cols_lower:
            label_col = cols_lower[c]
            break
    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect cols. Found: {list(df.columns)}")
    return text_col, label_col

TEXT_COL, LABEL_COL = detect_cols(train_df)

for df in (train_df, val_df, test_df):
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df[TEXT_COL] = df[TEXT_COL].astype(str).fillna("")

print("\n✅ Loaded splits:")
print(f"TEXT_COL={TEXT_COL} | LABEL_COL={LABEL_COL}")
print(f"Train: {len(train_df)} Val: {len(val_df)} Test: {len(test_df)}")
print("Train label dist:", dict(pd.Series(train_df[LABEL_COL]).value_counts()))

# =========================
# 3) TORCH DATASET
# =========================
class TextClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

def make_loader(df, tokenizer, max_len, bs, shuffle=False):
    ds = TextClsDataset(
        df[TEXT_COL].tolist(),
        df[LABEL_COL].tolist(),
        tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0)

# =========================
# 4) TRAIN / EVAL LOOPS
# =========================
@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    all_labels = []
    all_preds = []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        out = model(**batch)
        logits = out.logits
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        labels = batch["labels"].detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    acc = float(accuracy_score(y_true, y_pred))
    f1  = float(f1_score(y_true, y_pred))
    return acc, f1

def train_one_run(
    run_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_name: str,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    max_len: int,
    warmup_ratio: float = 0.0,
):
    seed_all(SEED)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(DEVICE)

    train_loader = make_loader(train_df, tokenizer, max_len, batch_size, shuffle=True)
    val_loader   = make_loader(val_df, tokenizer, max_len, batch_size, shuffle=False)

    # optimizer: torch.optim.AdamW (NOT HF deprecated AdamW)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # simple linear warmup scheduler (optional)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_f1 = -1.0
    best_state = None
    start = time.time()
    global_step = 0

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            loss = out.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())
            global_step += 1

        avg_loss = total_loss / max(1, len(train_loader))
        val_acc, val_f1 = eval_model(model, val_loader)

        print(f"[{run_name}] epoch {ep}/{epochs} | train_loss={avg_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    train_time = time.time() - start

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # final val metrics
    val_acc, val_f1 = eval_model(model, val_loader)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "val_acc": float(val_acc),
        "val_f1": float(val_f1),
        "train_time_sec": float(train_time),
        "params": {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "max_len": int(max_len),
            "warmup_ratio": float(warmup_ratio),
        }
    }

# =========================
# 5) SWARM (PSO) SEARCH
#    - searches lr + weight_decay + batch_size + max_len
#    - uses small subsets to be FAST
# =========================
SWARM_ENABLED = True

NUM_PARTICLES = 6
NUM_ITERS     = 3

TRAIN_SUBSET = 9000
VAL_SUBSET   = 3000

# search ranges
LR_MIN, LR_MAX = 5e-6, 3e-5
WD_MIN, WD_MAX = 0.0, 0.05
BS_CHOICES = [16, 32]
MAXLEN_CHOICES = [96, 128, 160]

EPOCHS_SWARM = 1   # keep fast
EPOCHS_FINAL = 1   # you can set 2 later if needed

def sample_subset(df, n):
    if n >= len(df):
        return df
    return df.sample(n=n, random_state=SEED).reset_index(drop=True)

train_small = sample_subset(train_df, TRAIN_SUBSET)
val_small   = sample_subset(val_df, VAL_SUBSET)

def clip(x, lo, hi):
    return max(lo, min(hi, x))

def particle_to_params(pos):
    # pos = [lr, wd, bs_idx, maxlen_idx]
    lr = clip(pos[0], LR_MIN, LR_MAX)
    wd = clip(pos[1], WD_MIN, WD_MAX)
    bs = BS_CHOICES[int(round(clip(pos[2], 0, len(BS_CHOICES)-1)))]
    ml = MAXLEN_CHOICES[int(round(clip(pos[3], 0, len(MAXLEN_CHOICES)-1)))]
    return lr, wd, bs, ml

def swarm_search():
    print("\n==========================================================================================")
    print("🐝 SWARM SEARCH (PSO) for BERTweet + AdamW hyperparams (fast subset)")
    print("==========================================================================================")

    # init particles
    particles = []
    velocities = []
    pbest_pos = []
    pbest_score = []

    for _ in range(NUM_PARTICLES):
        lr0 = random.uniform(LR_MIN, LR_MAX)
        wd0 = random.uniform(WD_MIN, WD_MAX)
        bs0 = random.uniform(0, len(BS_CHOICES)-1)
        ml0 = random.uniform(0, len(MAXLEN_CHOICES)-1)
        pos = np.array([lr0, wd0, bs0, ml0], dtype=np.float64)
        vel = np.zeros_like(pos)
        particles.append(pos)
        velocities.append(vel)
        pbest_pos.append(pos.copy())
        pbest_score.append(-1.0)

    gbest_pos = pbest_pos[0].copy()
    gbest_score = -1.0

    # PSO constants
    w = 0.6
    c1 = 1.6
    c2 = 1.6

    for it in range(1, NUM_ITERS+1):
        print(f"\n--- PSO ITER {it}/{NUM_ITERS} ---")

        for i in range(NUM_PARTICLES):
            lr, wd, bs, ml = particle_to_params(particles[i])

            run_name = f"SWARM_p{i+1}_it{it}"
            res = train_one_run(
                run_name=run_name,
                train_df=train_small,
                val_df=val_small,
                model_name=MODEL_NAME,
                lr=lr,
                weight_decay=wd,
                batch_size=bs,
                epochs=EPOCHS_SWARM,
                max_len=ml,
                warmup_ratio=0.0
            )

            score = res["val_f1"]  # optimize val F1
            print(f"   particle {i+1}: lr={lr:.2e} wd={wd:.3f} bs={bs} max_len={ml} -> val_f1={score:.4f}")

            # personal best
            if score > pbest_score[i]:
                pbest_score[i] = score
                pbest_pos[i] = particles[i].copy()

            # global best
            if score > gbest_score:
                gbest_score = score
                gbest_pos = particles[i].copy()

            # free memory
            del res["model"]
            torch.cuda.empty_cache()

        # update velocities and positions
        for i in range(NUM_PARTICLES):
            r1 = np.random.rand(4)
            r2 = np.random.rand(4)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_pos[i] - particles[i])
                + c2 * r2 * (gbest_pos - particles[i])
            )
            particles[i] = particles[i] + velocities[i]

    lr, wd, bs, ml = particle_to_params(gbest_pos)
    best = {"lr": float(lr), "weight_decay": float(wd), "batch_size": int(bs), "max_len": int(ml), "best_val_f1_subset": float(gbest_score)}
    return best

# =========================
# 6) RUN SWARM + FINAL TRAIN
# =========================
best_params = None
if SWARM_ENABLED:
    best_params = swarm_search()
else:
    best_params = {"lr": 1e-5, "weight_decay": 0.01, "batch_size": 32, "max_len": 128, "best_val_f1_subset": None}

# Save best params (JSON-safe)
with open(os.path.join(OUT_DIR, "best_params.json"), "w") as f:
    json.dump(best_params, f, indent=2)

print("\n✅ BEST PARAMS (from swarm subset search):")
print(best_params)

print("\n==========================================================================================")
print("🚀 FINAL TRAINING on FULL TRAIN set with BEST PARAMS")
print("==========================================================================================")

final_res = train_one_run(
    run_name="BERTweet_AdamW_FINAL",
    train_df=train_df,
    val_df=val_df,
    model_name=MODEL_NAME,
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"],
    batch_size=best_params["batch_size"],
    epochs=EPOCHS_FINAL,
    max_len=best_params["max_len"],
    warmup_ratio=0.0
)

# =========================
# 7) TEST EVAL + SAVE PROBS (for ROC later)
# =========================
@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs_pos = []
    y_true = []
    for batch in loader:
        labels = batch["labels"].numpy()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(**batch).logits
        prob = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        probs_pos.append(prob)
        y_true.append(labels)
    return np.concatenate(y_true), np.concatenate(probs_pos)

tokenizer = final_res["tokenizer"]
model = final_res["model"]

val_loader  = make_loader(val_df, tokenizer, best_params["max_len"], best_params["batch_size"], shuffle=False)
test_loader = make_loader(test_df, tokenizer, best_params["max_len"], best_params["batch_size"], shuffle=False)

val_y, val_p = predict_probs(model, val_loader)
test_y, test_p = predict_probs(model, test_loader)

# threshold 0.5 metrics
val_pred = (val_p >= 0.5).astype(int)
test_pred = (test_p >= 0.5).astype(int)

val_acc = float(accuracy_score(val_y, val_pred))
val_f1  = float(f1_score(val_y, val_pred))
test_acc = float(accuracy_score(test_y, test_pred))
test_f1  = float(f1_score(test_y, test_pred))

final_metrics = {
    "model_name": MODEL_NAME,
    "best_params": best_params,
    "final_val_acc": val_acc,
    "final_val_f1": val_f1,
    "final_test_acc": test_acc,
    "final_test_f1": test_f1,
    "train_time_sec": float(final_res["train_time_sec"])
}

with open(os.path.join(OUT_DIR, "final_metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=2)

# Save preds for ROC/PR later
np.savez(os.path.join(OUT_DIR, "val_preds.npz"), y_true=val_y.astype(np.int64), prob_pos=val_p.astype(np.float32))
np.savez(os.path.join(OUT_DIR, "test_preds.npz"), y_true=test_y.astype(np.int64), prob_pos=test_p.astype(np.float32))

# Save model
model_dir = os.path.join(OUT_DIR, "final_model")
os.makedirs(model_dir, exist_ok=True)
tokenizer.save_pretrained(model_dir)
model.save_pretrained(model_dir)

print("\n✅ DONE ✅")
print("Saved to:", OUT_DIR)
print(json.dumps(final_metrics, indent=2))
EOF
