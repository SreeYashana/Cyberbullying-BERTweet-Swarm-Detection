
import os, json, time, random
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer, set_seed
)

# -----------------------------
# 0) HARD GUARANTEE: venv only
# -----------------------------
import sys, datasets
print("RUNNING:", sys.executable)
print("numpy:", np.__version__)
print("datasets:", datasets.__version__, datasets.__file__)
assert "/content/venv" in sys.executable, "ERROR: Not running in venv python!"
assert "/content/venv" in datasets.__file__, "ERROR: datasets is not from venv!"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# -----------------------------
# 1) YOUR FILES
# -----------------------------
BALANCED_PATH = "/content/FinalBalancedDataset.csv"
HATEDAY_PATH  = "/content/hateday_v2_hf_final (1).parquet"  # optional

OUT_DIR = "/content/ablation_results"
os.makedirs(OUT_DIR, exist_ok=True)

# Split ratios (fixed split)
SPLIT_RATIOS = (0.80, 0.10, 0.10)  # train/val/test

# Training config
SEED = 42
MAX_LEN = 128

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# 2) Load balanced dataset (your columns)
# -----------------------------
if not os.path.exists(BALANCED_PATH):
    raise FileNotFoundError(f"Missing file: {BALANCED_PATH}")

df = pd.read_csv(BALANCED_PATH)
print("\n✅ Loaded balanced CSV:", BALANCED_PATH)
print("Columns:", list(df.columns))

# Your dataset columns from your output
TEXT_COL = "tweet"
LABEL_COL = "Toxicity"
if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
    raise ValueError(f"Expected columns '{TEXT_COL}' and '{LABEL_COL}' not found!")

df = df[[TEXT_COL, LABEL_COL]].dropna()
df[TEXT_COL] = df[TEXT_COL].astype(str)
df[LABEL_COL] = df[LABEL_COL].astype(int).clip(0, 1)

df = df.rename(columns={TEXT_COL: "text", LABEL_COL: "label"})
print("Label distribution:", df["label"].value_counts().to_dict())
print("Rows:", len(df))

# -----------------------------
# 3) Fixed split (proof split)
# -----------------------------
df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n = len(df)
tr = int(n * SPLIT_RATIOS[0])
va = int(n * SPLIT_RATIOS[1])

train_df = df.iloc[:tr].copy()
val_df   = df.iloc[tr:tr+va].copy()
test_df  = df.iloc[tr+va:].copy()

train_csv = "/content/train_fixed.csv"
val_csv   = "/content/val_fixed.csv"
test_csv  = "/content/test_fixed.csv"
train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

print("\n✅ Fixed split saved:")
print(train_csv, len(train_df))
print(val_csv, len(val_df))
print(test_csv, len(test_df))

dsd = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False),
    "validation": Dataset.from_pandas(val_df, preserve_index=False),
    "test": Dataset.from_pandas(test_df, preserve_index=False),
})
print("\nDATA SIZES:", {k: len(dsd[k]) for k in dsd.keys()})

# -----------------------------
# 4) Metrics (binary)
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * precision * recall / max(1e-12, (precision + recall))

    return {"accuracy": float(accuracy), "f1": float(f1), "precision": float(precision), "recall": float(recall)}

# -----------------------------
# 5) IMPORTANT FIX (NO set_format("torch"))
#    This avoids NumPy2 copy=False crash inside datasets TorchFormatter.
# -----------------------------
def tokenize_dataset(dsd: DatasetDict, tokenizer):
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    dsd_tok = dsd.map(tok, batched=True, desc="Tokenizing")
    dsd_tok = dsd_tok.remove_columns(["text"]).rename_column("label", "labels")

    # ✅ DO NOT DO THIS (causes your error):
    # dsd_tok.set_format(type="torch")

    # Force pure python output (safe with numpy2)
    dsd_tok = dsd_tok.with_format("python")
    return dsd_tok

@dataclass
class Experiment:
    name: str
    model_name: str
    epochs: int = 1
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.0

def run_experiment(exp: Experiment):
    print("\n" + "="*90)
    print(f"EXPERIMENT: {exp.name}")
    print(f"model: {exp.model_name} | epochs: {exp.epochs} | bs: {exp.batch_size} | lr: {exp.lr} | wd: {exp.weight_decay}")
    print("="*90)

    tokenizer = AutoTokenizer.from_pretrained(exp.model_name, use_fast=True)
    dsd_tok = tokenize_dataset(dsd, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(exp.model_name, num_labels=2).to(DEVICE)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    ts = int(time.time())
    exp_out = os.path.join(OUT_DIR, f"{exp.name}_{ts}")

    args = TrainingArguments(
        output_dir=exp_out,
        eval_strategy="epoch",          # ✅ new name (no warning)
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=exp.batch_size,
        per_device_eval_batch_size=exp.batch_size,
        num_train_epochs=exp.epochs,
        learning_rate=exp.lr,
        weight_decay=exp.weight_decay,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dsd_tok["train"],
        eval_dataset=dsd_tok["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    val_metrics  = trainer.evaluate(dsd_tok["validation"])
    test_metrics = trainer.evaluate(dsd_tok["test"])

    return {
        "experiment": exp.name,
        "model_name": exp.model_name,
        "epochs": exp.epochs,
        "batch_size": exp.batch_size,
        "lr": exp.lr,
        "weight_decay": exp.weight_decay,
        "train_time_sec": float(train_time),
        "val_metrics": {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
        "test_metrics": {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))}
    }

# -----------------------------
# 6) Run BERT baseline
# -----------------------------
experiments = [
    Experiment(name="BERT_baseline", model_name="bert-base-uncased", epochs=1, batch_size=32, lr=2e-5, weight_decay=0.0),
]

all_results = []
for exp in experiments:
    all_results.append(run_experiment(exp))

# -----------------------------
# 7) Save results
# -----------------------------
json_path = os.path.join(OUT_DIR, "results.json")
csv_path  = os.path.join(OUT_DIR, "results.csv")

with open(json_path, "w") as f:
    json.dump(all_results, f, indent=2)

rows = []
for r in all_results:
    rows.append({
        "experiment": r["experiment"],
        "model_name": r["model_name"],
        "epochs": r["epochs"],
        "batch_size": r["batch_size"],
        "lr": r["lr"],
        "weight_decay": r["weight_decay"],
        "train_time_sec": r["train_time_sec"],
        "val_accuracy": r["val_metrics"].get("eval_accuracy"),
        "val_f1": r["val_metrics"].get("eval_f1"),
        "test_accuracy": r["test_metrics"].get("eval_accuracy"),
        "test_f1": r["test_metrics"].get("eval_f1"),
    })

pd.DataFrame(rows).to_csv(csv_path, index=False)

print("\n✅ DONE. Saved:")
print("JSON:", json_path)
print("CSV :", csv_path)
print("\n--- Summary ---")
for row in rows:
    print(row)

EOF