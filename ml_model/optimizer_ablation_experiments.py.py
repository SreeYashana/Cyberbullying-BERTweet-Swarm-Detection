import os, json, time, random
import numpy as np
import pandas as pd
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset as TorchDataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed
)

# -----------------------------
# 0) HARD GUARANTEE (venv only)
# -----------------------------
import sys, datasets
print("RUNNING:", sys.executable)
print("numpy:", np.__version__)
print("datasets:", datasets.__version__, datasets.__file__)
assert "/content/venv" in sys.executable, "Not running venv python"
assert "/content/venv" in datasets.__file__, "datasets not from venv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# -----------------------------
# 1) PATHS
# -----------------------------
TRAIN_PATH = "/content/train_fixed.csv"
VAL_PATH   = "/content/val_fixed.csv"
TEST_PATH  = "/content/test_fixed.csv"

OUT_DIR = "/content/ablation_results"
os.makedirs(OUT_DIR, exist_ok=True)

SEED    = 42
MAX_LEN = 128

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# 2) Auto-detect columns
# -----------------------------
TEXT_CANDIDATES  = ["tweet", "text", "comment", "content", "sentence", "post"]
LABEL_CANDIDATES = ["Toxicity", "toxicity", "label", "labels", "target", "y", "class"]

def detect_cols(df: pd.DataFrame):
    cols = list(df.columns)
    text_col = next((c for c in TEXT_CANDIDATES if c in cols), None)
    label_col = next((c for c in LABEL_CANDIDATES if c in cols), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Could not detect text/label cols. Found: {cols[:30]}")
    return text_col, label_col

def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    tcol, lcol = detect_cols(df)
    df = df[[tcol, lcol]].dropna().rename(columns={tcol:"text", lcol:"label"})
    df["label"] = df["label"].astype(int)
    return df

train_df = load_csv(TRAIN_PATH)
val_df   = load_csv(VAL_PATH)
test_df  = load_csv(TEST_PATH)

print("\n✅ Loaded fixed split")
print("Train rows:", len(train_df), "Val rows:", len(val_df), "Test rows:", len(test_df))
print("Train label dist:", train_df["label"].value_counts().to_dict())

# -----------------------------
# 3) Pure Torch dataset (NO HF datasets)
# -----------------------------
class TextClsTorchDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def build_torch_splits(tokenizer):
    train_enc = tokenizer(train_df["text"].tolist(), truncation=True, max_length=MAX_LEN)
    val_enc   = tokenizer(val_df["text"].tolist(), truncation=True, max_length=MAX_LEN)
    test_enc  = tokenizer(test_df["text"].tolist(), truncation=True, max_length=MAX_LEN)

    train_ds = TextClsTorchDataset(train_enc, train_df["label"].tolist())
    val_ds   = TextClsTorchDataset(val_enc,   val_df["label"].tolist())
    test_ds  = TextClsTorchDataset(test_enc,  test_df["label"].tolist())
    return train_ds, val_ds, test_ds

# -----------------------------
# 4) Metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    acc  = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 2 * prec * rec / max(1e-9, prec + rec)

    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

# -----------------------------
# 5) Custom Trainer with optimizer choice
# -----------------------------
class OptimizerTrainer(Trainer):
    def __init__(self, *args, optimizer_name="adamw", **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_name = optimizer_name.lower().strip()

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        decay_parameters = self.get_decay_parameter_names(self.model)
        grouped = [
            {
                "params": [p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                "weight_decay": 0.0,
            },
        ]

        if self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(grouped, lr=self.args.learning_rate, momentum=0.9)
        elif self.optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(grouped, lr=self.args.learning_rate, betas=(0.9, 0.999), eps=1e-8)
        else:
            raise ValueError("optimizer_name must be 'sgd' or 'adamw'")
        return self.optimizer

# -----------------------------
# 6) Experiments
# -----------------------------
@dataclass
class Experiment:
    name: str
    model: str
    optimizer: str
    lr: float
    bs: int
    wd: float
    epochs: int

def run_experiment(exp: Experiment):
    print("\n" + "="*95)
    print(f"EXPERIMENT: {exp.name}")
    print(f"model={exp.model} | opt={exp.optimizer} | lr={exp.lr} | bs={exp.bs} | wd={exp.wd} | epochs={exp.epochs}")
    print("="*95)

    tokenizer = AutoTokenizer.from_pretrained(exp.model, use_fast=True)
    train_ds, val_ds, test_ds = build_torch_splits(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(exp.model, num_labels=2)

    args = TrainingArguments(
        output_dir=f"{OUT_DIR}/{exp.name}",
        evaluation_strategy="epoch",
        save_strategy="no",
        learning_rate=exp.lr,
        per_device_train_batch_size=exp.bs,
        per_device_eval_batch_size=exp.bs,
        num_train_epochs=exp.epochs,
        weight_decay=exp.wd,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        seed=SEED
    )

    trainer = OptimizerTrainer(
        optimizer_name=exp.optimizer,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    val_metrics  = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds)

    return {
        "experiment": exp.name,
        "model": exp.model,
        "optimizer": exp.optimizer,
        "lr": exp.lr,
        "bs": exp.bs,
        "wd": exp.wd,
        "epochs": exp.epochs,
        "train_time_sec": float(train_time),
        "val_accuracy": float(val_metrics["eval_accuracy"]),
        "val_f1": float(val_metrics["eval_f1"]),
        "test_accuracy": float(test_metrics["eval_accuracy"]),
        "test_f1": float(test_metrics["eval_f1"]),
    }

experiments = [
    Experiment("BERT_baseline_SGD",       "bert-base-uncased",   "sgd",   2e-5, 32, 0.0, 1),
    Experiment("BERT_AdamW",              "bert-base-uncased",   "adamw", 2e-5, 32, 0.01, 1),
    Experiment("BERTweet_baseline_SGD",   "vinai/bertweet-base", "sgd",   2e-5, 32, 0.0, 1),
    Experiment("BERTweet_AdamW",          "vinai/bertweet-base", "adamw", 1e-5, 32, 0.01, 1),
]

results = []
for exp in experiments:
    results.append(run_experiment(exp))

# -----------------------------
# 7) Save results
# -----------------------------
df_new = pd.DataFrame(results)
csv_path = f"{OUT_DIR}/results.csv"
json_path = f"{OUT_DIR}/results.json"

df_new.to_csv(csv_path, index=False)
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ DONE. Results:")
print(df_new)
print("\nSaved:")
print("CSV :", csv_path)
print("JSON:", json_path)
EOF