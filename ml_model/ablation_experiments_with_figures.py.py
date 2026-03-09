
import os, json, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed
)

# ======== HARD GUARANTEE: venv only ========
import sys, datasets
print("RUNNING:", sys.executable)
print("numpy:", np.__version__)
print("datasets:", datasets.__version__, datasets.__file__)
assert "/content/venv" in sys.executable, "Not running venv python"
assert "/content/venv" in datasets.__file__, "datasets not from venv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# ======== Install sklearn if missing (only for curves/CM) ========
try:
    from sklearn.metrics import (
        roc_curve, auc,
        precision_recall_curve, average_precision_score,
        confusion_matrix
    )
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "scikit-learn==1.5.1"])
    from sklearn.metrics import (
        roc_curve, auc,
        precision_recall_curve, average_precision_score,
        confusion_matrix
    )

# =============================
# 1) Paths (your fixed split)
# =============================
TRAIN_PATH = "/content/train_fixed.csv"
VAL_PATH   = "/content/val_fixed.csv"
TEST_PATH  = "/content/test_fixed.csv"

OUT_ROOT = "/content/ablation_models"    # models saved here
FIG_ROOT = "/content/paper_figures"     # figures saved here
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(FIG_ROOT, exist_ok=True)

# Columns in your split CSV (you confirmed these)
TEXT_COL  = "text"
LABEL_COL = "label"

SEED = 42
MAX_LEN = 128
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================
# 2) Load split CSVs
# =============================
def load_csv(path):
    df = pd.read_csv(path)
    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"Expected cols {TEXT_COL},{LABEL_COL} but got {df.columns.tolist()}")
    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df

train_df = load_csv(TRAIN_PATH)
val_df   = load_csv(VAL_PATH)
test_df  = load_csv(TEST_PATH)

dsd = DatasetDict({
    "train": Dataset.from_pandas(train_df, preserve_index=False),
    "validation": Dataset.from_pandas(val_df, preserve_index=False),
    "test": Dataset.from_pandas(test_df, preserve_index=False),
})

print("DATA:", {k: len(v) for k,v in dsd.items()})

# =============================
# 3) Tokenize helper
# =============================
def tokenize_dataset(dsd, tokenizer):
    def tok(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, max_length=MAX_LEN)

    dsd_tok = dsd.map(tok, batched=True, desc="Tokenizing")
    dsd_tok = dsd_tok.remove_columns([TEXT_COL])
    dsd_tok = dsd_tok.rename_column(LABEL_COL, "labels")
    # IMPORTANT: keep HF default (do NOT force torch format here)
    # We'll let Trainer handle it safely with prediction step.
    return dsd_tok

# =============================
# 4) Plot helpers
# =============================
def softmax_2(logits):
    # logits: [N,2]
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def save_loss_curve(log_history, out_png):
    # Extract training loss points + eval metrics
    steps = []
    train_loss = []
    eval_steps = []
    eval_f1 = []

    for item in log_history:
        if "loss" in item and "epoch" in item and "step" in item:
            steps.append(item["step"])
            train_loss.append(item["loss"])
        if "eval_f1" in item and "step" in item:
            eval_steps.append(item["step"])
            eval_f1.append(item["eval_f1"])

    plt.figure()
    if len(steps) > 0:
        plt.plot(steps, train_loss)
        plt.xlabel("Step")
        plt.ylabel("Train Loss")
        plt.title("Training Loss Curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
    plt.close()

    # also save eval f1 curve if exists
    if len(eval_steps) > 0:
        out_png2 = out_png.replace("_train_loss.png", "_val_f1.png")
        plt.figure()
        plt.plot(eval_steps, eval_f1)
        plt.xlabel("Step")
        plt.ylabel("Validation F1")
        plt.title("Validation F1 Curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png2, dpi=300)
        plt.close()

def save_roc_pr_cm(y_true, y_prob1, y_pred, fig_dir, prefix):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{prefix}_roc.png"), dpi=300)
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob1)
    ap = average_precision_score(y_true, y_prob1)

    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{prefix}_pr.png"), dpi=300)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0","1"])
    plt.yticks(tick_marks, ["0","1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # annotate
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{prefix}_confusion_matrix.png"), dpi=300)
    plt.close()

    return {"roc_auc": float(roc_auc), "ap": float(ap), "cm": cm.tolist()}

# =============================
# 5) Compute metrics (for Trainer)
# =============================
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

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

# =============================
# 6) Experiments you asked
#    (BERT, BERT+AdamW, BERTweet, BERTweet+AdamW)
# =============================
EXPERIMENTS = [
    {
        "name": "BERT_baseline_SGD",
        "model": "bert-base-uncased",
        "optimizer": "sgd",
        "lr": 2e-5,
        "wd": 0.0,
        "epochs": 1,
        "bs": 32,
    },
    {
        "name": "BERT_AdamW",
        "model": "bert-base-uncased",
        "optimizer": "adamw",
        "lr": 2e-5,
        "wd": 0.01,
        "epochs": 1,
        "bs": 32,
    },
    {
        "name": "BERTweet_baseline_SGD",
        "model": "vinai/bertweet-base",
        "optimizer": "sgd",
        "lr": 2e-5,
        "wd": 0.0,
        "epochs": 1,
        "bs": 32,
    },
    {
        "name": "BERTweet_AdamW",
        "model": "vinai/bertweet-base",
        "optimizer": "adamw",
        "lr": 1e-5,
        "wd": 0.01,
        "epochs": 1,
        "bs": 32,
    },
]

# =============================
# 7) Run training + save plots
# =============================
all_rows = []

for exp in EXPERIMENTS:
    name = exp["name"]
    model_name = exp["model"]
    opt = exp["optimizer"]

    print("\n" + "="*95)
    print(f"EXPERIMENT: {name}")
    print(f"model={model_name} | opt={opt} | lr={exp['lr']} | wd={exp['wd']} | bs={exp['bs']} | epochs={exp['epochs']}")
    print("="*95)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    dsd_tok = tokenize_dataset(dsd, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    if torch.cuda.is_available():
        model.cuda()

    # output dirs
    ts = int(time.time())
    model_out = os.path.join(OUT_ROOT, f"{name}_{ts}")
    fig_dir   = os.path.join(FIG_ROOT, f"{name}_{ts}")
    os.makedirs(fig_dir, exist_ok=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=model_out,
        eval_strategy="epoch",
        save_strategy="epoch",             # IMPORTANT: save model so paper figures are reproducible
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=50,
        per_device_train_batch_size=exp["bs"],
        per_device_eval_batch_size=exp["bs"],
        num_train_epochs=exp["epochs"],
        learning_rate=exp["lr"],
        weight_decay=exp["wd"],
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
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # ---- Train
    trainer.train()

    # ---- Save loss curve + val f1 curve
    save_loss_curve(trainer.state.log_history, os.path.join(fig_dir, f"{name}_train_loss.png"))

    # ---- Predictions for VAL + TEST to build ROC/PR/CM
    val_pred = trainer.predict(dsd_tok["validation"])
    test_pred = trainer.predict(dsd_tok["test"])

    val_logits = val_pred.predictions
    val_y = val_pred.label_ids
    val_prob = softmax_2(val_logits)[:, 1]
    val_yhat = np.argmax(val_logits, axis=-1)

    test_logits = test_pred.predictions
    test_y = test_pred.label_ids
    test_prob = softmax_2(test_logits)[:, 1]
    test_yhat = np.argmax(test_logits, axis=-1)

    # ---- Save plots for VAL + TEST
    val_extra = save_roc_pr_cm(val_y, val_prob, val_yhat, fig_dir, prefix="val")
    test_extra = save_roc_pr_cm(test_y, test_prob, test_yhat, fig_dir, prefix="test")

    # ---- Trainer metrics
    val_metrics = trainer.evaluate(dsd_tok["validation"])
    test_metrics = trainer.evaluate(dsd_tok["test"])

    row = {
        "experiment": name,
        "model": model_name,
        "optimizer": opt,
        "epochs": exp["epochs"],
        "batch_size": exp["bs"],
        "lr": exp["lr"],
        "weight_decay": exp["wd"],
        "val_accuracy": float(val_metrics.get("eval_accuracy")),
        "val_f1": float(val_metrics.get("eval_f1")),
        "test_accuracy": float(test_metrics.get("eval_accuracy")),
        "test_f1": float(test_metrics.get("eval_f1")),
        "val_roc_auc": val_extra["roc_auc"],
        "val_ap": val_extra["ap"],
        "test_roc_auc": test_extra["roc_auc"],
        "test_ap": test_extra["ap"],
        "model_dir": model_out,
        "fig_dir": fig_dir
    }
    all_rows.append(row)

    # Save per-experiment meta (useful for paper + appendix)
    with open(os.path.join(fig_dir, "metrics.json"), "w") as f:
        json.dump({
            "trainer_val_metrics": val_metrics,
            "trainer_test_metrics": test_metrics,
            "extra_val": val_extra,
            "extra_test": test_extra,
            "config": exp
        }, f, indent=2)

    print(f"\n✅ Saved figures to: {fig_dir}")
    print("Files:", sorted([x for x in os.listdir(fig_dir) if x.endswith(".png")]))

# =============================
# 8) Save summary table
# =============================
df = pd.DataFrame(all_rows)
summary_csv = os.path.join(FIG_ROOT, "paper_figures_summary.csv")
summary_json = os.path.join(FIG_ROOT, "paper_figures_summary.json")
df.to_csv(summary_csv, index=False)
with open(summary_json, "w") as f:
    json.dump(all_rows, f, indent=2)

print("\n==================== FINAL SUMMARY ====================")
print(df[["experiment","val_accuracy","val_f1","val_roc_auc","test_accuracy","test_f1","test_roc_auc"]])
print("\nSaved summary:")
print("CSV :", summary_csv)
print("JSON:", summary_json)
print("\nAll figures root:", FIG_ROOT)

EOF