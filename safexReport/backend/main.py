import os
import json
import shutil
from email.message import EmailMessage
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Literal

import tweepy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# ✅ ONE FASTAPI APP ONLY
# ============================================================
app = FastAPI(title="SafeX Guard - Cyberbullying API", version="1.2")

# ✅ CORS (Live Server = 5501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5501",
        "http://localhost:5501",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME = "sreeyashana4/bertweet-cyberbullying"
HF_TOKEN = os.getenv("HF_TOKEN", "").strip() or None

MAX_LEN = int(os.getenv("MAX_LEN", "128"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.80"))

HF_HOME = os.getenv("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
HF_HUB_DIR = os.path.join(HF_HOME, "hub")
MODEL_CACHE_DIR = os.path.join(HF_HUB_DIR, "models--sreeyashana4--bertweet-cyberbullying")
LOCKS_DIR = os.path.join(HF_HUB_DIR, ".locks")
TMP_DIR = os.path.join(HF_HUB_DIR, "tmp")

# ✅ local fallback json file
BAD_ACCOUNTS_PATH = os.path.join(os.path.dirname(__file__), "data", "bad_accounts.json")

tokenizer = None
model = None
device = None

# LABEL_1 = abusive
LABEL_MAP = {"LABEL_0": "NON_CYBERBULLYING", "LABEL_1": "CYBERBULLYING"}


# ============================================================
# Helpers
# ============================================================
def _safe_rmtree(path: str) -> None:
    try:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _load_model_once(force_download: bool, use_safetensors: Optional[bool]):
    global tokenizer, model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    mdl_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    tok_kwargs["force_download"] = force_download
    mdl_kwargs["force_download"] = force_download

    if use_safetensors is False:
        mdl_kwargs["use_safetensors"] = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, **tok_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, **mdl_kwargs).to(device)
    model.eval()


def load_model_robust():
    try:
        _load_model_once(force_download=False, use_safetensors=False)
        print("✅ Model loaded (prefer .bin)")
        return
    except Exception as e1:
        print("⚠️ Attempt 1 failed:", e1)

    _safe_rmtree(MODEL_CACHE_DIR)
    _safe_rmtree(LOCKS_DIR)
    _safe_rmtree(TMP_DIR)

    try:
        _load_model_once(force_download=True, use_safetensors=False)
        print("✅ Model loaded after cleanup (prefer .bin)")
        return
    except Exception as e2:
        print("⚠️ Attempt 2 failed:", e2)

    _safe_rmtree(MODEL_CACHE_DIR)
    _safe_rmtree(LOCKS_DIR)
    _safe_rmtree(TMP_DIR)

    _load_model_once(force_download=True, use_safetensors=None)
    print("✅ Model loaded after cleanup (allow safetensors)")


def _infer(text: str) -> Dict[str, Any]:
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(0)
        probs = F.softmax(logits, dim=-1)

    pred_id = int(torch.argmax(probs).item())
    confidence = float(probs[pred_id].item())
    raw_label = f"LABEL_{pred_id}"
    verdict = LABEL_MAP.get(raw_label, raw_label)

    return {
        "raw_label": raw_label,
        "verdict": verdict,
        "confidence": confidence,
        "probs": {
            "NON_CYBERBULLYING": float(probs[0].item()) if probs.shape[0] > 0 else 0.0,
            "CYBERBULLYING": float(probs[1].item()) if probs.shape[0] > 1 else 0.0,
        },
    }


def _flag(verdict: str, confidence: float) -> bool:
    return verdict == "CYBERBULLYING" and confidence >= THRESHOLD


def _get_bearer_token() -> str:
    bearer = (
        os.getenv("TWITTER_BEARER_TOKEN")
        or os.getenv("X_BEARER_TOKEN")
        or os.getenv("BEARER_TOKEN")
        or os.getenv("barrer_token")
    )
    if not bearer:
        raise RuntimeError("Missing TWITTER_BEARER_TOKEN (or X_BEARER_TOKEN / BEARER_TOKEN) in .env")
    return bearer


def _build_query(keyword: str) -> str:
    keyword = (keyword or "").strip()
    if keyword:
        return f'"{keyword}" -is:retweet lang:en'
    bad_words = ["fuck", "shit", "asshole", "idiot", "stupid", "ugly", "loser", "shut up", "go die"]
    return " OR ".join([f'"{w}"' for w in bad_words]) + " -is:retweet lang:en"


# ✅ NEW: fallback reader (bad_accounts.json -> rows)
def _fallback_rows_from_bad_accounts(limit: int) -> List[Dict[str, Any]]:
    if not os.path.exists(BAD_ACCOUNTS_PATH):
        return []

    data = json.loads(open(BAD_ACCOUNTS_PATH, "r", encoding="utf-8").read())
    accounts = data.get("accounts", [])

    rows: List[Dict[str, Any]] = []

    for acc in accounts:
        username = acc.get("username", "unknown")
        bad_tweets = acc.get("bad_tweets", [])
        for tw in bad_tweets:
            txt = tw.get("text", "")
            conf = float(tw.get("model", {}).get("confidence", 0.0))
            flagged = bool(tw.get("model", {}).get("flagged", False))

            rows.append({
                "tweet_id": str(tw.get("tweet_id", "")),
                "username": username,
                "tweet_text": txt,
                "prediction": "Cyberbullying" if flagged else "Not Cyberbullying",
                "confidence": round(conf, 4),
                "flagged": flagged
            })

            if len(rows) >= limit:
                return rows

    return rows


# ============================================================
# Schemas
# ============================================================
class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class PredictResponse(BaseModel):
    verdict: Literal["CYBERBULLYING", "NON_CYBERBULLYING"]
    confidence: float
    flagged: bool
    threshold: float
    probabilities: Dict[str, float]
    notes: List[str]


class LiveScanRequest(BaseModel):
    keyword: Optional[str] = ""
    count: int = Field(default=10, ge=10, le=50)


class LiveScanRow(BaseModel):
    tweet_id: str
    username: str
    tweet_text: str
    prediction: str
    confidence: float
    flagged: bool


class LiveScanResponse(BaseModel):
    total_fetched: int
    returned: int
    rows: List[LiveScanRow]
    source: Optional[str] = None  # ✅ NEW: "x_api" or "bad_accounts.json"
    note: Optional[str] = None


# ============================================================
# Startup
# ============================================================
@app.on_event("startup")
def startup():
    load_model_robust()
    print("✅ Startup complete")


# ============================================================
# Routes
# ============================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": str(device),
        "threshold": THRESHOLD,
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    res = _infer(text)
    verdict = res["verdict"]
    conf = float(res["confidence"])
    flagged = _flag(verdict, conf)

    notes = []
    notes.append("Detected abusive/cyberbullying content." if verdict == "CYBERBULLYING" else "No cyberbullying detected.")
    notes.append(f"Flag rule: verdict == CYBERBULLYING and confidence >= {THRESHOLD}")

    return PredictResponse(
        verdict=verdict,
        confidence=round(conf, 4),
        flagged=flagged,
        threshold=THRESHOLD,
        probabilities={k: round(v, 4) for k, v in res["probs"].items()},
        notes=notes,
    )


@app.post("/live-scan", response_model=LiveScanResponse)
def live_scan(req: LiveScanRequest):
    query = _build_query(req.keyword)
    max_results = int(req.count)

    # ✅ Try X API first
    try:
        bearer = _get_bearer_token()
        client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=False)

        resp = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=["author_id", "text"],
            expansions=["author_id"],
            user_fields=["username"],
        )

        if not resp.data:
            return LiveScanResponse(total_fetched=0, returned=0, rows=[], source="x_api")

        users_by_id = {}
        if resp.includes and "users" in resp.includes:
            for u in resp.includes["users"]:
                users_by_id[str(u.id)] = u

        rows: List[LiveScanRow] = []
        for t in resp.data:
            text = (t.text or "").strip()
            if not text:
                continue

            pred = _infer(text)
            verdict = pred["verdict"]
            conf = float(pred["confidence"])
            flagged = _flag(verdict, conf)

            u = users_by_id.get(str(t.author_id))
            username = getattr(u, "username", "unknown") if u else "unknown"

            rows.append(
                LiveScanRow(
                    tweet_id=str(t.id),
                    username=username,
                    tweet_text=text,
                    prediction=("Cyberbullying" if flagged else "Not Cyberbullying"),
                    confidence=round(conf, 4),
                    flagged=flagged,
                )
            )

            if len(rows) >= max_results:
                break

        return LiveScanResponse(
            total_fetched=len(resp.data),
            returned=len(rows),
            rows=rows,
            source="x_api"
        )

    except Exception as e:
        msg = str(e)

        # ✅ If rate limited OR any X failure: use local JSON
        fallback_rows = _fallback_rows_from_bad_accounts(max_results)

        if fallback_rows:
            return LiveScanResponse(
                total_fetched=len(fallback_rows),
                returned=len(fallback_rows),
                rows=[LiveScanRow(**r) for r in fallback_rows],
                source="bad_accounts.json",
                note=f"X API failed ({msg}). Served fallback data from bad_accounts.json"
            )

        # If fallback file not present, return real error
        raise HTTPException(status_code=500, detail=f"X scan failed and fallback not available: {msg}")
