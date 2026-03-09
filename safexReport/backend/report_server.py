import os
import json
import smtplib
from pathlib import Path
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# ============================================================
# ✅ Load .env ONLY from this backend folder
# ============================================================
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH)


# ============================================================
# ✅ FastAPI App
# ============================================================
app = FastAPI(title="SafeX Guard - Report API", version="1.2")

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
        "null",  # sometimes file:// origins show as null
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ✅ Schema
# ============================================================
class ReportPayload(BaseModel):
    case_id: str = Field(min_length=1, max_length=60)
    username: str = Field(min_length=1, max_length=60)
    tweet_id: str = Field(min_length=1, max_length=80)
    tweet_text: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0)
    risk: Literal["Low", "Medium", "High"]


# ============================================================
# ✅ Helpers
# ============================================================
def _require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}. Check {ENV_PATH}")
    return v


def _append_local_log(payload: dict, status: str, error: Optional[str] = None) -> None:
    """
    Always save locally: reports_local.jsonl (1 JSON per line)
    """
    out_path = Path(__file__).with_name("reports_local.jsonl")
    record = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "payload": payload,
        "error": error,
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def send_email_via_gmail_smtp(to_email: str, subject: str, body: str) -> None:
    """
    ✅ STARTTLS (587) is the most reliable method for Gmail SMTP
    """
    sender = _require_env("GMAIL_SENDER")
    app_password = _require_env("GMAIL_APP_PASSWORD")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(sender, app_password)
        smtp.send_message(msg)


# ============================================================
# ✅ Routes
# ============================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "report",
        "env_loaded_from": str(ENV_PATH),
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/debug-env")
def debug_env():
    """
    ✅ Open this in browser to confirm .env loaded correctly
    http://127.0.0.1:8001/debug-env
    """
    sender = os.getenv("GMAIL_SENDER", "")
    pwd = os.getenv("GMAIL_APP_PASSWORD", "")
    return {
        "env_path": str(ENV_PATH),
        "gmail_sender": sender,
        "gmail_app_password_len": len(pwd),
    }


@app.post("/send-report")
def send_report(payload: ReportPayload):
    """
    Sends email to receiver. If email fails, saves locally and returns status=saved.
    """
    receiver = os.getenv("REPORT_RECEIVER", "sreeyashana4@gmail.com").strip()

    subject = f"[SafeX Guard] Cyberbullying Report — {payload.case_id}"
    body = "\n".join(
        [
            "Hello,",
            "",
            "A cyberbullying case has been detected and reviewed.",
            "",
            "--- Case Details ---",
            f"Case ID: {payload.case_id}",
            f"Account: {payload.username}",
            f"Tweet ID: {payload.tweet_id}",
            f"Risk Level: {payload.risk}",
            f"Confidence: {payload.confidence:.2f}",
            "",
            "Tweet Text:",
            f"\"{payload.tweet_text}\"",
            "",
            "Regards,",
            "SafeX Guard System",
        ]
    )

    _append_local_log(payload.model_dump(), status="received")

    try:
        send_email_via_gmail_smtp(receiver, subject, body)
        _append_local_log(payload.model_dump(), status="email_sent")
        return {"status": "success", "message": f"Email sent to {receiver}"}

    except RuntimeError as e:
        _append_local_log(payload.model_dump(), status="failed_env", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        _append_local_log(payload.model_dump(), status="email_failed_saved_locally", error=str(e))
        return {
            "status": "saved",
            "message": "Email failed, but report was saved locally (reports_local.jsonl).",
            "error": str(e),
        }
