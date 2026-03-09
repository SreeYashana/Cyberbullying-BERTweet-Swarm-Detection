import os
import smtplib
from email.message import EmailMessage
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class ReportPayload(BaseModel):
    case_id: str = Field(min_length=3, max_length=60)
    username: str = Field(min_length=1, max_length=40)
    tweet_id: str = Field(min_length=1, max_length=60)
    tweet_text: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0)
    risk: Literal["Low", "Medium", "High"]


def _require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def send_email_via_gmail_smtp(to_email: str, subject: str, body: str) -> None:
    sender = _require_env("GMAIL_SENDER")
    app_password = _require_env("GMAIL_APP_PASSWORD")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, app_password)
        smtp.send_message(msg)


app = FastAPI(title="SafeX Guard - Report API", version="1.0")

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


@app.get("/health")
def health():
    return {"status": "ok", "service": "report"}


@app.post("/send-report")
def send_report(payload: ReportPayload):
    receiver = "sreeyashana4@gmail.com"  # keep fixed as you want

    subject = f"[SafeX Guard] Cyberbullying Report — {payload.case_id}"

    safe_text = (payload.tweet_text or "").replace("\r", " ").strip()

    body = "\n".join([
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
        f"\"{safe_text}\"",
        "",
        "Regards,",
        "SafeX Guard System",
    ])

    try:
        send_email_via_gmail_smtp(receiver, subject, body)
        return {"status": "success", "message": f"Email sent to {receiver}"}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email send failed: {e}")
