from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path


REQUIRED_SMTP_ENV = [
    "SMTP_HOST",
    "SMTP_PORT",
    "SMTP_USER",
    "SMTP_PASSWORD",
    "SMTP_FROM",
    "SMTP_TO",
]


@dataclass(frozen=True)
class EmailSendResult:
    sent: bool
    reason: str
    missing_env: list[str]


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def build_tuning_email(subject: str, body: str, attachments: list[str] | None = None) -> tuple[EmailMessage | None, list[str]]:
    missing = [name for name in REQUIRED_SMTP_ENV if not os.environ.get(name)]
    if missing:
        return None, missing
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ["SMTP_FROM"]
    msg["To"] = os.environ["SMTP_TO"]
    msg.set_content(body)
    for item in attachments or []:
        path = Path(item)
        if not path.exists() or not path.is_file():
            continue
        msg.add_attachment(
            path.read_bytes(),
            maintype="application",
            subtype="octet-stream",
            filename=path.name,
        )
    return msg, []


def send_tuning_email(subject: str, body: str, attachments: list[str] | None = None) -> EmailSendResult:
    msg, missing = build_tuning_email(subject, body, attachments)
    if missing or msg is None:
        reason = f"SMTP not configured, skip email notification. Missing: {', '.join(missing)}"
        print(f"[email-notify] {reason}", flush=True)
        return EmailSendResult(sent=False, reason=reason, missing_env=missing)
    try:
        host = os.environ["SMTP_HOST"]
        port = int(os.environ["SMTP_PORT"])
        use_tls = _truthy(os.environ.get("SMTP_USE_TLS", "true"))
        if use_tls:
            with smtplib.SMTP(host, port, timeout=30) as smtp:
                smtp.starttls()
                smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=30) as smtp:
                smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
                smtp.send_message(msg)
    except Exception as exc:
        reason = f"SMTP send failed: {type(exc).__name__}: {exc}"
        print(f"[email-notify] {reason}", flush=True)
        return EmailSendResult(sent=False, reason=reason, missing_env=[])
    return EmailSendResult(sent=True, reason="sent", missing_env=[])


__all__ = ["EmailSendResult", "build_tuning_email", "send_tuning_email"]
