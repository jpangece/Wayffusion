from __future__ import annotations

from utils.email_notify import REQUIRED_SMTP_ENV, build_tuning_email, send_tuning_email


def test_send_tuning_email_missing_env_does_not_raise(monkeypatch):
    for name in REQUIRED_SMTP_ENV:
        monkeypatch.delenv(name, raising=False)
    result = send_tuning_email("subject", "body")
    assert result.sent is False
    assert set(result.missing_env) == set(REQUIRED_SMTP_ENV)


def test_build_tuning_email_with_env(monkeypatch, tmp_path):
    values = {
        "SMTP_HOST": "smtp.example.test",
        "SMTP_PORT": "587",
        "SMTP_USER": "user",
        "SMTP_PASSWORD": "password",
        "SMTP_FROM": "from@example.test",
        "SMTP_TO": "to@example.test",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    attachment = tmp_path / "report.md"
    attachment.write_text("report", encoding="utf-8")
    msg, missing = build_tuning_email("subject", "body", [str(attachment)])
    assert missing == []
    assert msg is not None
    assert msg["Subject"] == "subject"
    assert msg["From"] == values["SMTP_FROM"]
    assert msg["To"] == values["SMTP_TO"]


def test_send_tuning_email_uses_smtp_ssl_for_port_465(monkeypatch):
    values = {
        "SMTP_HOST": "smtp.example.test",
        "SMTP_PORT": "465",
        "SMTP_USER": "user",
        "SMTP_PASSWORD": "password",
        "SMTP_FROM": "from@example.test",
        "SMTP_TO": "to@example.test",
        "SMTP_USE_SSL": "1",
        "SMTP_USE_TLS": "0",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    calls = []

    class FakeSMTPSSL:
        def __init__(self, host, port, timeout):
            calls.append(("connect", host, port, timeout))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def login(self, user, password):
            calls.append(("login", user, password))

        def send_message(self, message):
            calls.append(("send", message["Subject"]))

    monkeypatch.setattr("utils.email_notify.smtplib.SMTP_SSL", FakeSMTPSSL)
    result = send_tuning_email("formal started", "body")
    assert result.sent is True
    assert calls[0][:3] == ("connect", "smtp.example.test", 465)
    assert calls[-1] == ("send", "formal started")
