"""
Alert dispatch. Telegram + Discord for routine/monitoring alerts, email reserved
for critical failures (bot halted, premature shutdown, wallet anomaly) since email
is the slowest channel but the one most likely to actually get seen if the bot
is down and Telegram/Discord aren't being watched live.
"""
import smtplib
from email.mime.text import MIMEText
from enum import Enum

import requests

from config.settings import settings


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Notifier:
    def send(self, message: str, severity: Severity = Severity.INFO) -> None:
        self._send_telegram(message, severity)
        self._send_discord(message, severity)
        if severity == Severity.CRITICAL:
            self._send_email(message)

    def _send_telegram(self, message: str, severity: Severity) -> None:
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            return
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        try:
            requests.post(
                url,
                json={
                    "chat_id": settings.telegram_chat_id,
                    "text": f"[{severity.value.upper()}] {message}",
                },
                timeout=10,
            )
        except requests.RequestException as e:
            print(f"Telegram alert failed: {e}")

    def _send_discord(self, message: str, severity: Severity) -> None:
        if not settings.discord_webhook_url:
            return
        try:
            requests.post(
                settings.discord_webhook_url,
                json={"content": f"**[{severity.value.upper()}]** {message}"},
                timeout=10,
            )
        except requests.RequestException as e:
            print(f"Discord alert failed: {e}")

    def _send_email(self, message: str) -> None:
        if not settings.alert_email_smtp_host or not settings.alert_email_to:
            return
        msg = MIMEText(message)
        msg["Subject"] = "[CRITICAL] Polymarket bot alert"
        msg["From"] = settings.alert_email_from
        msg["To"] = settings.alert_email_to
        try:
            with smtplib.SMTP(
                settings.alert_email_smtp_host, settings.alert_email_smtp_port
            ) as server:
                server.starttls()
                server.login(settings.alert_email_from, settings.alert_email_password)
                server.send_message(msg)
        except Exception as e:
            print(f"Email alert failed: {e}")


notifier = Notifier()
