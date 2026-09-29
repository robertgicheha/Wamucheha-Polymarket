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
    def send(
        self,
        message: str,
        severity: Severity = Severity.INFO,
        prefix: bool = True,
    ) -> None:
        """
        Dispatch to Telegram + Discord (email on CRITICAL).

        `prefix=False` suppresses the `[INFO]` banner for messages that are
        already self-describing — the periodic trade digest carries its own
        header, emoji and severity colouring, and a second machine label on
        top of that is noise.
        """
        self._send_telegram(message, severity, prefix)
        self._send_discord(message, severity, prefix)
        if severity == Severity.CRITICAL:
            self._send_email(message)

    def send_trade_digest(self, message: str) -> None:
        """Send a trade digest. Never prefixed — the digest has its own header."""
        self.send(message, Severity.INFO, prefix=False)

    def send_training(self, message: str, severity: Severity = Severity.INFO) -> None:
        """Send training mode notifications. Suppressed when TRAINING_MODE=false."""
        if not settings.training_mode:
            return
        self._send_telegram(message, severity, prefix=True)
        self._send_discord(message, severity, prefix=True)

    def _send_telegram(self, message: str, severity: Severity, prefix: bool = True) -> None:
        if not settings.telegram_bot_token or not settings.telegram_chat_id:
            return
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        text = f"[{severity.value.upper()}] {message}" if prefix else message
        try:
            requests.post(
                url,
                json={
                    "chat_id": settings.telegram_chat_id,
                    "text": text,
                },
                timeout=10,
            )
        except requests.RequestException as e:
            print(f"Telegram alert failed: {e}")

    def _send_discord(self, message: str, severity: Severity, prefix: bool = True) -> None:
        if not settings.discord_webhook_url:
            return
        content = f"**[{severity.value.upper()}]** {message}" if prefix else message
        try:
            requests.post(
                settings.discord_webhook_url,
                json={"content": content},
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
