import os
import logging

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("TopoAlpha.Notifier")


class TelegramNotifier:
    """Sends HTML-formatted trade alerts to a Telegram chat.

    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env.
    Silently no-ops when credentials are absent.
    """

    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self._url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_message(self, text: str) -> None:
        if not self.token or not self.chat_id:
            return
        try:
            requests.post(
                self._url,
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                timeout=5,
            )
        except Exception as exc:
            logger.error(f"[Telegram] {exc}")
