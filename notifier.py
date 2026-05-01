import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("TopoAlpha.Notifier")

class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_message(self, text):
        if not self.token or not self.chat_id:
            return
        try:
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML"
            }
            requests.post(self.base_url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"[TELEGRAM] Failed to send message: {e}")