from dotenv import load_dotenv
load_dotenv()

import os
import requests

# one-way telegram messages: the bot sends to the user's chat, nothing is read back
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TELEGRAM_MAX_LEN = 4000   # telegram limit is 4096 characters per message


def send_message(text):
    if not TOKEN or not CHAT_ID:
        raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    # long reports are split into several messages
    for i in range(0, len(text), TELEGRAM_MAX_LEN):
        response = requests.post(url, data={"chat_id": CHAT_ID, "text": text[i:i + TELEGRAM_MAX_LEN]}, timeout=30)
        if not response.ok:
            # the url holds the token, so only report telegram's own error text
            raise RuntimeError(f"Telegram send failed: {response.status_code} {response.text}")


if __name__ == "__main__":
    send_message("hello from my stock bot")
    print("sent")
