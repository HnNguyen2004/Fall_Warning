from telegram import Bot
import asyncio
import os
from dotenv import load_dotenv

# Nạp biến môi trường từ file .env
load_dotenv()

async def send_telegram(photo_path="alert.png"):
    try:
        my_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not my_token or not chat_id:
            print("⚠️ Telegram token hoặc chat_id chưa được cấu hình")
            return False
        
        bot = Bot(token=my_token)
        
        # Kiểm tra file tồn tại
        if not os.path.exists(photo_path):
            print(f"⚠️ File {photo_path} không tồn tại")
            return False
            
        with open(photo_path, "rb") as photo:
            await bot.send_photo(chat_id=chat_id, photo=photo, caption="🚨 Có xâm nhập, nguy hiểm!")
        
        print("✓ Gửi Telegram thành công")
        return True
        
    except Exception as ex:
        print(f"✗ Không thể gửi Telegram: {ex}")
        return False
