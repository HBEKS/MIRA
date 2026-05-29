import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
print(f"Testing API Key: {API_KEY[:20]}...")

# Ganti dengan model yang tersedia
response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "deepseek/deepseek-v4-flash:free",  # Model gratis yang tersedia
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 5
    }
)

print(f"Status: {response.status_code}")
if response.status_code == 200:
    print("✅ SUCCESS! API key valid!")
    print(response.json())
else:
    print(f"❌ FAILED: {response.text}")