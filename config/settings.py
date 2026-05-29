import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

class Settings:
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    
    def __init__(self):
        if not self.OPENROUTER_API_KEY or self.OPENROUTER_API_KEY.strip() == "":
            raise ValueError(f"❌ MIRA CONFIG ERROR: OPENROUTER_API_KEY kosong di file .env!")

settings = Settings()