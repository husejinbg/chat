import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_URL = ""
MODEL = "codestral-latest"
MAX_TOKENS = 8192
HISTORY_DIR = Path("history")
ACTIVE_CHAT_FILE = HISTORY_DIR / "active_chat.json"
INPUT_FILE = Path("input.txt")
OUTPUT_FILE = Path("output.md")