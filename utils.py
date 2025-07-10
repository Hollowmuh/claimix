import os
import json
import hashlib
from typing import Dict, Any

PROCESSED_FILE = "processed_emails.json" 
DOCUMENT_EXTS = {".pdf", ".docx", ".jpg", ".png", ".jpeg", ".txt", ".doc", ".tiff", ".tif"}
SESSIONS_DIR = "sessions"
MAX_ATTACHMENT_SIZE = 10*1024*1024
def generate_thread_id(email: str) -> str:
    thread_id = hashlib.md5(email.lower().encode()).hexdigest()[:12]
    print(f"[generate_thread_id] Generated thread ID: {thread_id} for email: {email}")
    return thread_id

def get_session_folder(email: str) -> str:
    tid = generate_thread_id(email)
    folder = os.path.join(SESSIONS_DIR, f"thread_{tid}")
    os.makedirs(os.path.join(folder, "attachments"), exist_ok=True)
    print(f"[get_session_folder] Using session folder: {folder}")
    return folder

def get_claim_file(email: str) -> str:
    folder = get_session_folder(email)
    path = os.path.join(folder, "claim.json")
    if not os.path.exists(path):
        # initialize with empty structure
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stage": "NEW"}, f)
    return path

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_claim_state(email: str) -> Dict:
    with open(get_claim_file(email), "r") as f:
        return json.load(f)

def save_claim_state(email: str, state: Dict):
    with open(get_claim_file(email), "w") as f:
        json.dump(state, f, indent=2)

def is_document(att) -> bool:
    ext = os.path.splitext(att.filename or "")[1].lower()
    return ext in DOCUMENT_EXTS

def load_processed() -> set:
    if os.path.exists(PROCESSED_FILE):
        return set(json.load(open(PROCESSED_FILE)))
    return set()

def save_processed(uid: str):
    s = load_processed()
    s.add(str(uid))
    with open(PROCESSED_FILE, "w") as f:
        json.dump(list(s), f)