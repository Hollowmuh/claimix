import os
import json
import base64
import hashlib
import time
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
from pdf2image import convert_from_path
# from PIL import Image
from document_processor import process_and_update_claim_session

from utils import generate_thread_id

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SESSIONS_DIR = "sessions"
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
PDF_EXT = '.pdf'

# JSON schema for attachment details
ATTACHMENT_DETAILS_SCHEMA = {
    "type": "object",
    "properties": {
        "attachment_details": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "details": {"type": "string"}
                },
                "required": ["name", "details"],
                "additionalProperties": False
            }
        }
    },
    "required": ["attachment_details"],
    "additionalProperties": False
}

SYSTEM_INSTRUCTION = """
You are the Attachment Details Assistant. Your only task is to take each filename in 'attachments'
along with its OCR outputs ('output_a', 'output_b', or 'text'), and produce for each attachment a 'details' string:
- Only use the OCR outputs as context, Do not save as a seperate entry.
- If no OCR text is available, use the image itself to generate a vivid description of what is visible (colors,
  objects, damage, context).

Return exactly one JSON object conforming to the schema provided. Do not ask questions or perform any classification.
"""


def encode_image(filepath: str) -> str:
    print(f"[encode_image] Encoding image: {filepath}")
    with open(filepath, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    print(f"[encode_image] Finished encoding image: {filepath}")
    return encoded

def get_image_inputs(folder: str, attachments: List[str]) -> List[Dict]:
    print(f"[get_image_inputs] Preparing image inputs from folder: {folder}")
    inputs = []
    for fname in attachments:
        path = os.path.join(folder, "attachments", fname)
        print(f"[get_image_inputs] Checking file: {path}")
        if not os.path.exists(path):
            print(f"[get_image_inputs] File does not exist: {path}")
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_IMAGE_EXTENSIONS:
            print(f"[get_image_inputs] File is supported image: {fname}")
            b64 = encode_image(path)
            inputs.append({
                "type": "input_image",
                "image_url": f"data:image/{ext[1:]};base64,{b64}"
            })
        elif ext == PDF_EXT:
            print(f"[get_image_inputs] File is PDF: {fname}, converting to images")
            pages = convert_from_path(path, dpi=200)
            for i, page in enumerate(pages):
                tmp = f"{path}_page_{i}.jpg"
                page.save(tmp, "JPEG")
                print(f"[get_image_inputs] Saved PDF page as image: {tmp}")
                b64 = encode_image(tmp)
                inputs.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}"
                })
                os.remove(tmp)
                print(f"[get_image_inputs] Removed temporary image: {tmp}")
    print(f"[get_image_inputs] Prepared {len(inputs)} image inputs")
    return inputs

def generate_attachment_details(
    sender_email: str,
    attachments: List[str]
) -> Dict:
    print(f"[generate_attachment_details] Starting for sender: {sender_email} with attachments: {attachments}")
    session_folder = os.path.join(SESSIONS_DIR, f"thread_{generate_thread_id(sender_email)}")
    os.makedirs(os.path.join(session_folder, "attachments"), exist_ok=True)
    print(f"[generate_attachment_details] Session folder: {session_folder}")

    # 1) Run OCR and get parsed_docs
    print("[generate_attachment_details] Running OCR and processing documents...")
    parsed_docs = process_and_update_claim_session(sender_email)
    print(f"[generate_attachment_details] OCR and document processing complete. Parsed docs: {list(parsed_docs.keys())}")

    # 2) Build user content blocks
    user_blocks = []
    for fname in attachments:
        doc = parsed_docs.get(fname, {})
        text = doc.get("output_a") or doc.get("output_b") or doc.get("text") or ""
        if text.strip():
            print(f"[generate_attachment_details] Adding OCR text for: {fname}")
            user_blocks.append({
                "type": "input_text",
                "text": f"{fname} OCR:\n{text.strip()[:1000]}"
            })
        else:
            print(f"[generate_attachment_details] No OCR text for: {fname}")
    # include images as base64
    print("[generate_attachment_details] Adding image inputs...")
    user_blocks.extend(get_image_inputs(session_folder, attachments))

    # 3) Call Responses API
    print("[generate_attachment_details] Calling OpenAI Responses API...")
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_blocks}
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "ATTACHMENT_DETAILS",
                "schema": ATTACHMENT_DETAILS_SCHEMA,
                "strict": True
            }
        }
    )
    print("[generate_attachment_details] Received response from OpenAI API.")
    result = json.loads(response.output_text)
    print(f"[generate_attachment_details] Parsed response: {json.dumps(result, indent=2)}")

    # 4) Save to attachment_data.json
    out_path = os.path.join(session_folder, "attachment_data.json")
    print(f"[generate_attachment_details] Saving results to: {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("[generate_attachment_details] Results saved successfully.")

    return result

# Example usage
if __name__ == "__main__":
    sender = "user@example.com"
    attachments = ["photo.jpg", "invoice.pdf"]
    os.makedirs(os.path.join(SESSIONS_DIR, f"thread_{generate_thread_id(sender)}", "attachments"), exist_ok=True)
    print("[main] Running generate_attachment_details...")
    details = generate_attachment_details(sender, attachments)
    print("[main] Attachment details generated:")
    print(json.dumps(details, indent=2))
