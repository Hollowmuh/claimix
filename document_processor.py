import os
import json
import re
import logging
import pytesseract
import time
import argparse
from typing import Dict, List, Tuple, Any
from PIL import Image
from pdf2image import convert_from_path
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FNOL_FIELDS = [
    "Full Name", "Date",  "Policy Number",
    "Phone Number","Date of Incident", "Time of Incident", 
    "Location of Incident", "Description of the Incident"
]

FIELD_KEY_MAP = {
    "Full Name": "full_name",
    "Date": "date",
    "Policy Number": "policy_number",
    "Phone Number": "phone_number",
    "Date of Incident": "incident_date",
    "Time of Incident": "incident_time",
    "Location of Incident": "location",
    "Description of the Incident": "description"
}

VALIDATION_PATTERNS = {
    "phone_number": r'^\+?[\d\s\-\(\)]{7,15}$',
    "policy_number": r'^[A-Z0-9\-]{5,20}$',
    "incident_date": r'^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}$|^\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}$',
    "date": r'^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}$|^\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}$',
    "incident_time": r'^\d{1,2}:\d{2}(\s?(AM|PM))?$'
}

SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
PDF_EXT = '.pdf'

class ProcessingResult:
    def __init__(self):
        self.extracted_fields: Dict[str, str] = {}
        self.validation_errors: Dict[str, str] = {}
        self.processing_errors: List[str] = []
        self.success: bool = False
    def to_dict(self) -> Dict[str, Any]:
        return {
            'extracted_fields': self.extracted_fields,
            'validation_errors': self.validation_errors,
            'processing_errors': self.processing_errors,
            'success': self.success,
            'extracted_count': len(self.extracted_fields),
            'error_count': len(self.validation_errors) + len(self.processing_errors)
        }


def process_image(image_path: str) -> dict:
    img = Image.open(image_path)
    import tempfile
    temp_pdf = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            temp_pdf = tmp.name
            img.save(temp_pdf, format='PDF')
        pdf_images = convert_from_path(temp_pdf)
        if not pdf_images:
            raise RuntimeError("Failed to convert PDF back to image for OCR.")
        img = pdf_images[0]
        output_a = pytesseract.image_to_string(img, config='--oem 3 --psm 3')
        output_b = pytesseract.image_to_string(img, config='--oem 3 --psm 12')
        return {"output_a": output_a, "output_b": output_b}
    finally:
        if temp_pdf and os.path.exists(temp_pdf):
            os.remove(temp_pdf)

def extract_text_from_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        return process_image(file_path)
    elif ext == PDF_EXT:
        for attempt in range(4):
            try:
                logger.info(f"OCR attempt{attempt} for {file_path}")
                text_pages = []
                pages = convert_from_path(file_path, dpi=300)
                for page in pages:
                    text_pages.append(pytesseract.image_to_string(page))
                return "\n".join(text_pages)
            except Exception as e:
                logger.error(f"OCR failed on attempt {attempt}: {e}")
                if attempt < 3:
                    time.sleep(1)
                    continue
                else:
                    raise
    else:
        raise ValueError(f'Unsupported extension: {ext}')

def extract_multi_line_value(lines: List[str], start_idx:int, field_name: str) -> str:
    pattern = re.escape(field_name) + r"\s*[:\-]?\s*(.+)"
    m = re.search(pattern, lines[start_idx], re.IGNORECASE)
    parts = [m.group(1).strip()] if m else []
    if field_name.lower().startswith('description'):
        for l in lines[start_idx+1 : start_idx+7]:
            if not l.strip():
                break
            if any(re.match(re.escape(f)+r"\s[:\-]", l, re.IGNORECASE) for f in FNOL_FIELDS):
                break
    return ''.join(parts)
def validate_field_value(key: str, val:str) -> Tuple[bool, str]:
    if not val: 
        return True, ''
    if key in VALIDATION_PATTERNS:
        if not re.match(VALIDATION_PATTERNS[key], val):
            return False, f"Invalid format for {key}: '{val}'"
    return True, ''
def parse_fnol_text(text: str) -> ProcessingResult:
    res = ProcessingResult()
    res.extracted_fields = {"text": text}
    res.success = bool(text.strip())
    return res

def retry_read_json(path:str) -> Any:
    for attempt in range(1,4):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Read Json failed attempt {attempt}: {e}")
            time.sleep(1)
    raise

def generate_thread_id(sender_email):
    return hashlib.md5(sender_email.lower().encode('utf-8')).hexdigest()[:12]

def process_and_update_claim_session(sender: str) -> dict:
    thread_id = generate_thread_id(sender)
    base = 'sessions'
    tf = os.path.join(base, f"thread_{thread_id}", 'attachments')
    if not os.path.isdir(tf):
        raise FileNotFoundError(tf)
    parsed_file = os.path.join(base, f"thread_{thread_id}", 'parsed_docs.json')
    parsed = {} if not os.path.exists(parsed_file) else json.load(open(parsed_file))

    for fname in os.listdir(tf):
        if fname in parsed:
            continue
        try:
            file_path = os.path.join(tf, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_IMAGE_EXTENSIONS:
                ocr_outputs = process_image(file_path)
                parsed[fname] = {
                    "output_a": ocr_outputs.get("output_a", ""),
                    "output_b": ocr_outputs.get("output_b", ""),
                    "success": bool(ocr_outputs.get("output_a", "").strip() or ocr_outputs.get("output_b", "").strip())
                }
            elif ext == PDF_EXT:
                text = extract_text_from_file(file_path)
                if not text.strip():
                    parsed[fname] = {'error': 'No text', 'success': False}
                    continue
                res = parse_fnol_text(text)
                parsed[fname] = res.to_dict()
            else:
                parsed[fname] = {'error': 'Unsupported file type', 'success': False}
        except Exception as e:
            parsed[fname] = {'error': str(e), 'success': False}

    with open(parsed_file, 'w') as f:
        json.dump(parsed, f, indent=2)
    return parsed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thread', required=True)
    args = parser.parse_args()
    print(json.dumps(process_and_update_claim_session(args.thread), indent=2))
