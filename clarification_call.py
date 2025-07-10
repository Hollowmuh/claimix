import os
import json
import hashlib
from typing import Dict
from openai import OpenAI
from dotenv import load_dotenv

from utils import get_session_folder, load_json, get_claim_file, save_json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SESSIONS_DIR = "sessions"

CLARIFY_SCHEMA = {
  "type": "object",
  "properties": {
    "expanded_incident_description": {
      "type": "string",
      "description": "A faithful, detailed restatement of everything the user said or that OCR and attachment details revealed."
    },
    "clarifying_question": {
      "type": "string",
      "description": "A single open‑ended question asking for the most critical missing context."
    }
  },
  "required": ["expanded_incident_description", "clarifying_question"],
  "additionalProperties": False
}

CLARIFY_INSTRUCTION =''' 
You are the Clarifying Question Assistant for an automotive insurance claim system. Your role is to analyze the user's initial message describing an incident involving their vehicle, as well as structured "attachment details". You will use this information to help prepare the claim for routing to specialist assistants by filling in missing information that the user has not yet provided.

You must perform **two specific actions only**:

1. expanded_incident_description: Faithfully restate what the user has already reported. Use their own language wherever possible, and supplement it with concise summaries of the attachment “details.” Do not invent or assume anything not explicitly stated. This should read as a brief, coherent narrative that includes all currently known facts.

2. clarifying_question: Based on what is known, identify the **most likely one or more categories of incident** (see assistant types below), and generate **a single, well-structured follow-up prompt** that elicits any critical missing information. The question may include multiple sub-parts if needed but should flow naturally and remain focused.

You may phrase this as:
  “To help us fully understand the situation and route your claim correctly, could you please clarify: [detailed, open-ended clarification based on likely incident types]”

⚠️ Do not classify the incident or assign it to specific categories directly. Instead, infer which types are probable and craft the question accordingly.

---

🚗 Specialist Assistant Modules and Scopes

You are expected to understand the following module structure so you can tailor your clarifying question appropriately. Each module addresses a different aspect of the claim:

### **Module 1 – Physical Loss & Damage**
- **accidental_and_glass_damage**: Collisions, parking knocks, vandalism, glass/windscreen damage, wrong fuel.
- **fire**: Fire, lightning, explosions.
- **theft**: Stolen vehicles, attempted theft, recovered vehicles.
- **ancillary_property**: Damage to accessories like child seats, in-car electronics, charging cables, roof boxes.

### **Module 2 – Third-Party Liability and Legal Exposure**
- **third_party_injury**: Bodily injury to others, emergency care, fatalities.
- **third_party_property**: Damage to another person’s car, building, fence, or structure.
- **special_liability**: Unusual cases like towing, autonomous mode, driving other people’s cars, non-public locations.
- **legal_and_statutory**: Solicitor or legal costs, inquest-related expenses, statutory RTA payments.

### **Module 3 – Personal Protection and Convenience**
- **personal_injury**: Injury to the claimant or passengers, medical treatment, assault.
- **personal_convenience**: Rental vehicle while repairing, continuation of journey.
- **personal_property**: Loss/damage to personal belongings inside the car.

### **Module 4 – Policy Governance and Eligibility**
- **territorial_usage**: Whether the vehicle was in a covered location or used for covered purposes.
- **general_exceptions**: Uncovered causes like war, terrorism, intoxication, cyber incidents.
- **vehicle_security**: Tracker installed, roadworthiness, MOT, ADAS features.
- **administrative**: Payment history, no-claim discount, proof of identity/address.

---

## 🔁 Workflow

1. Parse the user's initial message and attachment details.
2. Identify what facts are already known (what, where, who, how, and when).
3. Think: What modules are *probably* relevant?
4. Ask for any **crucial missing facts** that would help clarify which path the claim should follow.
5. Your output is a single JSON object with two fields:
   - `expanded_incident_description`: a concise summary of what’s known so far.
   - `clarifying_question`: a professionally written question that seeks missing details and encourages a complete response.
'''



def load_attachment_data(sender_email: str) -> list:
    folder = get_session_folder(sender_email)
    path = os.path.join(folder, "attachment_data.json")
    print(f"[load_attachment_data] Looking for attachment data at: {path}")
    if os.path.exists(path):
        data = json.load(open(path, "r"))
        print(f"[load_attachment_data] Loaded attachment data with {len(data.get('attachment_details', []))} entries")
        return data.get("attachment_details", [])
    print("[load_attachment_data] No attachment data found.")
    return []

def run_clarifying_question(
    sender_email: str,
    message_text: str
) -> Dict:
    """
    Runs the one-time clarifying question using the user's message
    and previously saved attachment_data.json, then saves the result
    to prelim_data.json in the session folder.
    """
    print(f"[run_clarifying_question] Starting clarifying question for: {sender_email}")
    # 1) load attachment details
    attachment_data = load_attachment_data(sender_email)  # Now a list

    # 2) compile attachment summary
    attachment_summary = ""
    for attachment in attachment_data:
        fname = attachment.get("name", "unknown")
        details = attachment.get("details", "")
        if details:
            print(f"[run_clarifying_question] Adding details for attachment: {fname}")
            attachment_summary += f"\n\n[{fname}]\n{details.strip()[:1000]}"

    # 3) build user content blocks
    print("[run_clarifying_question] Building user content blocks...")
    user_blocks = [{"type": "input_text", "text": message_text}]
    if attachment_summary:
        print("[run_clarifying_question] Including attachment summary in user blocks.")
        user_blocks.append({
            "type": "input_text",
            "text": "Attachment Details:\n" + attachment_summary.strip()
        })

    # 4) call the Responses API
    print("[run_clarifying_question] Calling OpenAI Responses API...")
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": CLARIFY_INSTRUCTION},
            {"role": "user",   "content": user_blocks}
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "CLARIFY_INCIDENT",
                "schema": CLARIFY_SCHEMA,
                "strict": True
            }
        }
    )
    print("[run_clarifying_question] Received response from OpenAI API.")
    result = json.loads(response.output_text)
    print(f"[run_clarifying_question] Parsed response: {json.dumps(result, indent=2)}")

    # 5) save output to prelim_data.json
    folder = get_session_folder(sender_email)
    claim_path = os.path.join(folder, "claim.json")
    print(f"[run_clarifying_question] Updating incident_description in: {claim_path}")    
    # Load existing claim or initialize new
    claim = load_json(claim_path) if os.path.exists(claim_path) else {}
    # Update only the incident description
    claim["incident_description"] = result["expanded_incident_description"]
    # Save updated claim.json
    with open(claim_path, "w", encoding="utf-8") as f:
        json.dump(claim, f, indent=2, ensure_ascii=False)
        print("[orchestrate] Saved expanded incident description.")
    
    from advanced_imap_listener import send_email
        # Send clarifying question via email (HTML formatted)
    subject = "Quick clarification needed to process your claim"
    html_body = (
        "<p>Thanks for reporting your incident. Based on the information so far, we need a quick clarification to route your claim appropriately.</p>"
        "<p><b>Please reply with the following:</b></p>"
        f"<p>{result['clarifying_question']}</p>"
    )
    send_email(to=sender_email, subject=subject, html=html_body)
    print("[orchestrate] Clarifying question sent via email.")
    print("[run_clarifying_question] incident_description saved to claim.json")
