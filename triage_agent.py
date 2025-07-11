import os
import json
import time
from typing import Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from utils import get_session_folder, load_json, get_claim_file, save_json
# -----------------------------------------------------------------------------
# Configuration & Helpers
# -----------------------------------------------------------------------------
load_dotenv()
TRIAGE_ASSISTANT_ID = os.getenv("TRIAGE_ASSISTANT_ID")  # Set this in your env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------------------------------------------------------
# Triage Runner
# -----------------------------------------------------------------------------

def run_triage(email: str, conversation_context: Optional[str]) -> Dict[str, Any]:
    """
   
    4) Saves 'incident_types' into claim.json (and updates stage).
    Returns the updated claim dict.
    """
    print(f"[run_triage] Starting triage for: {email}")
    folder = get_session_folder(email)
   
    # Build single user message containing both inputs
    user_content = [
        {
            "type": "text",
            "text": json.dumps({
                "conversation_context": conversation_context
            })
        }
    ]
    print("[run_triage] Built user content for assistant.")

    # 1) Create thread
    print("[run_triage] Creating OpenAI thread...")
    thread = client.beta.threads.create(
        messages=[{
            "role": "user",
            "content": user_content
        }]
    )
    print(f"[run_triage] Thread created with ID: {thread.id}")

    # 2) Dispatch the triage assistant
    print("[run_triage] Dispatching triage assistant...")
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=TRIAGE_ASSISTANT_ID
    )
    print(f"[run_triage] Run started with ID: {run.id}")

    # 3) Poll until done
    print("[run_triage] Polling for triage assistant completion...")
    while run.status not in ("completed", "failed"):
        print(f"[run_triage] Current run status: {run.status}")
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    print(f"[run_triage] Final run status: {run.status}")
    if run.status != "completed":
        print(f"[run_triage] ERROR: Triage run failed with status: {run.status}")
        raise RuntimeError(f"Triage run failed: {run.status}")

    # 4) Extract the JSON output from the final message
    incident_types = None
    incident_description = None
    print("[run_triage] Retrieving messages from thread...")
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    for msg in reversed(messages.data):
        if msg.role == "assistant":
            text_obj = msg.content[0].text
            text = text_obj.value if hasattr(text_obj, "value") else str(text_obj)
            if not text.strip():
                print("[run_triage] Skipping empty assistant message.")
                continue
            print("[run_triage] Assistant response:", text)
            try:
                parsed = json.loads(text)
                incident_types = parsed.get("parameters", {}).get("incident_type")
                incident_description = parsed.get("parameters", {}).get("incident_description")
                print(f"[run_triage] Parsed incident_types: {incident_types}, Incident Description: {incident_description}")
                if not incident_types:
                    print("[run_triage] ERROR: Triage assistant did not return incident_type")
                    raise RuntimeError("Triage assistant did not return incident_type")
            except json.JSONDecodeError as e:
                print(f"[run_triage] JSON decode error: {e}")
                continue  # skip messages that are not valid JSON

    if incident_types is None or incident_description is None:
        print("[run_triage] ERROR: Triage assistant did not return incident type or incident desctiption")
        raise RuntimeError("Triage assistant did not return incident_type or description")

    # 5) Save to claim.json
    claim_file = get_claim_file(email)
    print(f"[run_triage] Saving incident_types to claim file: {claim_file}")
    claim = load_json(claim_file)
    claim["incident_types"] = incident_types
    claim["incident_description"] = incident_description
    claim["stage"] = "TRIAGED"
    save_json(claim_file, claim)
    print("[run_triage] Claim updated and saved.")

    return claim
