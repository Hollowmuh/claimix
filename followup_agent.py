import os
import json
import time
from typing import Dict
from openai import OpenAI
from dotenv import load_dotenv

from utils import get_session_folder, load_json, save_json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FOLLOW_UP_SCHEMA = {
  "type": "object",
  "properties": {
    "email_html": {
      "type": "string",
      "description": "An HTML-formatted string containing a list of deduplicated follow-up questions."
    }
  },
  "required": ["email_html"],
  "additionalProperties": False
}

FOLLOW_UP_INSTRUCTION = """You are the Follow-Up Agent in an AI-powered automotive insurance claim system.

Your role is to help finalize the claim investigation by compiling a complete list of outstanding questions that have been asked by the specialist assistants (e.g., physical damage, third-party injury, theft, legal liability).

You will receive a structured input in the following format:

{
  "specialist_outputs": {
    "agent_1": { ... },
    "agent_2": { ... },
    ...
  }
}

Each agent’s data may include one or more of the following fields:
- "follow_up_question" (a single string)
- "follow_up_questions" (a list of strings)
- "questions" or "clarifying_questions" (lists)
- "notes" or free-text observations containing implicit questions

---

## 🔍 Your Task:

1. Extract all follow-up questions from the input JSON — including those in free-text fields like `notes`.
2. Normalize, deduplicate, and clean them:
   - Remove near-duplicates (e.g., "Was anyone injured?" vs "Were you or your passengers hurt?")
   - Ensure questions are well-formed, clear, and grammatically correct.
3. Format them into a numbered list with professional phrasing.
4. Return a single HTML-formatted string that begins with this prompt:

<b>To help us proceed with your claim, please respond to the following questions:</b><br><br>

Then follow it with each question on its own line using <br> line breaks:
1. Question one?<br>
2. Question two?<br>
...

---

## ✅ Output Format

Return exactly one JSON object with this structure:

{
  "email_html": "<b>To help us proceed with your claim...</b><br>1. ...<br>2. ...<br>3. ..."
}

Do not return anything outside of this JSON. No markdown, no prose explanations, no console-style output.

---

## ⚠️ Important Notes

- Only include follow-up questions. Do not include summaries or status notes.
- Combine similar questions into a single clear version.
- Avoid assistant names or metadata. Only show what the claimant needs to answer.
- Output must be professional, readable, and ready to send as an HTML email body.
"""

def run_follow_up_agent(email: str) -> Dict:
    folder = get_session_folder(email)
    follow_up_input_path = os.path.join(folder, "follow_up.json")

    # Load previously aggregated assistant data
    if not os.path.exists(follow_up_input_path):
        raise FileNotFoundError("follow_up.json not found.")

    follow_up_data = load_json(follow_up_input_path)
    specialist_outputs = follow_up_data.get("responses", {})

    if not specialist_outputs:
        raise ValueError("No specialist_outputs found in follow_up.json.")

    print(f"[follow_up] Loaded specialist_outputs from follow_up.json ({len(specialist_outputs)} agents).")

    # Call the OpenAI Responses API
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": FOLLOW_UP_INSTRUCTION},
            {"role": "user", "content": json.dumps({"specialist_outputs": specialist_outputs})}
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "FOLLOW_UP_QUESTIONS",
                "schema": FOLLOW_UP_SCHEMA,
                "strict": True
            }
        }
    )

    result = json.loads(response.output_text)
    print("[follow_up] Parsed response from assistant.")

    # Save email output
    follow_up_email_path = os.path.join(folder, "follow_up_email.json")
    save_json(follow_up_email_path, result)
    print("[follow_up] Saved follow-up email HTML.")

    # Send the email
    from advanced_imap_listener import send_email
    subject = "Further information required to process your claim"
    send_email(to=email, subject=subject, html=result["email_html"])
    print("[follow_up] Follow-up email sent.")

    # Reset follow_up.json
    save_json(follow_up_input_path, {
        "specialist_outputs": {}
    })
    print("[follow_up] follow_up.json has been reset.")

    return result
