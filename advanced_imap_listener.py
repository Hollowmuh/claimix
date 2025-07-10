import os
import time
import smtplib
from email.message import EmailMessage
from imap_tools.query import AND
from imap_tools.mailbox import MailBox
from dotenv import load_dotenv

from utils import (
    get_session_folder,
    is_document,
    load_processed,
    save_processed,
    MAX_ATTACHMENT_SIZE
)
from updated_layer import orchestrate

load_dotenv()

IMAP_HOST     = os.getenv('IMAP_HOST')
IMAP_PORT     = int(os.getenv('IMAP_PORT', 993))
IMAP_USER     = os.getenv('IMAP_USERNAME')
IMAP_PASSWORD = os.getenv('IMAP_PASSWORD')
SMTP_HOST     = os.getenv('SMTP_HOST')
SMTP_PORT     = int(os.getenv('SMTP_PORT', 587))

SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)


def send_email(to: str, subject: str, html: str):
    """Send HTML email using SMTP."""
    msg = EmailMessage()
    msg["From"] = IMAP_USER
    msg["To"] = to
    msg["Subject"] = subject if subject.startswith("Re:") else f"Re: {subject}"
    msg.set_content("Please view in an HTML-capable email client.")
    msg.add_alternative(html, subtype="html")

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(IMAP_USER, IMAP_PASSWORD)
        s.send_message(msg)
    print(f"[send_email] Email sent to {to}")


def poll_inbox(interval=10):
    """Main loop to poll inbox for new messages and route to orchestrator."""
    print("[poll_inbox] Starting inbox polling loop...")
    processed = load_processed()
    print(f"[poll_inbox] Loaded {len(processed)} processed message UIDs.")

    while True:
        print("[poll_inbox] Connecting to mailbox...")
        with MailBox(IMAP_HOST).login(IMAP_USER, IMAP_PASSWORD, initial_folder="INBOX") as mb:
            print("[poll_inbox] Connected. Fetching unseen messages...")

            for msg in mb.fetch(AND(seen=False), mark_seen=True):
                uid = str(msg.uid)
                print(f"[poll_inbox] Processing message UID: {uid}")
                if uid in processed:
                    print(f"[poll_inbox] UID {uid} already processed. Skipping.")
                    continue

                sender = msg.from_ or ""
                subject = msg.subject or "No Subject"
                body = msg.text or msg.html or ""
                print(f"[poll_inbox] Message from: {sender}, subject: {subject}")

                # Ensure session folder exists
                session_folder = get_session_folder(sender)
                os.makedirs(os.path.join(session_folder, "attachments"), exist_ok=True)

                # Save attachments
                print("[poll_inbox] Saving attachments...")
                attachments = []
                for att in msg.attachments:
                    if not is_document(att) or att.size > MAX_ATTACHMENT_SIZE:
                        print(f"[poll_inbox] Skipping attachment {att.filename} (not document or too large)")
                        continue
                    safe_name = att.filename.replace("/", "_")
                    path = os.path.join(session_folder, "attachments", safe_name)
                    with open(path, "wb") as f:
                        f.write(att.payload)
                    print(f"[poll_inbox] Saved attachment: {safe_name}")
                    attachments.append(safe_name)

                # Hand off to orchestration layer
                print("[poll_inbox] Handing off to orchestration layer...")
                try:
                    orchestrate(
                        email=sender,
                        user_message=body,
                        attachments=attachments
                    )
                except Exception as e:
                    print(f"[poll_inbox] ERROR during orchestration: {e}")

                # Mark message UID as processed
                save_processed(uid)
                print(f"[poll_inbox] Marked UID {uid} as processed.")

        print(f"[poll_inbox] Sleeping for {interval} seconds before next poll...")
        time.sleep(interval)


if __name__ == "__main__":
    print("[main] Starting poll_inbox()")
    poll_inbox()
