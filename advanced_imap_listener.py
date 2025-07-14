import os
import time
import smtplib
import ssl
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
from orchestrator import orchestrate

load_dotenv()

IMAP_HOST     = os.getenv('IMAP_HOST')
IMAP_PORT     = int(os.getenv('IMAP_PORT', 993))
IMAP_USER     = os.getenv('IMAP_USERNAME')
IMAP_PASSWORD = os.getenv('IMAP_PASSWORD')
SMTP_HOST     = os.getenv('SMTP_HOST')
SMTP_PORT     = int(os.getenv('SMTP_PORT', 587))

SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)


def send_email(to: str, subject: str, html: str, max_retries=3):
    """
    Send HTML email using SMTP with SSL/TLS handling and retry logic.
    
    Args:
        to (str): Recipient email address
        subject (str): Email subject
        html (str): HTML content
        max_retries (int): Maximum number of retry attempts
    
    Returns:
        dict: Success information including method used, or None if failed
    """
    msg = EmailMessage()
    msg["From"] = IMAP_USER
    msg["To"] = to
    msg["Subject"] = subject if subject.startswith("Re:") else f"Re: {subject}"
    msg.set_content("Please view in an HTML-capable email client.")
    msg.add_alternative(html, subtype="html")

    # Different connection methods to try
    connection_methods = [
        {
            "name": "STARTTLS",
            "port": SMTP_PORT,
            "use_ssl": False,
            "use_starttls": True
        },
        {
            "name": "SSL",
            "port": 465,  # Common SSL port
            "use_ssl": True,
            "use_starttls": False
        },
        {
            "name": "STARTTLS_ALT_PORT",
            "port": 587,  # Alternative STARTTLS port
            "use_ssl": False,
            "use_starttls": True
        },
        {
            "name": "NO_ENCRYPTION",
            "port": 25,  # Plain text port (last resort)
            "use_ssl": False,
            "use_starttls": False
        }
    ]

    for method in connection_methods:
        for attempt in range(max_retries):
            try:
                print(f"[send_email] Attempting {method['name']} connection (attempt {attempt + 1}/{max_retries})")
                
                if method["use_ssl"]:
                    # SSL connection
                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL(SMTP_HOST, method["port"], context=context) as server:
                        server.login(IMAP_USER, IMAP_PASSWORD)
                        server.send_message(msg)
                        
                        success_info = {
                            "method": method["name"],
                            "port": method["port"],
                            "ssl": True,
                            "starttls": False,
                            "attempt": attempt + 1,
                            "recipient": to
                        }
                        print(f"[send_email] SUCCESS: Email sent to {to} using {method['name']} (SSL)")
                        return success_info
                
                else:
                    # SMTP connection with optional STARTTLS
                    with smtplib.SMTP(SMTP_HOST, method["port"]) as server:
                        server.ehlo()  # Identify ourselves
                        
                        if method["use_starttls"]:
                            # Check if STARTTLS is available
                            if server.has_extn('STARTTLS'):
                                context = ssl.create_default_context()
                                server.starttls(context=context)
                                server.ehlo()  # Re-identify after STARTTLS
                            else:
                                print(f"[send_email] STARTTLS not supported on port {method['port']}")
                                continue
                        
                        server.login(IMAP_USER, IMAP_PASSWORD)
                        server.send_message(msg)
                        
                        success_info = {
                            "method": method["name"],
                            "port": method["port"],
                            "ssl": False,
                            "starttls": method["use_starttls"],
                            "attempt": attempt + 1,
                            "recipient": to
                        }
                        print(f"[send_email] SUCCESS: Email sent to {to} using {method['name']}")
                        return success_info

            except smtplib.SMTPAuthenticationError as e:
                print(f"[send_email] Authentication failed for {method['name']}: {e}")
                # Don't retry authentication errors
                break
                
            except smtplib.SMTPRecipientsRefused as e:
                print(f"[send_email] Recipient refused for {method['name']}: {e}")
                # Don't retry recipient errors
                break
                
            except smtplib.SMTPServerDisconnected as e:
                print(f"[send_email] Server disconnected for {method['name']} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except smtplib.SMTPConnectError as e:
                print(f"[send_email] Connection failed for {method['name']} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except smtplib.SMTPException as e:
                print(f"[send_email] SMTP error for {method['name']} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except ssl.SSLError as e:
                print(f"[send_email] SSL error for {method['name']} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                print(f"[send_email] Unexpected error for {method['name']} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        print(f"[send_email] All attempts failed for {method['name']}, trying next method...")

    # If all methods failed
    print(f"[send_email] FAILED: Could not send email to {to} after trying all methods")
    return None


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