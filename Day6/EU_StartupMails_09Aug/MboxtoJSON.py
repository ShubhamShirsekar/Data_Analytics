import mailbox
import json
import sys
import os

def get_email_body(msg):
    """Extracts the email body from plain text or HTML parts."""
    if msg.is_multipart():
        parts = []
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                try:
                    parts.append(part.get_payload(decode=True).decode(errors="ignore"))
                except:
                    pass
        return "\n".join(parts).strip()
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode(errors="ignore").strip()
    return ""  # No body found


def main():
    if len(sys.argv) < 2:
        print("❌ Usage: python MboxtoJSON.py <mbox_file>")
        sys.exit(1)

    mbox_file = sys.argv[1]

    if not os.path.exists(mbox_file):
        print(f"❌ File not found: {mbox_file}")
        sys.exit(1)

    # Load MBOX
    mbox = mailbox.mbox(mbox_file)

    emails = []
    for message in mbox:
        email_data = {
            "from": message['from'],
            "to": message['to'],
            "subject": message['subject'],
            "date": message['date'],
            "body": get_email_body(message)
        }
        emails.append(email_data)

    # Save as JSON
    output_file = os.path.splitext(mbox_file)[0] + ".json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(emails, f, ensure_ascii=False, indent=2)

    print(f"✅ Converted {len(emails)} emails to {output_file}")


if __name__ == "__main__":
    main()
