import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def is_placeholder(value: str) -> bool:
    """
    Checks if a configuration value is empty, None, or a generic placeholder.
    """
    if not value:
        return True
    placeholders = [
        "your_email@gmail.com",
        "your_app_specific_password_here",
        "recipient_email@example.com"
    ]
    return value.strip().lower() in placeholders or "your_" in value.lower()

def send_email_via_smtp(recipient: str, subject: str, body_html: str) -> bool:
    """
    Sends an HTML email via SMTP, or logs a mock email payload if configuration
    is missing or set to placeholder values.
    """
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_port = os.environ.get("SMTP_PORT")
    smtp_username = os.environ.get("SMTP_USERNAME") or os.environ.get("SMTP_USER")
    smtp_password = os.environ.get("SMTP_PASSWORD")
    smtp_sender = os.environ.get("SMTP_SENDER") or smtp_username

    # Determine if we should run in Mock mode
    use_mock = (
        is_placeholder(smtp_server) or
        is_placeholder(smtp_port) or
        is_placeholder(smtp_username) or
        is_placeholder(smtp_password) or
        is_placeholder(recipient)
    )

    if use_mock:
        print("\n+------------------------------------------------------------------------+")
        print("|                       [MOCK EMAIL DISPATCH]                            |")
        print("+------------------------------------------------------------------------+")
        print(f"| To:      {recipient}")
        print(f"| From:    {smtp_sender or 'mock-sender@example.com'}")
        print(f"| Subject: {subject}")
        print("| Body (HTML format):")
        # Print body lines with border padding
        for line in body_html.split("\n"):
            print(f"| {line}")
        print("+------------------------------------------------------------------------+\n")
        return True

    # Real SMTP execution
    try:
        port = int(smtp_port) if smtp_port else 587
        
        # Create message container
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = smtp_sender
        msg['To'] = recipient

        # Attach HTML content
        part = MIMEText(body_html, 'html')
        msg.attach(part)

        # Establish connection
        if port == 465:
            server = smtplib.SMTP_SSL(smtp_server, port)
        else:
            server = smtplib.SMTP(smtp_server, port)
            server.ehlo()
            if port == 587:
                server.starttls()
                server.ehlo()

        # Login and send
        server.login(smtp_username, smtp_password)
        server.sendmail(smtp_sender, recipient, msg.as_string())
        server.quit()
        
        print(f"Success: Real email sent successfully to {recipient}.")
        return True
    except Exception as e:
        print(f"Error: Real SMTP dispatch failed: {e}")
        print("Falling back to printing mock details...")
        # Fallback dump
        print(f"\n--- [FALLBACK MOCK] To: {recipient} | Subject: {subject} ---\n{body_html}\n")
        return False
