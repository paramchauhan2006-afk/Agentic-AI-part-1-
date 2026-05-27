import os
import sys
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Ensure the project root is in the python path for absolute imports when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.email_tool import send_email_via_smtp

# Load environment variables
load_dotenv()

# Initialize the Gemini API client
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else genai.Client()

MODEL_NAME = "gemini-3.5-flash"

# Define the Pydantic schema for structured output from the LLM
class EmailDraft(BaseModel):
    subject: str = Field(
        description="A captivating, highly professional, and brief subject line based on the email content."
    )
    html_body: str = Field(
        description=(
            "Clean, professional email body in basic HTML format. "
            "Use inline CSS styles for a premium look (e.g. font-family: sans-serif, neutral colors). "
            "Convert all markdown headers to <h2>/<h3>, lists to <ul>/<li>, and paragraphs to <p>. "
            "Do NOT include markdown markers or wrapper code blocks."
        )
    )

def run_email_agent(polished_content: str, recipient_email: str) -> dict:
    """
    Independent Email Agent that:
    1. Directs Gemini 2.5 Flash Lite to extract a subject and format markdown content into clean inline HTML.
    2. Uses Pydantic structured output to avoid parsing errors.
    3. Invokes the SMTP tool to deliver the email (or falls back to mock delivery).
    4. Returns a dispatch status report.
    """
    if not polished_content.strip():
        return {"status": "failed", "error": "Content cannot be empty"}

    prompt = f"""You are an Email Communications Specialist. Review the polished report provided below and perform two tasks:
1. Craft a captivating, professional, and context-relevant email subject line.
2. Transform the markdown report into a premium, clean HTML layout (using basic tags and inline CSS for clean spacing, modern fonts like Arial/Helvetica, and readable line heights).

Raw Report Markdown:
{polished_content}
"""

    try:
        # Request structured JSON matching our Pydantic schema
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EmailDraft,
                temperature=0.2
            )
        )
        
        # Parse the structured response
        draft = EmailDraft.model_validate_json(response.text)
        
        # Dispatch the email (SMTP or Mock)
        success = send_email_via_smtp(
            recipient=recipient_email,
            subject=draft.subject,
            body_html=draft.html_body
        )
        
        if success:
            return {
                "status": "success",
                "subject": draft.subject,
                "recipient": recipient_email,
                "html_body_preview": draft.html_body[:200] + "..."
            }
        else:
            return {
                "status": "failed",
                "error": "SMTP dispatch failed"
            }
            
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Email Agent execution error: {e}"
        }

if __name__ == "__main__":
    print("====================================================")
    print("Running Standalone Email Agent Test (The Communicator)...")
    print("====================================================")

    # Check if API Key is set
    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY environment variable is not set!")
        print("Please check your .env file and ensure GEMINI_API_KEY is configured.")
        print("Exiting test.")
        sys.exit(1)

    # Mock markdown report (usually from Content Agent)
    mock_markdown_report = (
        "## Executive Summary - Q3 Performance Overview\n\n"
        "This executive summary provides a concise overview of key performance indicators for the third quarter.\n\n"
        "### Financial Performance\n"
        "* **Revenue:** Q3 revenue grew by 15% to reach a total of $4.2 million.\n"
        "* **Gross Margin:** Stood stable at 78%.\n"
        "* **Operational Costs:** Saw an increase of $200,000.\n\n"
        "### Financial Stability\n"
        "* **Cash Runway:** Remains stable at 24 months."
    )

    test_recipient = "test@example.com"
    print(f"Mock Content Report:\n{mock_markdown_report}\n")
    print(f"Target Recipient: {test_recipient}\n")

    print("Drafting and sending email...")
    report = run_email_agent(mock_markdown_report, test_recipient)
    
    print("\n--- Agent Execution Report ---")
    import json
    print(json.dumps(report, indent=2))
    print("------------------------------")
