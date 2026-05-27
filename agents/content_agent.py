import os
import sys
from google import genai
from dotenv import load_dotenv

# Ensure the project root is in the python path for absolute imports when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Initialize the Gemini API client
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else genai.Client()

# Default to gemini-2.5-flash-lite to avoid rate limit / 503 issues on the free tier
DEFAULT_MODEL = "gemini-3.5-flash"

def run_content_agent(
    factual_summary: str,
    persona: str = "Technical Writer",
    format_type: str = "Executive Summary",
    model_name: str = DEFAULT_MODEL
) -> str:
    """
    Independent Content Agent that:
    1. Accepts raw factual summaries/data.
    2. Employs prompt engineering to adopt a specific persona and format type.
    3. Calls the Gemini API using google-genai SDK to polish the data without inventing facts.
    """
    if not factual_summary.strip():
        return "Error: Factual summary cannot be empty."

    # Construct the instruction and grounding constraints
    prompt = f"""You are a professional Content Specialist. Your task is to rewrite and format the raw factual data provided below.

Persona to adopt: {persona}
Target format: {format_type}

Rules:
1. Format and polish the raw facts according to the requested persona and target format.
2. Structure your output professionally using Markdown (e.g., proper headings, tables, or clean bullet points).
3. Do NOT add, extrapolate, or fabricate any new statistics, numbers, or facts. Any facts in your output must be directly supported by the input text.
4. If the input contains insufficient data to construct a full report of the requested format, organize what is there cleanly without making up filler data.

Raw Factual Input:
{factual_summary}

Formatted Output:"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error executing Content Agent call: {e}"

if __name__ == "__main__":
    print("====================================================")
    print("Running Standalone Content Agent Test (The Specialist)...")
    print("====================================================")

    # Check if API Key is set
    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY environment variable is not set!")
        print("Please check your .env file and ensure GEMINI_API_KEY is configured.")
        print("Exiting test.")
        sys.exit(1)

    # Mock RAG response data
    mock_rag_response = (
        "Q3 revenue grew by 15% to $4.2M. Gross margin is stable at 78%. "
        "Operational costs increased by $200k. Cash runway is stable at 24 months. "
        "Uptime target met at 99.98%."
    )

    print(f"Mock RAG Data Input:\n{mock_rag_response}\n")

    # Run the content agent with Technical Writer / Executive Summary settings
    print("Transforming data as a Technical Writer into an Executive Summary...")
    executive_summary = run_content_agent(
        factual_summary=mock_rag_response,
        persona="Technical Writer",
        format_type="Executive Summary"
    )

    print("\n--- Agent Output (Executive Summary) ---")
    print(executive_summary)
    print("----------------------------------------\n")

    # Run the content agent with Financial Analyst / Bulleted Brief settings
    print("Transforming data as a Financial Analyst into a Bulleted Brief...")
    bulleted_brief = run_content_agent(
        factual_summary=mock_rag_response,
        persona="Financial Analyst",
        format_type="Bulleted Brief"
    )

    print("\n--- Agent Output (Bulleted Brief) ---")
    print(bulleted_brief)
    print("-------------------------------------")
