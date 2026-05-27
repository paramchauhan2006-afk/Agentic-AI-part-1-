import os
import sys
import json
import warnings
from dotenv import load_dotenv

# Suppress deprecation warnings for cleaner verification log
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ensure the project root is in the python path for absolute imports when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agents.rag_agent import run_rag_agent
from agents.content_agent import run_content_agent
from agents.email_agent import run_email_agent

# Load environment variables
load_dotenv()

MODEL_NAME = "gemini-3.5-flash"

# ==============================================================================
# Define the Custom Python Tools (Skills) for the ADK Agents
# ==============================================================================

def run_rag_tool(query: str) -> str:
    """
    Retrieves corporate records and returns a factual summary matching the query.
    """
    print(f"\n[Tool Execution] RAG Tool invoked with query: '{query}'")
    result = run_rag_agent(query)
    print("[Tool Execution] RAG Tool retrieval complete.")
    return result

def run_content_tool(factual_summary: str) -> str:
    """
    Transforms raw facts into a polished, structured markdown Executive Summary report.
    """
    print("\n[Tool Execution] Content Tool invoked to polish summary data.")
    result = run_content_agent(
        factual_summary=factual_summary,
        persona="Technical Writer",
        format_type="Executive Summary"
    )
    print("[Tool Execution] Content formatting complete.")
    return result

def run_email_tool(polished_content: str, recipient_email: str) -> str:
    """
    Formulates a subject line, converts markdown to HTML, and dispatches the email.
    """
    print(f"\n[Tool Execution] Email Tool invoked for dispatch to: {recipient_email}")
    result = run_email_agent(
        polished_content=polished_content,
        recipient_email=recipient_email
    )
    print(f"[Tool Execution] Email dispatch result: {result.get('status')}")
    return json.dumps(result)

# ==============================================================================
# Define LlmAgents utilizing the custom tools
# ==============================================================================

research_agent = LlmAgent(
    name="ResearchAgent",
    model=MODEL_NAME,
    instruction=(
        "You are the Research Agent. Your sole responsibility is to extract relevant facts "
        "by executing the `run_rag_tool` using the user's initial search query. "
        "Ensure you pass the full query context to the tool, and output its exact response."
    ),
    tools=[run_rag_tool]
)

specialist_agent = LlmAgent(
    name="SpecialistAgent",
    model=MODEL_NAME,
    instruction=(
        "You are the Specialist Agent (Technical Writer). Your job is to take the factual summary "
        "output from the Research Agent and format it using the `run_content_tool`. "
        "Do NOT invent details. Pass the factual content directly to the tool and return the output."
    ),
    tools=[run_content_tool]
)

communicator_agent = LlmAgent(
    name="CommunicatorAgent",
    model=MODEL_NAME,
    instruction=(
        "You are the Communicator Agent. Your task is to extract the recipient's email address "
        "from the user prompt, read the formatted report, and dispatch it by executing the `run_email_tool`. "
        "Pass both the polished report and the recipient email to the tool, and output the response."
    ),
    tools=[run_email_tool]
)

# ==============================================================================
# Assemble the Sequential Workflow
# ==============================================================================
adk_workflow = SequentialAgent(
    name="CorporateReportPipeline",
    description="Sequential pipeline coordinating research, polishing, and email delivery.",
    sub_agents=[research_agent, specialist_agent, communicator_agent]
)

def run_adk_pipeline(query: str) -> list:
    """
    Initializes the Google ADK runner environment, runs the sequential workflow
    with the query, and prints the transition events.
    """
    # Create session and runner
    session_service = InMemorySessionService()
    runner = Runner(
        agent=adk_workflow,
        app_name="adk_flow_app",
        session_service=session_service,
        auto_create_session=True
    )
    
    # Run user prompt
    new_message = types.Content(role='user', parts=[types.Part(text=query)])
    
    print(f"Triggering ADK workflow runner with message: '{query}'")
    events = runner.run(
        user_id="adk_tester",
        session_id="session_adk_001",
        new_message=new_message
    )
    
    responses = []
    for event in events:
        # Log final model transitions / responses
        if event.is_final_response() and event.content:
            text = event.content.parts[0].text
            agent_name = event.author if event.author else "UnknownAgent"
            print(f"\n---> [ADK Transition] {agent_name} output completed:")
            print(text)
            print("====================================================")
            responses.append({"agent": agent_name, "output": text})
            
    return responses

if __name__ == "__main__":
    print("====================================================")
    print("Running End-to-End Google ADK Orchestration Pipeline...")
    print("====================================================")

    # Verify Gemini Key
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY is not configured in environment variables.")
        sys.exit(1)

    # Test Query
    test_query = "Give me an executive report on our Q3 financial progress for hello@acme.com"
    
    run_adk_pipeline(test_query)
    print("\nGoogle ADK Orchestration Workflow Finished Successfully!")
