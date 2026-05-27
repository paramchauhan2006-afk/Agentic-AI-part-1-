import os
import sys
import re
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Ensure the project root is in the python path for absolute imports when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.rag_agent import run_rag_agent
from agents.content_agent import run_content_agent
from agents.email_agent import run_email_agent

# Load environment variables
load_dotenv()

# Define the State struct for LangGraph orchestration
class AgentState(TypedDict):
    user_prompt: str
    factual_summary: str
    polished_content: str
    email_status: dict

# Define Node functions
def rag_node(state: AgentState) -> dict:
    """
    RAG Node: Accepts user prompt, performs vector search against knowledge base,
    and grounds the results using Gemini to generate a factual summary.
    """
    print("\n--- [STARTING NODE: RAG NODE] ---")
    print(f"User Query: {state['user_prompt']}")
    
    # Extract only the query part before any email references for the RAG search
    clean_query = state['user_prompt']
    email_match = re.search(r'for\s+([\w\.-]+@[\w\.-]+\.\w+)', clean_query, re.IGNORECASE)
    if email_match:
        # Strip email suffix to keep query focused on search context
        clean_query = clean_query[:email_match.start()].strip()
    
    print(f"Cleaned Query for Retrieval: {clean_query}")
    summary = run_rag_agent(clean_query)
    print(f"Factual Summary Retrieved (length: {len(summary)} characters).")
    
    return {"factual_summary": summary}

def content_node(state: AgentState) -> dict:
    """
    Content Node: Takes the raw factual summary and reformats/polishes it
    using the specified persona and style configurations.
    """
    print("\n--- [STARTING NODE: CONTENT NODE] ---")
    summary = state["factual_summary"]
    
    # We default to a Technical Writer persona and Executive Summary format
    polished = run_content_agent(
        factual_summary=summary,
        persona="Technical Writer",
        format_type="Executive Summary"
    )
    print("Content successfully polished and styled.")
    return {"polished_content": polished}

def email_node(state: AgentState) -> dict:
    """
    Email Node: Takes the polished report content, drafts a subject line,
    formats it as HTML, and executes SMTP or Mock email dispatch.
    """
    print("\n--- [STARTING NODE: EMAIL NODE] ---")
    polished = state["polished_content"]
    
    # Parse email from the user prompt
    recipient = "test@example.com"
    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', state["user_prompt"])
    if email_match:
        recipient = email_match.group(0)
        print(f"Extracted recipient email from prompt: {recipient}")
    else:
        # Fallback to env or default
        env_receiver = os.environ.get("SMTP_RECEIVER")
        if env_receiver:
            recipient = env_receiver
            print(f"No email found in prompt. Using SMTP_RECEIVER from .env: {recipient}")
        else:
            print(f"No email found in prompt or environment. Using fallback: {recipient}")

    status = run_email_agent(
        polished_content=polished,
        recipient_email=recipient
    )
    print(f"Email dispatch completed with status: {status.get('status')}")
    return {"email_status": status}

# Assemble the StateGraph
workflow = StateGraph(AgentState)

# 1. Add all nodes
workflow.add_node("rag_node", rag_node)
workflow.add_node("content_node", content_node)
workflow.add_node("email_node", email_node)

# 2. Define execution flow connections (START >> rag_node >> content_node >> email_node >> END)
workflow.add_edge(START, "rag_node")
workflow.add_edge("rag_node", "content_node")
workflow.add_edge("content_node", "email_node")
workflow.add_edge("email_node", END)

# Compile the workflow graph
compiled_graph = workflow.compile()

if __name__ == "__main__":
    print("====================================================")
    print("Running End-to-End LangGraph Orchestration Pipeline...")
    print("====================================================")

    # Verify Gemini Key
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY is not configured in environment variables.")
        sys.exit(1)

    # Test Query
    input_query = "Give me an executive report on our Q3 financial progress for hello@acme.com"
    
    initial_state = {
        "user_prompt": input_query,
        "factual_summary": "",
        "polished_content": "",
        "email_status": {}
    }

    print(f"Initializing LangGraph with input state: {initial_state}\n")

    # Execute graph streaming output to watch state updates
    for output in compiled_graph.stream(initial_state):
        for node_name, state_update in output.items():
            print(f"\n---> Node '{node_name}' Finished and returned:")
            import json
            # Pretty print state modifications
            print(json.dumps(state_update, indent=2))
            print("====================================================")
            
    print("\nOrchestration Flow Finished Successfully!")
