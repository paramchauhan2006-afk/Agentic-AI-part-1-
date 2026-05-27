import os
import sys
from google import genai
from dotenv import load_dotenv

# Ensure the project root is in the python path for absolute imports when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.rag_tool import retrieve_top_k

# Load environment variables
load_dotenv()

# Initialize the Gemini API client
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else genai.Client()

MODEL_NAME = "gemini-3.5-flash"

def run_rag_agent(query: str) -> str:
    """
    Independent RAG Agent that:
    1. Retrieves relevant contexts using the RAG tool.
    2. Constructs a strict prompt incorporating the contexts.
    3. Calls Gemini 2.5 Flash to generate a factual, context-grounded response.
    """
    # 1. Retrieve the top K matching contexts (K=3)
    contexts = retrieve_top_k(query, k=3)
    
    if not contexts:
        return "No relevant context files found in the knowledge base. Please verify the knowledge_base folder contains data."

    # 2. Format the retrieved contexts into a string
    context_str = ""
    for idx, ctx in enumerate(contexts):
        context_str += f"\n--- Context Source: {ctx['source']} (Similarity: {ctx['similarity']:.4f}) ---\n"
        context_str += f"{ctx['text']}\n"

    # 3. Formulate the strict grounding prompt
    prompt = f"""You are a precise, factual Research Agent. Your task is to answer the user's query strictly using only the retrieved contexts provided below.

Rules:
- Ground your answer completely and exclusively in the provided context.
- If the answer to the query cannot be found or inferred directly from the context, state: "Based on the provided context, I cannot find the answer to this question."
- Do not make up facts, numbers, or assume details not present in the text.
- Cite the source files (e.g. sample_data.txt) when presenting facts.

Retrieved Contexts:
{context_str}

Query: {query}

Factual Summary:"""

    # 4. Invoke the Gemini Model using google-genai SDK
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error executing RAG Agent call: {e}"

if __name__ == "__main__":
    print("====================================================")
    print("Running Standalone RAG Agent Test (The Researcher)...")
    print("====================================================")

    # Check if API Key is set
    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY environment variable is not set!")
        print("Please check your .env file and ensure GEMINI_API_KEY is configured.")
        print("Exiting test.")
        sys.exit(1)

    # Test Query
    test_query = "What was Acme Corp's revenue growth in Q3, and what caused operational costs to rise?"
    print(f"Query: {test_query}\n")
    
    print("Retrieving context and generating response...")
    result = run_rag_agent(test_query)
    
    print("\n--- Agent Response ---")
    print(result)
    print("----------------------")
