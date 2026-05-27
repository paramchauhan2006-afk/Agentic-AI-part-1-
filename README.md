# Multi-Pipeline Agentic AI Framework
> A professional, production-ready multi-agent framework orchestrating LangChain/LangGraph and Google Agent Development Kit (ADK) pipelines with a shared core logic layer.

---

## 🏗️ Project Architecture

This repository adopts a modular, **hybrid agentic design**. By segregating agent orchestration environments from the core logical definitions of agents, tools, and analytics, the system allows identical agent behavior to run across two distinct runtimes:
1. **LangGraph Flow**: Graph-based, state-driven, fine-grained control flow.
2. **Google ADK Flow**: Event-driven, model-driven collaboration and orchestration.

### Architectural Diagram

```
                     ┌────────────────────────┐
                     │   Execution Pipelines  │
                     └────────────┬───────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌──────────────────┐                              ┌──────────────────┐
│  LangGraph Flow  │                              │  Google ADK Flow │
│ (Stateful Graph) │                              │  (Event-Driven)  │
└────────┬─────────┘                              └────────┬─────────┘
         │                                                 │
         └────────────────────────┬────────────────────────┘
                                  ▼
                     ┌────────────────────────┐
                     │   Shared Logic Layer   │
                     └────────────┬───────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│    RAG Agent     │      │  Content Agent   │      │   Email Agent    │
│  (Vector Search) │      │ (Reports/Drafts) │      │  (SMTP Dispatch) │
└────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   ▼
                      ┌────────────────────────┐
                      │ Shared Tools & Engines │
                      └────────────┬───────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
         ┌────────────────────┐        ┌────────────────────┐
         │    Custom Tools    │        │  Analytics Engine  │
         │ (Retrieval, SMTP)  │        │ (Forecasting, etc.)│
         └────────────────────┘        └────────────────────┘
```

---

## 📂 Project Directory Structure

```text
.
├── agents/                       # Shared framework-agnostic agent logic
│   └── __init__.py
├── analytics/                    # Time-series forecasting & quantitative analysis engine
│   └── __init__.py
├── pipelines/                    # Orchestration workflows
│   ├── adk_flow/                 # Google Agent Development Kit (ADK) pipeline
│   │   └── __init__.py
│   └── langgraph_flow/           # LangChain / LangGraph stateful workflow
│       └── __init__.py
├── tools/                        # Custom agent utility tools (SMTP, Vector DB, etc.)
│   └── __init__.py
├── .env.example                  # Environment configuration template
├── README.md                     # Architecture blueprint and roadmap
└── requirements.txt              # Unified package requirements
```

---

## 🛠️ Installation & Setup

1. **Prerequisites**: Python 3.9 or higher.
2. **Setup Virtual Environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and populate details:
   ```bash
   copy .env.example .env
   ```

---

## 🤖 Core Agents Blueprint

We define 3 core agents designed to execute independently prior to sequence orchestration:

| Agent Name | Core Responsibility | Key Components |
| :--- | :--- | :--- |
| **RAG Agent** | Semantic lookup, parsing, and context retrieval. | Chroma/FAISS (Vector Store), Gemini Embeddings (`google-genai`), document loader. |
| **Content Agent** | Information synthesis, summarization, and report generation. | Gemini 2.x/3.x models, Pydantic structured output, mathematical/forecasting interpretation. |
| **Email Agent** | Outbound distribution and notifications. | SMTP connection pooling, template renderer, status logger. |

---

## 🔄 Step-by-Step Execution Sequence

To ensure robustness, the implementation lifecycle is structured in three consecutive phases:

### Phase 1: Independent Agent Unit Implementation
Each agent must be implemented as a framework-agnostic component under `/agents/` that exposes a clean, input-output functional interface.
*   **RAG Agent**: Accepts a textual query, performs vector search, and returns matched document snippets.
*   **Content Agent**: Accepts raw data/analytics and retrieved context, processes them via LLM, and returns structured text.
*   **Email Agent**: Accepts recipient address, subject, and body, validates via Pydantic, and executes SMTP delivery.

### Phase 2: Shared Tool Integration
Utility tools under `/tools/` are mapped to Pydantic classes or standard Python functions with robust typing. These tools are passed directly to the agents as function schemas, allowing LLMs to execute tool calls dynamically.

### Phase 3: Pipeline Sequence Orchestration
Once the individual agents and tools are fully tested, we sequence them to form a unified pipeline where:
`Data Input ➔ RAG Agent Lookup ➔ Analytics Synthesis ➔ Content Agent Generation ➔ Email Agent Dispatch`.

#### LangGraph Orchestration (`pipelines/langgraph_flow/`)
*   Define a graph state object representing the workflow context.
*   Model each agent/tool step as a Node.
*   Establish state transitions and control flow routes as Edges.

#### Google ADK Orchestration (`pipelines/adk_flow/`)
*   Initialize agents using ADK's `Agent` primitives.
*   Set up an event listener/publisher sequence.
*   Leverage ADK runtime for agent collaboration and delegation.
