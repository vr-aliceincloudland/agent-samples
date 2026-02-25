# Tractor Repair Assistant Agent

This project implements an AI-powered agent designed to assist technicians with tractor repair and maintenance tasks. The agent uses a Retrieval Augmented Generation (RAG) approach, first consulting internal repair manuals stored in an Oracle Vector Database and then falling back to a web search if necessary.

The agent is built using LangChain, LangGraph, OpenAI, and Oracle 26ai Database.

## How It Works

The agent's logic is orchestrated by a graph built with LangGraph. The flow is as follows:

1.  **Retrieve**: When a user asks a question, the agent first retrieves relevant documents from a pre-loaded collection of tractor repair manuals stored in an Oracle Vector Database.
2.  **Grade Documents**: The retrieved documents are then passed to an LLM to check for relevance to the user's question. This step filters out erroneous or unhelpful retrievals.
3.  **Conditional Routing**: Based on the relevance grade:
    *   If the documents are **relevant**, the agent proceeds to generate an answer based on them.
    *   If the documents are **not relevant**, the agent performs a web search using Tavily to find up-to-date information.
4.  **Generate**: The agent uses the context (either from the manuals or the web search) to generate a comprehensive and helpful response for the technician, emphasizing safety protocols.
5.  **Observability**: All steps, LLM calls, and agent decisions are traced using Langfuse for debugging and monitoring.

## Prerequisites

Before you begin, ensure you have the following:
*   Python 3.9+
*   An Oracle Database (e.g., Oracle Database 23ai Free) with network access.
*   An Oracle Wallet for connecting to your database.
*   API keys for:
    *   OpenAI
    *   Tavily
*   Credentials for a Langfuse project (`LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST`).

## Setup

1.  **Clone the Repository**
    ```bash
    # Or simply use the existing project files
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    A `requirements.txt` is not provided, but you can install the necessary packages with pip:
    ```bash
    pip install langchain langchain-openai langchain-oracledb langchain-community langgraph oracledb python-dotenv tavily-python langfuse
    ```

4.  **Set up Environment Variables**
    Create a `.env` file in the root of the project directory and add your credentials. The Oracle Wallet files should be placed in a directory (e.g., `wallet/`) and the path configured in the `.env` file.

    ```env
    # .env
    OPENAI_API_KEY="sk-..."
    TAVILY_API_KEY="tvly-..."

    # Langfuse
    LANGFUSE_SECRET_KEY="sk-lf-..."
    LANGFUSE_PUBLIC_KEY="pk-lf-..."
    LANGFUSE_HOST="https://cloud.langfuse.com" # Or your self-hosted instance

    # Oracle Database
    DB_USER="your_db_user"
    DB_PASSWORD="your_db_password"
    DB_DSN="your_db_dsn"
    WALLET_DIR="./wallet" # Path to your Oracle Wallet directory
    ```
    *Note: The `agent.py` script expects the wallet password to be the same as the database user password.*

5.  **Load Data into Oracle Vector DB**
    This agent assumes that the tractor manuals have already been processed and loaded into an Oracle Vector Store table named `TRACTOR_MANUALS`. You will need to run a separate script to ingest your documents.

## Running the Agent

Once the setup is complete, you can run the agent from your terminal:

```bash
python agent.py
```

The application will start, and you can interact with it directly in the command line.

```
Welcome to the Tractor Repair and Maintenance AI Assistant.
Type 'exit' to quit.

Technician: How do I check the hydraulic fluid level?

AI Agent: To check the hydraulic fluid level, first ensure the tractor is on level ground and the engine is off. Locate the hydraulic fluid reservoir, which is typically near the rear of thetractor. Clean the area around the dipstick or sight glass to prevent contamination. If you have a dipstick, remove it, wipe it clean, re-insert it fully, and then remove it again to check the level. The fluid should be between the 'MIN' and 'MAX' marks. If you have a sight glass, the level should be visible within the marked range. Always use the recommended hydraulic fluid type specified in your manual.
```

To stop the agent, type `exit` or `quit`.

## Project Files

*   `agent.py`: The main entry point of the application. It initializes the LLM, tools, vector store, and defines the LangGraph workflow. It also contains the command-line interface loop.
*   `tools.py`: Defines the `AgentState` and the `AgentTools` class, which contains the implementation for each node in the graph (e.g., `retrieve_node`, `grade_documents_node`, `generate_node`).
*   `.env`: (To be created by you) Stores all the necessary credentials and configuration variables.