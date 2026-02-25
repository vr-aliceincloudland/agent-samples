import os
from dotenv import load_dotenv
import oracledb

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_oracledb.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from tools import AgentState, AgentTools
from langfuse import get_client
from langfuse.langchain import CallbackHandler


load_dotenv() # For loading environment variables from a .env file

# --- Setup Credentials & DB Connection ---
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_DSN = os.getenv('DB_DSN')
WALLET_DIR = os.getenv('WALLET_DIR')

print(f"OpenAI Key: {OPENAI_API_KEY}")

# Initialize Langfuse client
langfuse = get_client()

# Verify LangFuse connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
 
# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

connection = oracledb.connect(
    user=DB_USER, password=DB_PASSWORD,
    dsn=DB_DSN, config_dir=WALLET_DIR,
    wallet_location=WALLET_DIR, wallet_password=DB_PASSWORD
)

# --- Initialize Vector Store, LLM & Tools ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = OracleVS(
    client=connection,
    embedding_function=embeddings,
    table_name="TRACTOR_MANUALS",
    distance_strategy=DistanceStrategy.COSINE
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
web_search_tool = TavilySearchResults(k=3)

# --- Initialize Tools ---
tools = AgentTools(retriever=retriever, llm=llm, web_search_tool=web_search_tool)

# --- Build the Graph ---
workflow = StateGraph(AgentState)

# Define the nodes
workflow.add_node("retrieve", tools.retrieve_node)
workflow.add_node("grade_documents", tools.grade_documents_node)
workflow.add_node("generate", tools.generate_node)
workflow.add_node("web_search", tools.web_search_node)

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    tools.decide_to_generate,
    {"web_search": "web_search", "generate": "generate"},
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# --- Run the Agent ---
if __name__ == "__main__":
    print("Welcome to the Tractor Repair and Maintenance AI Assistant.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("Technician: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        inputs = {"messages": [HumanMessage(content=user_input)]}
        # Run the graph
        for output in app.stream(inputs, config={"callbacks": [langfuse_handler]}):
            for key, value in output.items():
                if key == "generate":
                    print(f"\nAI Agent: {value['messages'][-1].content}\n")
    
    langfuse.shutdown()