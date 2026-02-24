import os
from dotenv import load_dotenv

import oracledb
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_oracledb.vectorstores import oraclevs
from langchain_oracledb.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy

load_dotenv() # For loading environment variables from a .env file


# 1. Set environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_DSN = os.getenv('DB_DSN')
WALLET_DIR = os.getenv('WALLET_DIR')
SAMPLE_TEXT_FILE_PATH = os.getenv('SAMPLE_TEXT_FILE_PATH')


# 2. Connect to Oracle Database 23ai
db_pwd = DB_PASSWORD
dsn = DB_DSN
wallet_dir = WALLET_DIR

print("Connecting to Oracle DB...")
connection = oracledb.connect(
    user="ADMIN",
    password=db_pwd,
    dsn=dsn,
    config_dir=wallet_dir,
    wallet_location=wallet_dir,
    wallet_password=db_pwd
)
print("Connected successfully!")

# 3. Load and split the manual
print("Loading document...")
# If using PDF: loader = PyPDFLoader("cx750_manual.pdf")
loader = TextLoader(SAMPLE_TEXT_FILE_PATH)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# 4. Initialize Embeddings and Oracle Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("Vectorizing and loading into Oracle 26ai...")
vector_store = OracleVS.from_documents(
    chunks,
    embeddings,
    client=connection,
    table_name="TRACTOR_MANUALS",
    distance_strategy=DistanceStrategy.COSINE
)
print("Data successfully loaded into Oracle Vector DB!")