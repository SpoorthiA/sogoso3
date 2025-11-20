"""Configuration management for SOGOSO."""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Performance Configuration
MAX_TOKENS = 300  # Optimized for speed - aggressive reduction for sub-10s latency
REQUEST_TIMEOUT = 8  # Faster timeout for quick failures

# Feature Flags
ENABLE_PROMOTIONS_AGENT = False  # Set to True when promotions.json is available

# Langfuse Configuration
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# PostgreSQL Configuration
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "sogoso_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")

# PostgreSQL Connection String
POSTGRES_CONNECTION_STRING = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"

# Data Paths
DATA_DIR = "./Data"
KNOWLEDGE_FILE = os.path.join(DATA_DIR, "knowledge.json")
PRODUCTS_FILE = os.path.join(DATA_DIR, "products.json")
PROMOTIONS_FILE = os.path.join(DATA_DIR, "promotions.json")

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
