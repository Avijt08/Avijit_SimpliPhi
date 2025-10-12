# Enhanced API configuration for Property RAG System
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Get the project root directory (go up three levels from src/config.py to reach the main project root)
project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

print(f"Loading .env from: {env_path}")
print(f".env file exists: {env_path.exists()}")

class Config:
    """Configuration class for Property RAG system"""
    
    # API configuration
    API_HOST = "127.0.0.1"
    API_PORT = 8000
    
    # Backend URL
    BACKEND_URL = f"http://{API_HOST}:{API_PORT}"
    
    # Google Gemini API configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    
    # ChromaDB configuration
    CHROMADB_PERSIST_DIR = os.getenv("CHROMADB_PERSIST_DIR", "./chromadb")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Vector embedding configuration
    EMBEDDING_DIMENSION = 768
    USE_EMBEDDINGS = True
    
    # RAG system configuration
    MAX_SEARCH_RESULTS = 20
    DEFAULT_SEARCH_RESULTS = 10
    
    # LLM Response configuration
    MAX_CONTEXT_PROPERTIES = 5
    ENABLE_QUERY_ANALYSIS = True

# Legacy variables for backward compatibility
API_HOST = Config.API_HOST
API_PORT = Config.API_PORT
BACKEND_URL = Config.BACKEND_URL
GOOGLE_API_KEY = Config.GOOGLE_API_KEY
GEMINI_MODEL = Config.GEMINI_MODEL
CHROMADB_PERSIST_DIR = Config.CHROMADB_PERSIST_DIR
EMBEDDING_MODEL = Config.EMBEDDING_MODEL
EMBEDDING_DIMENSION = Config.EMBEDDING_DIMENSION
USE_EMBEDDINGS = Config.USE_EMBEDDINGS
MAX_SEARCH_RESULTS = Config.MAX_SEARCH_RESULTS
DEFAULT_SEARCH_RESULTS = Config.DEFAULT_SEARCH_RESULTS
MAX_CONTEXT_PROPERTIES = Config.MAX_CONTEXT_PROPERTIES
ENABLE_QUERY_ANALYSIS = Config.ENABLE_QUERY_ANALYSIS