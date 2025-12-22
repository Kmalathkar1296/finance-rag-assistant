import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class Config:
    # API Keys
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
    
    # Database
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    
    # Embedding Model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Data Generation Settings
    DEFAULT_AR_RECORDS = 100
    DEFAULT_CLAIMS_RECORDS = 200
    DEFAULT_BUDGET_YEARS = 2
    
    # Paths
    DATA_RAW_DIR = "data/raw"
    DATA_PROCESSED_DIR = "data/processed"
    OUTPUTS_REPORTS_DIR = "outputs/reports"
    OUTPUTS_EXPORTS_DIR = "outputs/exports"
    
    # RAG Settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 100
    RETRIEVAL_K = 5

    @classmethod
    def ensure_directories(cls):
        directories = [
            cls.DATA_RAW_DIR,
            cls.DATA_PROCESSED_DIR,
            cls.OUTPUTS_REPORTS_DIR,
            cls.OUTPUTS_EXPORTS_DIR,
            cls.CHROMA_PERSIST_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    @staticmethod
    def get_timestamp():
        return datetime.now().strftime("%Y%m%d_%H%M%S")
