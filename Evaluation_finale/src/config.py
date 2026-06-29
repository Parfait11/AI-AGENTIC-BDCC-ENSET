from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma_db"

LLM_MODEL_NAME = "llama3.2:3b"
#EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
COLLECTION_NAME = "cholera_guidelines"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 30
RETRIEVAL_K = 4
MAX_REWRITES = 2