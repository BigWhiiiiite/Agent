import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DOCS_DIR = BASE_DIR / "docs"
EMBEDDING_CACHE_PATH = BASE_DIR / "embedding_cache.json"
VECTOR_INDEX_PATH = BASE_DIR / "vector_index.json"
TOP_K = 3
MAX_AGENT_STEPS = 6
MIN_SEMANTIC_SIMILARITY = float(os.getenv("MIN_SEMANTIC_SIMILARITY", "0.2"))
