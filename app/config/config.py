import os
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

_DEFAULT_UPLOAD_DIR_RELATIVE_TO_CONFIG_FILE = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
)
 
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
 
# Add other config variables as needed
API_VERSION = "v1"
 
@dataclass
class Config:
    """Consolidated application configuration settings"""

    # === Milvus Configuration ===
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION_NAME: str = os.getenv("MILVUS_COLLECTION_NAME", "collection")
    MILVUS_DB_NAME: str = os.getenv("MILVUS_DB_NAME", "hv_doc")
    MILVUS_USER: str = os.getenv("MILVUS_USER", "root")
    MILVUS_PASSWORD: str = os.getenv("MILVUS_PASSWORD", "Milvus")
    MILVUS_TIMEOUT: float = 30.0  # seconds
    MILVUS_K: int = int(os.getenv("MILVUS_K", "10"))
    LIST_OF_SPARSE_DOCUMENTS: List[str] = field(default_factory=lambda: json.loads(os.getenv("LIST_OF_SPARSE_DOCUMENTS", '["List of Franchisees", "List of Franchisees Who Have Left the System"]')))
    K_FOR_SPARSE_SEARCH: int = int(os.getenv("K_FOR_SPARSE_SEARCH", "1"))  # Limit to 1 result for large documents

    # Milvus Search Parameters
    MILVUS_SEARCH_EF: int = int(os.getenv("MILVUS_SEARCH_EF", "250"))
    MILVUS_SEARCH_DROP_RATIO: float = float(os.getenv("MILVUS_SEARCH_DROP_RATIO", "0.2"))
    MILVUS_INDEX_M: int = int(os.getenv("MILVUS_INDEX_M", "32"))
    MILVUS_INDEX_EF_CONSTRUCTION: int = int(os.getenv("MILVUS_INDEX_EF_CONSTRUCTION", "250"))
    MILVUS_INDEX_DROP_RATIO: float = float(os.getenv("MILVUS_INDEX_DROP_RATIO", "0.2"))
    fetch_k: Optional[int] = int(os.getenv("MILVUS_FETCH_K", "50"))  
    MILVUS_RANKER_TYPE: str = os.getenv("MILVUS_RANKER_TYPE", "rrf")
    MILVUS_RANKER_PARAMS: Dict[str, Any] = field(default_factory=lambda: json.loads(os.getenv("MILVUS_RANKER_PARAMS", "{}")))
    MILVUS_SPARSE_RANKER_PARAMS: Dict[str, Any] = field(default_factory=lambda: json.loads(os.getenv("MILVUS_SPARSE_RANKER_PARAMS", "{}")))

    # === Model Configuration ===
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "bge-m3")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4000"))  # Reduced from 7000 to leave room for context
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "http://localhost:8000/v1")
    VLLM_EMBEDDING_URL: str = os.getenv("VLLM_EMBEDDING_URL", "http://localhost:8020/v1")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY","EMPTY")  # Default for vLLM compatibility
    MODEL_CONTEXT_LENGTH: int = 7000  # Maximum context length for the model
    LLM_REPHRASER_MAX_TOKENS: int = int(os.getenv("LLM_REPHRASER_MAX_TOKENS", "100"))
    LLM_GUIDED_MESSAGE_MAX_TOKENS: int = int(os.getenv("LLM_GUIDED_MESSAGE_MAX_TOKENS", "500"))
    LLM_MESSAGE_MAX_TOKENS: int = int(os.getenv("LLM_MESSAGE_MAX_TOKENS", "300"))   
    LLM_GENERAL_CLS_MAX_TOKENS: int = int(os.getenv("LLM_GENERAL_CLS_MAX_TOKENS", "100"))
    LLM_NAVIGATION_MAX_TOKENS: int = int(os.getenv("LLM_NAVIGATION_MAX_TOKENS", "100"))
    LLM_MESSAGE_TEMPERATURE: float = float(os.getenv("LLM_MESSAGE_TEMPERATURE", "0.8"))

    # === Redis Configuration ===
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_MESSAGE_TTL: int = int(os.getenv("REDIS_MESSAGE_TTL", "3600"))
    REDIS_MAX_CONNECTIONS: int = int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
    REDIS_RETRY_ATTEMPTS: int = int(os.getenv("REDIS_RETRY_ATTEMPTS", "3"))
    REDIS_RETRY_DELAY: float = float(os.getenv("REDIS_RETRY_DELAY", "1.0"))
    CONVERSATION_HISTORY_LIMIT: int = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "5"))
    # === Server Configuration ===
    PORT: int = int(os.getenv("PORT", "8098"))
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    @property
    def MILVUS_URI(self) -> str:
        """Construct Milvus URI from host and port."""
        return f"grpc://{self.MILVUS_HOST}:{self.MILVUS_PORT}"

    @property
    def MILVUS_INDEX_PARAMS(self) -> List[Dict[str, Any]]:
        """Get Milvus index parameters."""
        return [
            {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {
                    "M": self.MILVUS_INDEX_M,
                    "efConstruction": self.MILVUS_INDEX_EF_CONSTRUCTION
                }
            },
            {
                "metric_type": "BM25",
                "index_type": "SPARSE_INVERTED_INDEX",
                "params": {
                    "drop_ratio_build": self.MILVUS_INDEX_DROP_RATIO
                }
            }
        ]

    @property
    def MILVUS_SEARCH_PARAMS(self) -> List[Dict[str, Any]]:
        """Get Milvus search parameters."""
        return [
            {
                "metric_type": "COSINE",
                "params": {
                    "ef": self.MILVUS_SEARCH_EF
                }
            },
            {
                "metric_type": "BM25",
                "params": {
                    "drop_ratio_search": self.MILVUS_SEARCH_DROP_RATIO
                }
            }
        ]

# Create a singleton instance of the configuration
config = Config()