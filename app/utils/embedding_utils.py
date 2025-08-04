import logging
from langchain_openai import OpenAIEmbeddings
from app.config.config import config

logger = logging.getLogger(__name__)

# --- Dense Embeddings ---
_dense_embedding_model = None

def get_dense_embedding_model() -> OpenAIEmbeddings:
    """Initializes and returns the dense embedding model client (via vLLM)."""
    global _dense_embedding_model
    if _dense_embedding_model is None:
        logger.info(f"Initializing dense embedding model: {config.EMBEDDING_MODEL_NAME} via {config.VLLM_EMBEDDING_URL}")
        try:
            _dense_embedding_model = OpenAIEmbeddings(
                model=config.EMBEDDING_MODEL_NAME,
                openai_api_base=config.VLLM_EMBEDDING_URL,
                openai_api_key=config.LLM_API_KEY
            )
            logger.info("Dense embedding model initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize dense embedding model: {e}", exc_info=True)
            raise
    return _dense_embedding_model
