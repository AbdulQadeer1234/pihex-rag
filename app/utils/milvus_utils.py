# app/utils/milvus_utils.py
import logging
from typing import List
from pymilvus import connections, db, Collection
from langchain_core.documents import Document
from app.config.config import config
from app.utils.embedding_utils import (get_dense_embedding_model)
from langchain_milvus import Milvus, BM25BuiltInFunction
import threading

logger = logging.getLogger(__name__)

# Thread-safe singleton pattern
_vector_store_lock = threading.Lock()
_vector_store_instance = None

def initialize_embeddings():
    return get_dense_embedding_model()

def setup_milvus_database(db_name=config.MILVUS_DB_NAME) -> bool:
    """Setup Milvus database connection."""
    try:
        # Remove any existing connections
        if connections.has_connection("default"):
            connections.disconnect("default")
            
        # Connect with proper host string format
        connections.connect(
            alias="default",
            host=str(config.MILVUS_HOST),  # Ensure host is str type
            port=config.MILVUS_PORT,
            timeout=config.MILVUS_TIMEOUT
        )

        # Create or switch to database
        if db_name in db.list_database():
            db.using_database(db_name)
        else:
            db.create_database(db_name)
            db.using_database(db_name)
        
        logger.info(f"Successfully connected to Milvus and initialized database: {db_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to setup Milvus database: {str(e)}", exc_info=True)
        return False

def get_total_documents_in_collection(collection_name: str) -> int:
    """Return total number of documents (entities) in a Milvus collection."""
    try:
        collection = Collection(name=collection_name)
        collection.load()
        return collection.num_entities
    except Exception as e:
        logger.error(f"Error fetching document count for collection '{collection_name}': {e}")
        return -1
    
async def create_vector_store(
    documents: List[Document], 
    embeddings, 
    db_name: str = config.MILVUS_DB_NAME, 
    collection_name: str = config.MILVUS_COLLECTION_NAME
) -> Milvus:
    """Create and populate Milvus vector store."""
    try:
        # Ensure database connection
        if not setup_milvus_database(db_name):
            logger.error("Failed to setup Milvus database")
            return None

        # Create vector store with proper configuration
        vector_store = Milvus.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            connection_args={
                "host": config.MILVUS_HOST,
                "port": config.MILVUS_PORT,
                "user": config.MILVUS_USER,
                "password": config.MILVUS_PASSWORD,
                "db_name": db_name,
                "timeout": config.MILVUS_TIMEOUT
            },
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
            consistency_level="Strong",
            index_params=config.MILVUS_INDEX_PARAMS,
            search_params=config.MILVUS_SEARCH_PARAMS
        )

        # Verify collection was created
        if not vector_store:
            logger.error("Failed to create vector store")
            return None

        logger.info(f"Created vector store with {len(documents)} documents in collection '{collection_name}'")
        total_docs = get_total_documents_in_collection(collection_name)
        logger.info(f"Total documents now in collection '{collection_name}': {total_docs}")
        return vector_store

    except Exception as e:
        logger.error(f"Error creating vector store: {e}", exc_info=True)
        return None


async def get_vector_store(force_reinit: bool = False) -> Milvus:
    """Thread-safe singleton pattern for vector store."""
    global _vector_store_instance
    
    if not force_reinit and _vector_store_instance is not None:
        logger.info("Returning existing vector store instance")
        return _vector_store_instance
        
    with _vector_store_lock:
        if not force_reinit and _vector_store_instance is not None:
            return _vector_store_instance
            
        embeddings = initialize_embeddings()
        if not embeddings:
            logger.error("Failed to initialize embeddings")
            return None
            
        _vector_store_instance = await create_vector_store(
            documents=[],
            embeddings=embeddings,
            db_name=config.MILVUS_DB_NAME,
            collection_name=config.MILVUS_COLLECTION_NAME
        )
        logger.info("Created new vector store instance")
        return _vector_store_instance

async def index_document_chunks(documents: List[Document], collection_name: str = config.MILVUS_COLLECTION_NAME, alias: str = "default", batch_size: int = 128):
    """
    Inserts document chunks into the specified Milvus collection using the existing vector store.
    """
    if not documents:
        logger.warning("No document chunks provided for indexing.")
        return None
        
    try:
        # Get existing vector store instance
        vector_store = await get_vector_store()
        if not vector_store:
            logger.error("Failed to get vector store")
            return None

        vector_store.add_documents(documents)

        logger.info(f"Indexed {len(documents)} documents into collection '{collection_name}'")

        total_docs = get_total_documents_in_collection(collection_name)
        if total_docs >= 0:
            logger.info(f"Total documents in collection '{collection_name}': {total_docs}")
        return vector_store

    except Exception as e:
        logger.error(f"Error in index_document_chunks: {e}", exc_info=True)
        return None