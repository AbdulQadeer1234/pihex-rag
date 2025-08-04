import logging
from typing import List, Optional, Dict, Any, Literal
from langchain_core.documents import Document
from app.config.config import config
from app.utils.milvus_utils import get_vector_store

logger = logging.getLogger(__name__)

async def retrieve_documents(
    query: str,
    k: Optional[int] = None,
    expr: Optional[str] = None,
    fetch_k: Optional[int] = None,
    ranker_type: Optional[Literal["rrf", "weighted"]] = None,
    ranker_params: Optional[Dict[str, Any]] = None,
    sparse_search: Optional[bool] = False,
    **kwargs: Any
) -> List[Document]:
    """
    Perform hybrid search combining dense vector search and sparse BM25 search.
    
    Args:
        query (str): The query string to search for
        k (Optional[int]): Number of documents to return
        expr (Optional[str]): Expression for metadata filtering
        fetch_k (Optional[int]): Number of results to fetch before ranking
        ranker_type (Optional[Literal["rrf", "weighted"]]): Type of ranker to use
        ranker_params (Optional[Dict[str, Any]]): Parameters for the ranker
        **kwargs: Additional arguments to pass to hybrid search
        
    Returns:
        List[Document]: List of retrieved documents
    """
    try:
        vector_store = await get_vector_store()
        if not vector_store:
            logger.error("Failed to get vector store")
            return []
        
        logger.info(f"Performing hybrid search for query: {query}")

        k = k or config.MILVUS_K
        fetch_k = fetch_k or config.fetch_k
        
        # Use search parameters from config
        search_params = config.MILVUS_SEARCH_PARAMS
        
        if sparse_search:
            results = await vector_store._acollection_hybrid_search(
            query=query,
            k=k,
            param=search_params,
            expr=expr,
            fetch_k=fetch_k,
            ranker_type=ranker_type or config.MILVUS_RANKER_TYPE,
            ranker_params=ranker_params or config.MILVUS_SPARSE_RANKER_PARAMS,
            **kwargs
        )
        
        else:
            results = await vector_store._acollection_hybrid_search(
                query=query,
                k=k,
                param=search_params,
                expr=expr,
                fetch_k=fetch_k,
                ranker_type=ranker_type or config.MILVUS_RANKER_TYPE,
                ranker_params=ranker_params or config.MILVUS_RANKER_PARAMS,
                **kwargs
            )
        
        if not results:
            return []
        # Convert results to Document format with proper metadata
        documents = []
        for hit in results[0]:  # First list contains hits for the first query
            entity = hit.get("entity", {})
            
            metadata = {
                "document_name": entity.get("document_inserted", ""),
                "section_name": entity.get("section_name", ""),
                "heading": entity.get("heading", ""),
                "sub_heading": entity.get("sub_heading", ""),
                "distance": hit.get("distance", 0.0)
            }
            
            doc = Document(
                page_content=entity.get("text", ""),
                metadata=metadata
            )
            documents.append(doc)
        logger.info(f"Retrieved {len(documents)} documents for query: {query}")
        return documents

    except Exception as e:
        logger.error(f"Error performing hybrid search: {e}", exc_info=True)
        return []
