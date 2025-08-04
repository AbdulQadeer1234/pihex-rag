import logging
from typing import List
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

async def prepare_document_context(docs: List[Document]) -> str:
    """
    Prepare document context for the LLM by formatting documents with metadata.
    Removes duplicate documents to reduce context size.
    
    Args:
        docs (List[Document]): List of documents to format
        
    Returns:
        str: Formatted context string
    """
    if not docs:
        return ""
        
    # Track seen contents to avoid duplicates
    seen_contents = set()
    context_parts = []
    
    for i, doc in enumerate(docs, start=1):
        try:
            metadata = doc.metadata.get('metadata', {}) if isinstance(doc.metadata.get('metadata'), dict) else doc.metadata
            content = doc.page_content
            
            # Skip if we've seen this content before
            if content in seen_contents:
                continue
            seen_contents.add(content)

            context_part = (
                f"document_name: {metadata.get('document_name', '')} \n"
                f"section_name: {metadata.get('section_name', '')} \n"
                f"heading: {metadata.get('heading', '')} \n"
                f"sub_heading: {metadata.get('sub_heading', '')} \n"
                f"page_content: {content} \n"
            )
            context_parts.append(context_part)
            
        except Exception as doc_error:
            logger.error(f"Error processing document {i}: {doc_error}", exc_info=True)
            continue
    
    return "\n\n".join(context_parts)