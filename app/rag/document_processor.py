import os
import re
import hashlib
import json
from typing import List, Dict, Union, TextIO, Iterator, Tuple, IO
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
import logging

logger = logging.getLogger(__name__)

def load_markdown_content(file_path_or_obj: Union[str, TextIO, IO[bytes]]) -> str: 
    """Load content from either a file path or a file-like object."""
    try:
        if isinstance(file_path_or_obj, (str, bytes, os.PathLike)):
            with open(file_path_or_obj, 'r', encoding='utf-8') as f:
                return f.read()
        elif hasattr(file_path_or_obj, 'read'):
            # It's a file-like object
            current_pos = None
            if hasattr(file_path_or_obj, 'tell') and hasattr(file_path_or_obj, 'seek'):
                try:
                    current_pos = file_path_or_obj.tell()
                    file_path_or_obj.seek(0)  # Go to start of file
                except Exception: # pragma: no cover
                    current_pos = None
            content = file_path_or_obj.read()

            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            if current_pos is not None and hasattr(file_path_or_obj, 'seek'):
                try:
                    file_path_or_obj.seek(current_pos)  # Restore position
                except Exception: 
                    pass # Non-seekable stream after read
            return content
        else:
            raise ValueError("Input must be either a file path or a readable file-like object")
    except Exception as e:
        logger.error(f"Error reading markdown content: {str(e)}")
        raise ValueError(f"Error reading markdown content: {str(e)}")


def load_and_chunk_document(file_path_or_obj: Union[str, TextIO, IO[bytes]]) -> List[Document]:
    """Loads a markdown document and splits it into header-based chunks."""
    doc_name = (
        os.path.basename(file_path_or_obj)
        if isinstance(file_path_or_obj, (str, bytes, os.PathLike))
        else getattr(file_path_or_obj, 'name', '')
    )
    
    try:
        content_str = load_markdown_content(file_path_or_obj)
    except ValueError as e:
        logger.error(f"Failed to load document {doc_name}: {e}", exc_info=True)
        return []

    headers_to_split_on = [
        ("#", "header1"),
        ("##", "header2"),
        ("###", "header3"),
    ]
    
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    try:
        split_docs = splitter.split_text(content_str)
        
        chunks = []
        for d in split_docs:
            metadata = d.metadata if hasattr(d, 'metadata') else {}
            chunks.append(Document(
                page_content=d.page_content,
                metadata={
                    "document_name": doc_name,
                    "section_name": metadata.get("header1", "").strip(),
                    "heading": metadata.get("header2", "").strip(),
                    "sub_heading": metadata.get("header3", "").strip(),
                }
            ))

        logger.info(f"Generated {len(chunks)} chunks for document: {doc_name}.")
        return chunks
    except Exception as e:
        logger.error(f"Error during chunking of document {doc_name}: {e}", exc_info=True)
        raise

