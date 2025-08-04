import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from langchain_core.documents import Document
from app.config.config import config
from app.models.models import IngestResponse
from app.rag.document_processor import load_and_chunk_document
from app.utils.milvus_utils import index_document_chunks

logger = logging.getLogger(__name__)
ingest_router = APIRouter()

@ingest_router.post("/ingest",
             response_model=IngestResponse,
             summary="Upload and index one or multiple documents",
             tags=["Ingestion"])
async def ingest_documents(
    file: List[UploadFile] = File(...)
):
    """
    Accepts one or multiple document files (e.g., Markdown) and processes them for chunking and indexing.
    Returns success only after all documents are successfully indexed.
    """
    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")

    processed_files = []
    all_documents = []

    try:
        # Process all files
        for current_file in file:
            if not current_file.filename:
                continue

            try:
                # Read file content
                content = await current_file.read()
                file_content = content.decode('utf-8')
                
                # Create a temporary file-like object
                from io import StringIO
                file_obj = StringIO(file_content)
                file_obj.name = current_file.filename
                
                # Process the document
                documents = load_and_chunk_document(file_obj)
                
                if documents:
                    all_documents.extend(documents)
                    processed_files.append(current_file.filename)
                    logger.info(f"Successfully processed file: {current_file.filename}")
                else:
                    logger.warning(f"No chunks generated from {current_file.filename}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"No valid content found in {current_file.filename}"
                    )

            except Exception as e:
                logger.error(f"Failed to process file {current_file.filename}: {e}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Could not process file {current_file.filename}: {str(e)}"
                )
            finally:
                await current_file.close()

        # Index all documents synchronously
        if all_documents:
            vector_store = await index_document_chunks(all_documents, config.MILVUS_COLLECTION_NAME)
            if not vector_store:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to index documents"
                )
            logger.info(f"Successfully indexed {len(all_documents)} chunks from {len(processed_files)} files")
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid documents to index"
            )

    except Exception as e:
        logger.error(f"Failed during document processing and indexing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing and indexing failed: {str(e)}"
        )

    return IngestResponse(
        message=f"Successfully processed and indexed {len(processed_files)} files",
        filename=", ".join(processed_files)
    )