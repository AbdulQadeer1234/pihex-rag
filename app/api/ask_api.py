from fastapi import APIRouter, HTTPException
from app.models.models import AnswerPayload, QueryRequest
from app.src.workflow import process_query
import logging

logger = logging.getLogger(__name__)

ask_router = APIRouter()

@ask_router.post("/ask", response_model=AnswerPayload)
async def ask_question(request: QueryRequest):
    question = request.question
    logger.info(f"Received question: {question}")
    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' field.")
    
    # Call RAG chain or LLM with the prompt and question
    result = await process_query(question)

    result = AnswerPayload(
        answer = result.answer,
        category = result.category,
        confidence=result.confidence, 
        sources = result.sources
    )
    # Validate and return
    return result