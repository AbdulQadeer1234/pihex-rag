# Updated sections in: workflow.py

import logging
import asyncio
from typing import Dict, Any
from app.models.models import AnswerPayload
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from app.utils.llm_utils import get_llm_doc
from app.rag.retriever import retrieve_documents
from app.utils.prompts import PROMPT_TEMPLATE
from app.utils.rag_utils import prepare_document_context

logger = logging.getLogger(__name__)

async def process_query(question: str) -> Dict[str, Any]:
    """
    Given a user question, retrieve relevant documents, construct context, and get structured answer from LLM.
    Returns dict matching AnswerPayload schema.
    """
    # Retrieve relevant documents
    docs = await retrieve_documents(question, k=2)
    # context = "\n".join([doc.page_content for doc in docs]) if docs else ""
    # sources = [
    #     {"doc": doc.metadata.get("source", "unknown"), "snippet": doc.page_content[:120]} for doc in docs
    # ] if docs else []
    context = await prepare_document_context(docs)
    logger.info(f"Prepared context: {context[:200]}...")  # Log first 200 chars for brevity

    # Build prompt
    prompt_template = PROMPT_TEMPLATE
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("human", f"Context: {context}\nUser Question: {question}\nJSON Output:")
    ])

    llm = get_llm_doc()
    parser = JsonOutputParser(pydantic_object=AnswerPayload)

    # Run LLM and parse output
    chain = prompt | llm | parser
    result = await chain.ainvoke({"context": context, "question": question})
    logger.info(f"LLM result: {result}")

    # Parse and validate result against AnswerPayload schema
    try:
        validated = AnswerPayload.parse_obj(result)
    except Exception as e:
        logger.error(f"LLM output did not match AnswerPayload schema: {e}")
        raise ValueError("Invalid LLM output format")

    logger.info(f"Final structured answer: {validated}")
    return validated
