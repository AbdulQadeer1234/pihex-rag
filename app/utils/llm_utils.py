"""LLM configuration and initialization"""
import os
from langchain_openai import ChatOpenAI
from app.config.config import config    
from langchain_community.llms import VLLMOpenAI

import logging

logger = logging.getLogger(__name__)

class LLMError(Exception):
    """Custom exception for LLM-related errors"""
    pass


def get_llm(
    temperature: float = 0,
    max_tokens: int = 10,
    **kwargs
):
    """
    Get a configured LLM instance with caching.
    
    Args:
        max_tokens: Optional override for token limit
        temperature: Optional override for temperature
        model_kwargs: Optional additional model parameters
        
    Returns:
        VLLMOpenAI: Configured LLM instance
        
    Raises:
        LLMError: If LLM initialization fails
    """
    try:
        return ChatOpenAI(
            openai_api_key=config.LLM_API_KEY,
            openai_api_base=config.LLM_API_BASE,
            model_name=config.LLM_MODEL_NAME,
            temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
            max_tokens=max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
            model_kwargs={**kwargs}
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise LLMError(f"LLM initialization failed: {str(e)}")
    


def get_llm_doc(
    temperature: float = 0,
    max_tokens: int = 10,
    **kwargs
):
    """
    Get a configured LLM instance with caching.
    
    Args:
        max_tokens: Optional override for token limit
        temperature: Optional override for temperature
        model_kwargs: Optional additional model parameters
        
    Returns:
        VLLMOpenAI: Configured LLM instance
        
    Raises:
        LLMError: If LLM initialization fails
    """
    try:
        return VLLMOpenAI(
            openai_api_key=config.LLM_API_KEY,
            openai_api_base=config.LLM_API_BASE,
            model_name=config.LLM_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            model_kwargs={**kwargs}
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise LLMError(f"LLM initialization failed: {str(e)}")