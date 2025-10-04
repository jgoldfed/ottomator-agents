"""
Provider utilities for configuring Ollama-backed models.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.ollama import OllamaProvider

# Load environment variables
load_dotenv()

# Default values for local Ollama setup
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LLM_MODEL = "gpt-oss:120b-cloud"
DEFAULT_EMBEDDING_MODEL = "mxbai-embed-large:latest"


def _get_base_url() -> str:
    """Return the Ollama base URL, defaulting to the local instance."""
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)


def _get_api_key() -> str:
    """Return the Ollama API key or a placeholder if not required."""
    return os.getenv("OLLAMA_API_KEY", "ollama")


def _normalize_model_name(model_name: str) -> str:
    """Strip provider prefixes like 'ollama:' from model names."""
    if model_name.lower().startswith("ollama:"):
        return model_name.split(":", 1)[1]
    return model_name


@lru_cache(maxsize=1)
def get_llm_model() -> OpenAIModel:
    """Get an `OpenAIModel` configured to talk to Ollama."""
    raw_choice = os.getenv("LLM_CHOICE", DEFAULT_LLM_MODEL)
    llm_choice = _normalize_model_name(raw_choice)
    
    # Use OllamaProvider with the base URL
    provider = OllamaProvider(
        base_url=_get_base_url(),
        api_key=_get_api_key(),
    )
    
    return OpenAIModel(llm_choice, provider=provider)


@lru_cache(maxsize=1)
def get_embedding_client() -> AsyncOpenAI:
    """Return an AsyncOpenAI client pointed at Ollama for embeddings."""
    return AsyncOpenAI(
        base_url=_get_base_url(),
        api_key=_get_api_key(),
    )


def get_embedding_model() -> str:
    """Return the embedding model name."""
    return os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_ingestion_model() -> OpenAIModel:
    """Return model configuration used during ingestion tasks."""
    return get_llm_model()


def validate_configuration() -> bool:
    """Validate that critical environment variables are present."""
    required_vars = ["DATABASE_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False

    return True


def get_model_info() -> dict:
    """Return information about the active model configuration."""
    return {
        "llm_provider": "ollama",
        "llm_model": _normalize_model_name(os.getenv("LLM_CHOICE", DEFAULT_LLM_MODEL)),
        "embedding_provider": "ollama",
        "embedding_model": _normalize_model_name(get_embedding_model()),
        "base_url": _get_base_url(),
    }