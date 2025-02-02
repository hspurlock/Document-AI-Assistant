"""LLM provider module to avoid circular dependencies"""

import os
from langchain_ollama import ChatOllama

def get_llm(model_name: str = "llama2") -> ChatOllama:
    """Get an LLM instance with the specified model.
    
    This function is used for chat functionality in the agent.
    For image processing, we use direct Ollama API calls instead.
    
    Args:
        model_name: Name of the Ollama model to use
        
    Returns:
        ChatOllama instance configured for chat
        
    Raises:
        ConnectionError: If cannot connect to Ollama server
        ValueError: If model initialization fails
    """
    ollama_base_url = "http://ollama:11434"
    
    try:
        # Test connection to Ollama
        import requests
        try:
            response = requests.get(f"{ollama_base_url}/api/tags")
            response.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama server: {str(e)}")
        
        # Initialize the LLM
        return ChatOllama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0.0,  # Use deterministic responses
            streaming=False,
            request_timeout=60,  # 60 second timeout
            num_ctx=4096  # Larger context window
        )
    except Exception as e:
        raise ValueError(f"Failed to initialize LLM with model {model_name}: {str(e)}")
