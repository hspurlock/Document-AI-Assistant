"""LLM provider module to avoid circular dependencies"""

import os
from langchain_ollama import ChatOllama

def get_llm(model_name: str = "llama2") -> ChatOllama:
    """Get an LLM instance with the specified model.
    
    Supports both chat and vision models:
    - llama2: Standard chat model
    - llama3.2-vision: Alternative vision model
    
    Args:
        model_name: Name of the Ollama model to use
        
    Returns:
        ChatOllama instance configured for chat or vision
        
    Raises:
        ConnectionError: If cannot connect to Ollama server
        ValueError: If model initialization fails
    """
    ollama_base_url = "http://document-ai-assistant-ollama-1:11434"
    
    try:
        # Test connection to Ollama
        import requests
        try:
            response = requests.get(f"{ollama_base_url}/api/tags")
            response.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Ollama server: {str(e)}")
        
        # Check if model exists and install if needed
        try:
            response = requests.get(f"{ollama_base_url}/api/show", json={"name": model_name})
            if response.status_code == 404:
                print(f"Model {model_name} not found. Attempting to pull...")
                response = requests.post(
                    f"{ollama_base_url}/api/pull",
                    json={"name": model_name},
                    stream=True
                )
                response.raise_for_status()
                print(f"Successfully pulled model {model_name}")
            else:
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to check/pull model {model_name}: {str(e)}")
            
        # Initialize the LLM with vision support if needed
        kwargs = {
            "model": model_name,
            "base_url": ollama_base_url,
            "temperature": 0.0,  # Use deterministic responses
            "streaming": False,
            "request_timeout": 120,  # 120 second timeout for vision
            "num_ctx": 4096  # Larger context window
        }
        
        # Add vision-specific settings
        if model_name in ["llava", "llama3.2-vision"]:
            kwargs["format"] = "json"
            kwargs["seed"] = 42  # For consistent responses
            kwargs["num_predict"] = 1024  # Longer responses for image description
        
        return ChatOllama(**kwargs)
    except Exception as e:
        raise ValueError(f"Failed to initialize LLM with model {model_name}: {str(e)}")
