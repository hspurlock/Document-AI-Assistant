# Document AI Assistant

A powerful document processing and chat application built with Streamlit, LangChain, and Pydantic. This application can process various document types, embed their content, and allow you to chat with your documents using the Deepseek-r1 language model.

## Features

- Support for multiple document formats:
  - PDF
  - Word (DOCX)
  - PowerPoint (PPTX)
  - Excel (XLSX)
  - Text files (TXT)
  - Markdown (MD)
  - HTML
- Automatic text extraction and chunking
- Vector storage using Qdrant
- Chat interface using Deepseek-r1 model
- Clean and intuitive Streamlit interface

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU (recommended, but not required)

## Installation and Usage

### Option 1: Docker Deployment (Recommended)

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Start the application using Docker Compose:
```bash
docker compose up --build -d
```

3. Pull the Deepseek-r1 model:
```bash
docker exec -it document-ai-assistant-ollama-1 ollama pull deepseek-r1
```

4. Access the application at `http://localhost:8501`

### Option 2: Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the Docker containers for Qdrant and Ollama:
```bash
docker compose up -d qdrant ollama
```

5. Pull the Deepseek-r1 model:
```bash
docker exec -it document-ai-assistant-ollama-1 ollama pull deepseek-r1
```

6. Start the Streamlit application:
```bash
streamlit run app.py
```

7. Access the application at `http://localhost:8501`

## Usage Guide

1. Upload documents using the sidebar
2. Wait for the documents to be processed and embedded
3. Start chatting with your documents!

## Project Structure

- `app.py` - Streamlit web interface
- `agent.py` - Main AI agent implementation
- `document_processor.py` - Document processing logic
- `models.py` - Pydantic data models
- `docker-compose.yml` - Docker services configuration
- `Dockerfile` - Application container configuration

## Dependencies

- Streamlit - Web interface
- LangChain - LLM framework
- Pydantic - Data validation
- Qdrant - Vector database
- Ollama - Local LLM hosting
- Various document processing libraries (python-docx, python-pptx, etc.)

## License

MIT
