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
- Python 3.8+
- NVIDIA GPU (recommended)

## Installation

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

4. Start the Docker containers:
```bash
docker-compose up -d
```

5. Pull the Deepseek-r1 model:
```bash
docker exec -it pydantic_test-ollama-1 ollama pull deepseek-r1
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload documents using the sidebar

4. Start chatting with your documents!

## Project Structure

- `app.py` - Streamlit web interface
- `agent.py` - Main AI agent implementation
- `document_processor.py` - Document processing logic
- `models.py` - Pydantic data models
- `docker-compose.yml` - Docker services configuration

## Dependencies

- Streamlit - Web interface
- LangChain - LLM framework
- Pydantic - Data validation
- Qdrant - Vector database
- Ollama - Local LLM hosting
- Various document processing libraries (python-docx, python-pptx, etc.)

## License

MIT
