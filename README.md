# Document AI Assistant

A powerful document processing and chat application built with Streamlit, LangChain, and Qdrant. This application processes various document types, embeds their content for semantic search, and provides an intelligent chat interface powered by large language models.

## Features

### Document and Image Processing
- Support for multiple formats:
  - PDF documents
  - Word documents (DOCX)
  - Text files (TXT)
  - Images (PNG, JPG, JPEG, GIF, BMP)
- Advanced image analysis:
  - AI-powered image descriptions
  - Object detection and recognition
  - Text extraction from images (OCR)
  - Fallback processing for complex images
- Automatic text extraction and smart chunking
- Progress tracking for large documents
- Clear error handling and status messages

### Smart Chat Interface
- Context-aware responses from documents
- Source attribution for answers
- Document-specific querying
- Real-time chat with thinking indicators
- Multiple LLM model support

### Document Management
- Selective document search
- Document statistics (chunks, pages, words)
- Easy document deletion
- Upload status tracking

### Security & Performance
- Rate limiting protection
- Input sanitization
- Efficient vector search with Qdrant
- Scalable architecture

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU (recommended, but not required)

## Screenshots

![Document AI Assistant Interface](docs/images/Screenshot%20from%202025-01-31%2010-23-12.png)

*The Document AI Assistant provides an intuitive interface for uploading, managing, and chatting with your documents.*

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

3. Pull required models for Ollama:
```bash
docker exec -it document-ai-assistant-ollama-1 ollama pull llama3.2
docker exec -it document-ai-assistant-ollama-1 ollama pull deepseek-r1
docker exec -it document-ai-assistant-ollama-1 ollama pull llama3.2-vision
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

5. Pull required models for Ollama:
```bash
docker exec -it document-ai-assistant-ollama-1 ollama pull llama3.2
docker exec -it document-ai-assistant-ollama-1 ollama pull deepseek-r1
docker exec -it document-ai-assistant-ollama-1 ollama pull llama3.2-vision
```

6. Start the Streamlit application:
```bash
streamlit run app.py
```

## Technical Details

### Document Processing

- **Text Extraction**: Uses specialized libraries for each document type:
  - PDF: PyPDF for text extraction
  - DOCX: python-docx for parsing
  - TXT/MD: Direct reading with encoding detection
  - Images: Multi-stage processing pipeline

- **Image Analysis**:
  - Uses llama3.2-vision for image processing:
    - Advanced text extraction from images
    - Rich image descriptions
    - Object detection and classification
    - Direct API integration with Ollama
  - Optimized for accuracy and performance
    - Fast processing of complex images
    - High-quality text extraction
  - Enhanced Metadata:
    - Image dimensions and format
    - Detected objects and text
    - Vision model descriptions
    - Automatic text chunking for vector storage

- **Content Processing**:
  - Smart text chunking with overlap
  - Unified chunking for both documents and images
  - Rich metadata extraction and preservation
  - Vector embedding for semantic search
  - Specialized image text representation
  - Automatic chunk size optimization

### Vector Search

- Uses Qdrant for efficient similarity search
- Enhanced metadata for images and documents
  - Image dimensions and format
  - Page numbers and chunk types
  - Source tracking and checksums
- Optimized chunk size for better context
- Unified vector storage for all content types

### Security

- Rate limiting on API endpoints
- Input sanitization and validation
- Secure file handling with cleanup
- Temporary storage for uploads

7. Access the application at `http://localhost:8501`

## Usage Guide

1. **Upload Documents and Images**
   - Use the file uploader in the sidebar
   - Supported formats:
     - Documents: PDF, DOCX, TXT
     - Images: PNG, JPG, JPEG, GIF, BMP
   - Monitor upload progress and status messages
   - Real-time processing feedback
   - Automatic text extraction from images

2. **Select Documents to Query**
   - Choose specific documents in the sidebar to focus your search
   - Leave all unselected to search across all documents
   - View document statistics (chunks, pages, words)
   - Remove documents you no longer need

3. **Chat with Your Documents**
   - Ask questions in the chat interface
   - Get context-aware responses from selected documents
   - See source documents used for each answer
   - Watch real-time response indicators
   - Choose different LLM models for responses

## Project Structure

- `app.py` - Streamlit interface and chat functionality
- `agent.py` - AI agent with document querying and LLM integration
- `document_processor.py` - Document parsing and chunking
- `models.py` - Data models and validation
- `security.py` - Rate limiting and input sanitization
- `docker-compose.yml` - Container orchestration
- `Dockerfile` - Application containerization

## Core Components

### Frontend
- **Streamlit**: Interactive web interface
- **Real-time Updates**: Progress indicators and status messages
- **Document Management**: Upload, select, and delete functionality

### Backend
- **LangChain**: Document processing and LLM integration
- **Qdrant**: Vector storage and semantic search
- **Ollama**: Local LLM hosting and inference

### Processing
- **Document Handling**: PDF, DOCX, and TXT support
- **Text Extraction**: Automatic content parsing
- **Chunking**: Smart text segmentation

### Security
- **Rate Limiting**: Request throttling
- **Input Validation**: Content sanitization
- **Error Handling**: Clear error messages

## License

MIT
