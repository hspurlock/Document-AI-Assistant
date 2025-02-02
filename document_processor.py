import hashlib
import magic
import io
import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from pypdf import PdfReader
from docx import Document
from PIL import Image
from bs4 import BeautifulSoup
import markdown
import chardet
import base64
from models import ProcessedFile, DocumentChunk, ImageMetadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from security import RateLimiter, validate_content
from langchain_core.messages import HumanMessage
from llm_provider import get_llm

class DocumentProcessor:
    """Handles processing of different document types"""
    
    # Initialize rate limiter as a class variable
    _rate_limiter = RateLimiter()
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        self.mime = magic.Magic(mime=True)
        
        # Initialize vision model with fallbacks
        try:
            self.vision_model = get_llm("llava")
        except Exception as e1:
            try:
                self.vision_model = get_llm("llama3.2-vision")
            except Exception as e2:
                self.vision_model = get_llm("llama2")
    
    def process_file(self, file_path: Path, original_filename: str = None) -> Optional[ProcessedFile]:
        """Process a document or image file and return chunks with metadata.
        
        Handles multiple file types:
        - Documents (PDF, DOCX, TXT): Extracts and chunks text
        - Images: Uses vision model to extract text and create chunks
        
        For images, the process includes:
        - Extracting text using Llava vision model
        - Creating unified chunks for vector storage
        - Preserving image metadata (dimensions, format)
        
        Args:
            file_path: Path to the file to process
            original_filename: Optional original name of the file
            
        Returns:
            ProcessedFile with chunks and metadata, or None if processing fails
        """
        try:
            # Check rate limit
            self._rate_limiter.check_rate_limit('process_file')
            
            # Get original filename if not provided
            if not original_filename:
                original_filename = file_path.name
            
            # Process file
            file_type = self._get_file_type(file_path)
            content, image_metadata = self._extract_content(file_path, file_type)
            
            # For non-image files, validate content and create chunks
            if not image_metadata:
                is_valid, error_msg = validate_content(content)
                if not is_valid:
                    raise ValueError(f"Invalid content: {error_msg}")
                chunks = self._split_content(content, file_path, file_type, original_filename)
            else:
                # For images, create chunks from image text
                chunks = [DocumentChunk(
                    text=chunk,
                    metadata={
                        "source": original_filename or str(file_path),
                        "page": "1",  # Images are single page
                        "chunk_type": "image_text",
                        "chunk_index": i,
                        "width": image_metadata.width,
                        "height": image_metadata.height,
                        "format": image_metadata.format
                    }
                ) for i, chunk in enumerate(image_metadata.chunks)]
            
            checksum = self._calculate_checksum(file_path)
            
            return ProcessedFile(
                filename=original_filename,
                file_type=file_type,
                checksum=checksum,
                chunks=chunks,
                metadata={"content": content} if bool(image_metadata) else {},
                is_image=bool(image_metadata),
                image_metadata=image_metadata
            )
        except Exception as e:
            return None
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type from extension"""
        ext = file_path.suffix.lower()[1:] if file_path.suffix else ''
        
        # Map extensions to types
        if ext in ['txt', 'md', 'html']:
            return ext
        elif ext == 'pdf':
            return 'pdf'
        elif ext == 'docx':
            return 'docx'
        elif ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp']:
            return 'image'
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _extract_content(self, file_path: Path, file_type: str) -> Tuple[str, Optional[ImageMetadata]]:
        """Extract text content from file based on type"""
        # Check if file is an image
        mime_type = self.mime.from_file(str(file_path))
        if mime_type.startswith('image/'):
            text, metadata = self._extract_image(file_path)
            if not metadata:
                raise ValueError(f"Failed to analyze image: {file_path.name}")
            return text, metadata
            
        # Handle other file types
        content = None
        if file_type == 'pdf':
            content = self._extract_pdf(file_path)
        elif file_type == 'docx':
            content = self._extract_docx(file_path)
        elif file_type in ['txt', 'md', 'html']:
            content = self._extract_text(file_path, file_type)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        # Validate non-image content
        is_valid, error = validate_content(content)
        if not is_valid:
            raise ValueError(f"Invalid content: {error}")
            
        return content, None
    
    def _split_content(self, content: str, file_path: Path, file_type: str, original_filename: str = None) -> List[DocumentChunk]:
        """Split content into chunks with metadata"""
        texts = self.text_splitter.split_text(content)
        chunks = []
        
        for i, text in enumerate(texts):
            # Determine page number (if available)
            page = i // 2 + 1  # Simple estimation, 2 chunks per page
            
            chunks.append(DocumentChunk(
                text=text,
                metadata={
                    "source": original_filename or str(file_path),
                    "page": str(page),
                    "chunk_type": "text",
                    "chunk_index": i
                }
            ))
        
        return chunks
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF"""
        reader = PdfReader(file_path)
        return "\n\n".join(page.extract_text() for page in reader.pages)
    
    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file.
        
        Extracts text while preserving:
        - Paragraphs with proper spacing
        - Tables (converts to text)
        - Lists and numbering
        - Headers and sections
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted text with preserved formatting
        """
        doc = Document(file_path)
        content = []
        
        # Extract headers
        for section in doc.sections:
            header = section.header
            if header.text.strip():
                content.append(header.text.strip())
        
        # Extract main content
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():  # Skip empty paragraphs
                content.append(paragraph.text.strip())
        
        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                row_text = ' | '.join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:  # Skip empty rows
                    content.append(row_text)
        
        return '\n\n'.join(content)
    
    def _extract_image(self, file_path: Path) -> Tuple[str, ImageMetadata]:
        """Extract text and metadata from image file using the Llava vision model.
        
        Uses a multi-stage process:
        1. Extract basic image metadata (dimensions, format)
        2. Process image with Llava for text extraction
        3. Create text chunks for vector storage
        4. Store all metadata including extracted text
        
        Falls back to llama3.2-vision or llama2 if Llava is unavailable.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Tuple containing:
            - Extracted text as a string
            - ImageMetadata with dimensions, format, and text chunks
        """
        text = ""
        metadata = None
        vision_error = None
        frame_count = 1
        
        try:
            # Create images directory if it doesn't exist
            images_dir = Path("images")
            images_dir.mkdir(exist_ok=True)
            
            # Copy image to images directory in Docker container
            target_path = Path("/app/images") / file_path.name
            target_path.parent.mkdir(exist_ok=True)
            shutil.copy2(file_path, target_path)
            
            # Get image metadata
            try:
                img = Image.open(target_path)
                metadata = ImageMetadata(
                    width=img.width,
                    height=img.height,
                    format=img.format,
                    mode=img.mode,
                    has_text=False,
                    detected_objects=[],
                    frame_count=1
                )
                img.close()
            except Exception as e:
                raise ValueError(f"Failed to extract image metadata: {str(e)}")
            
            # Convert image to base64 for Ollama
            import base64
            abs_image_path = target_path
            with open(abs_image_path, 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Process with vision model using direct Ollama API
            prompt = "Extract and list ALL text from this image exactly as it appears. Include every word, number, and symbol. Preserve line breaks and spacing."
            
            # Prepare request for Ollama API
            import requests
            ollama_url = "http://ollama:11434/api/generate"
            request_data = {
                "model": "llava",
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            }
            
            try:
                # Make request to Ollama API
                response = requests.post(ollama_url, json=request_data, timeout=60)
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                
                # Extract response text based on response format
                if 'message' in result:
                    # New format with message object
                    vision_response = result['message'].get('content', '')
                else:
                    # Old format with direct response
                    vision_response = result.get('response', '')
                    
                if not vision_response:
                    raise ValueError("Ollama API returned empty response")
                
                # Clean up the image file after successful processing
                if target_path.exists():
                    target_path.unlink()
            except Exception as e:
                # Clean up the image file in case of error
                if target_path.exists():
                    target_path.unlink()
                raise ValueError(f"Failed to process image with vision model: {str(e)}")
            
            # Store the vision model response and extract text
            metadata.vision_description = vision_response
            text = vision_response.strip()
            
            # Store the extracted text
            metadata.detected_text = text
            
            # Create a comprehensive text representation for vector store
            text_parts = [
                f"Image File: {file_path.name}",
                f"Dimensions: {metadata.width}x{metadata.height}",
                f"Format: {metadata.format}",
                "\nExtracted Text:",
                text
            ]
            
            # Join text parts and create chunks
            full_text = '\n'.join(text_parts)
            metadata.has_text = bool(metadata.detected_text)
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(full_text)
            
            # Store chunks in metadata
            metadata.chunks = chunks
            
            # Return the full text and metadata
            return full_text, metadata
            
        except Exception as e:
            # Create default metadata if none exists
            if metadata is None:
                metadata = ImageMetadata(
                    width=0,
                    height=0,
                    format="unknown",
                    mode="unknown",
                    has_text=False,
                    detected_objects=[],
                    frame_count=1,
                    vision_error=str(e)
                )
            return "", metadata
        return text, metadata
    
    def _extract_text(self, file_path: Path, file_type: str) -> str:
        """Extract text from text-based files"""
        # Detect encoding
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        
        # Read file with detected encoding
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()
            
        if file_type == 'md':
            # Convert markdown to plain text
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
        elif file_type == 'html':
            # Convert HTML to plain text
            soup = BeautifulSoup(content, 'html.parser')
            return soup.get_text()
        else:
            return content
