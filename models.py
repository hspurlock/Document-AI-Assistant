from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path

class DocumentChunk(BaseModel):
    """Model for document chunks"""
    text: str
    metadata: Dict
    embedding: Optional[List[float]] = None

class ImageMetadata(BaseModel):
    """Model for image-specific metadata and extracted text.
    
    Stores both basic image information and AI-extracted content:
    - Basic: dimensions, format, mode
    - Vision: AI description, detected objects
    - Text: Extracted text and chunked content
    - Error tracking: Vision model errors
    
    The chunks field stores preprocessed text segments ready for
    vector storage, maintaining consistency with document processing.
    """
    width: int
    height: int
    format: str
    mode: str
    has_text: bool = False
    frame_count: int = 1                     # Number of frames (for MPO files)
    vision_error: Optional[str] = None
    vision_description: Optional[str] = None  # AI-generated description of the image
    detected_text: Optional[str] = None      # Text detected in the image
    detected_objects: List[str] = []         # List of objects detected in the image
    chunks: List[str] = []                   # Chunked text for vector store

class ProcessedFile(BaseModel):
    """Model for tracking processed files"""
    filename: str
    file_type: str
    checksum: str
    chunks: List[DocumentChunk]
    processed_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict = Field(default_factory=dict)
    is_image: bool = False
    image_metadata: Optional[ImageMetadata] = None

class ChatMessage(BaseModel):
    """Model for chat messages"""
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatSession(BaseModel):
    """Model for chat sessions"""
    messages: List[ChatMessage] = Field(default_factory=list)
    context_files: List[str] = Field(default_factory=list)
