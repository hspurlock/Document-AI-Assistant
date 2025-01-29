from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from pathlib import Path

class ProcessedFile(BaseModel):
    """Model for tracking processed files"""
    filename: str
    file_type: str
    checksum: str
    chunks: int
    processed_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict = Field(default_factory=dict)

class ChatMessage(BaseModel):
    """Model for chat messages"""
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatSession(BaseModel):
    """Model for chat sessions"""
    messages: List[ChatMessage] = Field(default_factory=list)
    context_files: List[str] = Field(default_factory=list)

class DocumentChunk(BaseModel):
    """Model for document chunks"""
    content: str
    metadata: Dict
    embedding: Optional[List[float]] = None
