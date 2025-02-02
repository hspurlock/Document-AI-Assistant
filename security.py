from functools import wraps
from time import time
from collections import defaultdict
import html
import re
from typing import Optional, Callable, Any

# Rate limiting
class RateLimiter:
    def __init__(self):
        self._calls = defaultdict(list)
        self.max_calls = 5
        self.time_frame = 60  # seconds
    
    def check_rate_limit(self, func_name: str) -> None:
        """Check if the rate limit has been exceeded for a given function"""
        now = time()
        
        # Clean old calls
        self._calls[func_name] = [
            t for t in self._calls[func_name]
            if now - t < self.time_frame
        ]
        
        # Check if rate limit exceeded
        if len(self._calls[func_name]) >= self.max_calls:
            oldest_call = min(self._calls[func_name])
            remaining = self.time_frame - (now - oldest_call)
            raise Exception(f"Rate limit exceeded. Try again in {remaining:.1f} seconds.")
        
        # Add new call
        self._calls[func_name].append(now)

# Input sanitization
def sanitize_input(text: str) -> str:
    """Sanitize input text to prevent XSS and other injection attacks."""
    if not isinstance(text, str):
        return ""
    # Remove potentially dangerous HTML
    text = html.escape(text)
    # Remove potential script injections
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'data:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'vbscript:', '', text, flags=re.IGNORECASE)
    return text

# Content validation
def validate_content(content: str, max_length: int = 10_000_000) -> tuple[bool, Optional[str]]:
    """Validate content for security concerns."""
    if not content:
        return False, "Content is empty"
    
    if len(content) > max_length:
        return False, f"Content exceeds maximum length of {max_length} characters"
    
    # Add more content validation rules here
    return True, None

# File validation
class FileValidator:
    # Document extensions
    DOCUMENT_EXTENSIONS = {'txt', 'md', 'docx', 'pdf'}
    
    # Image extensions
    IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    
    # All allowed extensions
    ALLOWED_EXTENSIONS = DOCUMENT_EXTENSIONS | IMAGE_EXTENSIONS
    
    # Size limits
    MAX_DOCUMENT_SIZE = 10 * 1024 * 1024  # 10MB for documents
    MAX_IMAGE_SIZE = 5 * 1024 * 1024      # 5MB for images
    
    @classmethod
    def validate_file(cls, filename: str, file_size: int) -> tuple[bool, Optional[str]]:
        """Validate file properties."""
        # Check file extension
        if '.' not in filename:
            return False, "No file extension found"
        
        ext = filename.rsplit('.', 1)[1].lower()
        if ext not in cls.ALLOWED_EXTENSIONS:
            return False, (f"File type '{ext}' not allowed. Supported types: " +
                          f"Documents: {', '.join(sorted(cls.DOCUMENT_EXTENSIONS))}, " +
                          f"Images: {', '.join(sorted(cls.IMAGE_EXTENSIONS))}")
        
        # Check file size based on type
        if ext in cls.IMAGE_EXTENSIONS:
            if file_size > cls.MAX_IMAGE_SIZE:
                return False, f"Image size exceeds maximum limit of {cls.MAX_IMAGE_SIZE/1024/1024:.1f}MB"
        else:
            if file_size > cls.MAX_DOCUMENT_SIZE:
                return False, f"Document size exceeds maximum limit of {cls.MAX_DOCUMENT_SIZE/1024/1024:.1f}MB"
        
        return True, None

# Session management
class SessionManager:
    SESSION_TIMEOUT = 3600  # 1 hour
    
    @staticmethod
    def check_session_timeout(last_activity: float) -> bool:
        """Check if session has timed out."""
        return (time() - last_activity) > SessionManager.SESSION_TIMEOUT
