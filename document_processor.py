import hashlib
from pathlib import Path
from typing import List, Dict, Optional
import docx
from pptx import Presentation
from pypdf import PdfReader
import openpyxl
from bs4 import BeautifulSoup
import markdown
import chardet
from models import ProcessedFile, DocumentChunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Handles processing of different document types"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
    def process_file(self, file_path: Path) -> ProcessedFile:
        """Process a file and return chunks with metadata"""
        file_type = self._get_file_type(file_path)
        content = self._extract_content(file_path, file_type)
        chunks = self._split_content(content, file_path, file_type)
        checksum = self._calculate_checksum(file_path)
        
        return ProcessedFile(
            filename=file_path.name,
            file_type=file_type,
            checksum=checksum,
            chunks=len(chunks),
            metadata={"path": str(file_path)}
        )
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type from extension"""
        return file_path.suffix.lower()[1:]
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _extract_content(self, file_path: Path, file_type: str) -> str:
        """Extract text content from file based on type"""
        if file_type == 'pdf':
            return self._extract_pdf(file_path)
        elif file_type == 'docx':
            return self._extract_docx(file_path)
        elif file_type == 'pptx':
            return self._extract_pptx(file_path)
        elif file_type == 'xlsx':
            return self._extract_xlsx(file_path)
        elif file_type in ['txt', 'md', 'html']:
            return self._extract_text(file_path, file_type)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _split_content(self, content: str, file_path: Path, file_type: str) -> List[DocumentChunk]:
        """Split content into chunks"""
        texts = self.text_splitter.split_text(content)
        return [
            DocumentChunk(
                content=chunk,
                metadata={
                    "source": file_path.name,
                    "file_type": file_type,
                    "chunk_index": i
                }
            )
            for i, chunk in enumerate(texts)
        ]
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF"""
        reader = PdfReader(file_path)
        return "\n\n".join(page.extract_text() for page in reader.pages)
    
    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX"""
        doc = docx.Document(file_path)
        return "\n\n".join(paragraph.text for paragraph in doc.paragraphs)
    
    def _extract_pptx(self, file_path: Path) -> str:
        """Extract text from PPTX"""
        prs = Presentation(file_path)
        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_runs.append(shape.text)
        return "\n\n".join(text_runs)
    
    def _extract_xlsx(self, file_path: Path) -> str:
        """Extract text from XLSX"""
        wb = openpyxl.load_workbook(file_path, data_only=True)
        texts = []
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            sheet_texts = []
            for row in ws.iter_rows():
                row_texts = [str(cell.value) for cell in row if cell.value is not None]
                if row_texts:
                    sheet_texts.append(" | ".join(row_texts))
            if sheet_texts:
                texts.append(f"Sheet: {sheet}\n" + "\n".join(sheet_texts))
        return "\n\n".join(texts)
    
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
