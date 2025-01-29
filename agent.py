from pathlib import Path
from typing import List, Dict, Optional
from models import ProcessedFile, ChatMessage, ChatSession, DocumentChunk
from document_processor import DocumentProcessor
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

class AIAgent:
    """AI Agent for processing documents and answering questions"""
    
    def __init__(self, 
                 collection_name: str = "documents",
                 model_name: str = "deepseek-r1",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        # Initialize components
        self.doc_processor = DocumentProcessor()
        self.collection_name = collection_name
        
        # Initialize LLM
        self.llm = OllamaLLM(model=model_name)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize Qdrant client
        self.client = QdrantClient("localhost", port=6333)
        
        # Initialize vector store
        self._init_vector_store()
        
        # Initialize chat prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions based on the provided context. "
                      "If you don't know the answer, just say that you don't know. "
                      "Always provide your response in a clear, concise manner."),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])
        
        # Setup the retrieval chain
        self.retrieval_chain = (
            {"context": self.vector_store.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _init_vector_store(self):
        """Initialize or get existing vector store"""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        # Create collection if it doesn't exist
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # all-MiniLM-L6-v2 embedding size
                    distance=models.Distance.COSINE
                )
            )
        
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings
        )
    
    def process_file(self, file_path: Path) -> ProcessedFile:
        """Process a file and add it to the vector store"""
        # Process the file
        processed_file = self.doc_processor.process_file(file_path)
        
        # Get chunks with embeddings
        chunks = self.doc_processor._split_content(
            self.doc_processor._extract_content(file_path, processed_file.file_type),
            file_path,
            processed_file.file_type
        )
        
        # Add chunks to vector store
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        
        return processed_file
    
    def chat(self, question: str, session: Optional[ChatSession] = None) -> str:
        """Process a chat message and return the response"""
        try:
            # Get response from the chain
            response = self.retrieval_chain.invoke(question)
            
            # Update session if provided
            if session is not None:
                session.messages.append(ChatMessage(role="user", content=question))
                session.messages.append(ChatMessage(role="assistant", content=response))
            
            return response
            
        except Exception as e:
            error_msg = f"Error processing your question: {str(e)}"
            if session is not None:
                session.messages.append(ChatMessage(role="error", content=error_msg))
            return error_msg
