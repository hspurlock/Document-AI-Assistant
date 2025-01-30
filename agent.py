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
from qdrant_client.models import Filter, FieldCondition, MatchValue
import os
import uuid

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
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.llm = OllamaLLM(model=model_name, base_url=ollama_base_url)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize Qdrant client
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.client = QdrantClient(qdrant_host, port=qdrant_port)
        
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
    
    def _delete_document_vectors(self, filename: str):
        """Delete all vectors associated with a specific document"""
        try:
            # Create a filter for the specific document
            file_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=filename)
                    )
                ]
            )
            
            # Delete points with the filter
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=file_filter
            )
            return True
        except Exception as e:
            print(f"Error deleting vectors for {filename}: {str(e)}")
            return False
    
    def delete_document(self, source: str) -> bool:
        """Delete a document from the vector store"""
        try:
            # Delete all points for this document
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=source)
                            )
                        ]
                    )
                )
            )
            return True
        except Exception as e:
            print(f"Error deleting document {source}: {str(e)}")
            return False
    
    def process_file(self, file_path: Path, is_update: bool = False, original_filename: str = None) -> Optional[ProcessedFile]:
        """Process a file and store its contents in the vector store"""
        try:
            # Process the file
            processed_file = self.doc_processor.process_file(file_path, original_filename)
            if not processed_file:
                return None
            
            # Create points for vector store
            points = []
            for chunk in processed_file.chunks:
                points.append(models.PointStruct(
                    id=str(uuid.uuid4()),
                    payload={
                        'text': chunk.text,
                        'source': original_filename or str(file_path),  # Use original filename if available
                        'filename': original_filename or file_path.name,
                        'checksum': processed_file.checksum,
                        'page': chunk.metadata.get('page', ''),
                        'chunk_type': chunk.metadata.get('chunk_type', 'text'),
                    },
                    vector=self.embeddings.embed_query(chunk.text)
                ))
            
            # Update vector store
            if is_update:
                # Delete existing points for this file
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="filename",  # Use filename instead of source
                                    match=models.MatchValue(value=original_filename or file_path.name)
                                )
                            ]
                        )
                    )
                )
            
            # Insert new points
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            return processed_file
        except Exception as e:
            print(f"Error in process_file: {str(e)}")
            return None
    
    def get_vector_store_contents(self, limit: int = 1000) -> Dict:
        """Get contents of the vector store organized by document"""
        try:
            # Get all points from the collection
            response = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            # Organize results by document
            documents = {}
            for point in response[0]:
                source = point.payload.get('source', 'Unknown')
                if source not in documents:
                    documents[source] = {
                        'chunks': [],
                        'total_chunks': 0,
                        'pages': set(),
                        'word_count': 0
                    }
                
                text = point.payload.get('text', '')
                page = point.payload.get('page', '')
                
                documents[source]['chunks'].append({
                    'id': point.id,
                    'text': text,
                    'page': page,
                    'word_count': len(text.split())
                })
                documents[source]['total_chunks'] += 1
                documents[source]['pages'].add(page)
                documents[source]['word_count'] += len(text.split())
            
            # Convert page sets to sorted lists
            for doc in documents.values():
                doc['pages'] = sorted(list(doc['pages']))
            
            # Add collection stats
            stats = {
                'total_documents': len(documents),
                'total_chunks': sum(doc['total_chunks'] for doc in documents.values()),
                'total_words': sum(doc['word_count'] for doc in documents.values())
            }
            
            return {
                'documents': documents,
                'stats': stats
            }
        except Exception as e:
            print(f"Error getting vector store contents: {str(e)}")
            return {'documents': {}, 'stats': {'total_documents': 0, 'total_chunks': 0, 'total_words': 0}}
    
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
