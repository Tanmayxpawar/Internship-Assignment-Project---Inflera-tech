import os
from typing import List, Dict, Any
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import json
import logging
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the document processor with a sentence transformer model.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        logger.info("Initializing DocumentProcessor...")
        self.model = SentenceTransformer(model_name)
        
        # Ensure the database directory exists
        db_dir = "chroma_db"
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Using database directory: {db_dir}")
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=db_dir)
        
        # Create embedding function using ChromaDB's utility
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        
        # Create or get the collection with proper configuration
        self.collection = self.client.get_or_create_collection(
            name="document_embeddings",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )
        logger.info("DocumentProcessor initialized successfully")

    def extract_text_from_pdf(self, pdf_path: str) -> List[str]:
        """
        Extract text from a PDF file and split it into chunks.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[str]: List of text chunks
        """
        reader = PdfReader(pdf_path)
        chunks = []
        
        for page in reader.pages:
            text = page.extract_text()
            # Split text into smaller chunks (you can adjust the chunk size)
            sentences = text.split('. ')
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) > 500:  # Maximum chunk size
                    chunks.append('. '.join(current_chunk) + '.')
                    current_chunk = [sentence]
                    current_length = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence)
            
            if current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
        
        return chunks

    def process_document(self, pdf_path: str, metadata: Dict[str, Any] = None) -> None:
        """
        Process a single PDF document and store its embeddings in ChromaDB.
        
        Args:
            pdf_path (str): Path to the PDF file
            metadata (Dict[str, Any], optional): Additional metadata for the document
        """
        logger.info(f"Processing document: {pdf_path}")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract text chunks
        chunks = self.extract_text_from_pdf(pdf_path)
        logger.info(f"Extracted {len(chunks)} chunks from document")
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Create a list of metadata dictionaries, one for each chunk
        metadatas = []
        for i in range(len(chunks)):
            chunk_metadata = {
                "source": pdf_path,
                "chunk_index": str(i),
                **metadata
            }
            metadatas.append(chunk_metadata)
        
        # Generate IDs for each chunk
        ids = [f"{os.path.basename(pdf_path)}_{i}" for i in range(len(chunks))]
        
        # Store in ChromaDB
        try:
            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully added document to ChromaDB: {pdf_path}")
        except Exception as e:
            logger.error(f"Error adding document to ChromaDB: {str(e)}")
            raise

    def process_directory(self, directory_path: str, metadata: Dict[str, Any] = None) -> None:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path (str): Path to the directory containing PDF files
            metadata (Dict[str, Any], optional): Additional metadata for the documents
        """
        logger.info(f"Processing directory: {directory_path}")
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            pdf_path = os.path.join(directory_path, pdf_file)
            try:
                self.process_document(pdf_path, metadata)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")

    def query_documents(self, query: str, n_results: int = 3) -> str:
        """
        Query the document database for similar content.
        
        Args:
            query (str): The query text
            n_results (int): Number of results to return
            
        Returns:
            str: JSON string containing the results
        """
        logger.info(f"Querying documents with: {query}")
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results for better readability
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
            
            logger.info(f"Found {len(formatted_results)} results")
            return json.dumps(formatted_results, indent=2)
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            return "[]" 