from typing import List, Dict, Optional
import json
from pathlib import Path
import html2text
from openai import OpenAI
import numpy as np
import logging
import re
import os

# Set up module-level logger
logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    Retrieval-Augmented Generation (RAG) retriever for finding relevant content.
    
    This class implements semantic search functionality for retrieving the most relevant
    text chunks from a corpus of documents based on similarity to a query. It uses
    OpenAI embeddings to create vector representations of both the corpus and queries.
    
    The retriever can load pre-computed embeddings from a file or process new articles
    and compute embeddings on the fly. It's designed to work with a corpus of articles
    that are chunked into smaller pieces for more precise retrieval.
    
    Attributes:
        client (OpenAI): OpenAI client instance for generating embeddings
        model (str): OpenAI embedding model name to use
        chunks (List[Dict]): List of text chunks with metadata
        embeddings (np.ndarray): Matrix of embedding vectors corresponding to chunks
    """
    
    def __init__(self, openai_client: OpenAI):
        """
        Initialize the RAG retriever with an OpenAI client.
        
        Args:
            openai_client (OpenAI): An initialized OpenAI client instance with valid API key
        """
        self.client = openai_client
        self.model = "text-embedding-3-small"  # Default embedding model
        self.chunks = []  # Will store text chunks with metadata
        self.embeddings = None  # Will store embedding vectors as numpy array
        
    def load_articles(self, file_paths: List[str], embeddings_file: Optional[str] = None):
        """
        Load articles and their embeddings from files.
        
        This method attempts to load pre-computed embeddings from the specified file if provided.
        If embeddings can't be loaded, it falls back to processing the articles from scratch.
        
        Args:
            file_paths (List[str]): List of paths to JSON files containing articles
            embeddings_file (Optional[str]): Path to a numpy file containing pre-computed embeddings
            
        Returns:
            None
            
        Note:
            Pre-computed embeddings significantly improve initialization speed.
        """
        if embeddings_file and os.path.exists(embeddings_file):
            # Try to load pre-computed embeddings first (faster)
            self._load_precomputed_embeddings(embeddings_file)
            logger.info("Loaded pre-computed embeddings")
            return
            
        # Fall back to processing articles and computing embeddings if no pre-computed file
        articles = []
        for file_path in file_paths:
            try:
                # Load each JSON file and extract articles
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Handle both list and single object formats
                    articles.extend(data if isinstance(data, list) else [data])
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        # Process loaded articles to generate chunks and embeddings
        self.process_articles(articles)
        
    def _load_precomputed_embeddings(self, file_path: str):
        """
        Load pre-computed embeddings from a numpy file.
        
        This internal method loads both the embedding vectors and their corresponding
        text chunks from a previously saved numpy file, performing validations to ensure
        the loaded data has the expected format and dimensions.
        
        Args:
            file_path (str): Path to the numpy file containing embeddings and chunks
            
        Raises:
            FileNotFoundError: If the embeddings file doesn't exist
            ValueError: If the file format is invalid or embeddings have wrong dimensions
        """
        try:
            logger.info(f"Loading pre-computed embeddings from {file_path}")
            if not os.path.exists(file_path):
                logger.error(f"Embeddings file not found: {file_path}")
                raise FileNotFoundError(f"Embeddings file not found: {file_path}")
                
            # Load the numpy file containing both embeddings and chunks
            data = np.load(file_path, allow_pickle=True)
            
            # Validate the loaded data structure
            if 'embeddings' not in data or 'chunks' not in data:
                logger.error("Invalid embeddings file format - missing required arrays")
                raise ValueError("Invalid embeddings file format - missing required arrays")
                
            # Extract embeddings matrix and chunks list
            embeddings = data['embeddings']
            chunks = json.loads(str(data['chunks']))
            
            # Verify that data is not empty
            if len(embeddings) == 0 or len(chunks) == 0:
                logger.error("Empty embeddings or chunks in file")
                raise ValueError("Empty embeddings or chunks in file")
                
            # Check embedding dimensions (model-specific)
            if embeddings.shape[1] != 1536:  # Expected embedding dimension for text-embedding-3-small
                logger.error(f"Invalid embedding dimension: {embeddings.shape[1]}, expected 1536")
                raise ValueError(f"Invalid embedding dimension: {embeddings.shape[1]}, expected 1536")
                
            # Store loaded data in instance variables
            self.embeddings = embeddings
            self.chunks = chunks
            logger.info(f"Successfully loaded {len(chunks)} chunks with embeddings of shape {embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Error loading pre-computed embeddings: {e}")
            # Don't set empty values, let the error propagate
            raise
            
    def process_articles(self, articles: List[Dict]):
        """
        Process articles to generate text chunks and their embeddings.
        
        This method takes a list of article dictionaries, splits each article into
        manageable chunks, and generates embeddings for each chunk using the OpenAI API.
        
        Args:
            articles (List[Dict]): List of article dictionaries, each containing 
                                  at least 'title' and 'content' fields
                                  
        Returns:
            None
        """
        self.chunks = []
        embeddings_list = []
        
        for article in articles:
            title = article.get("title", "Untitled")
            content = article.get("content", "")
            
            # Split the article content into smaller, overlapping chunks
            chunks = self._chunk_text(content, title)
            
            # Generate embeddings for each chunk
            for chunk in chunks:
                embedding = self._generate_embedding(chunk["text"])
                if embedding:
                    self.chunks.append(chunk)
                    embeddings_list.append(embedding)
                    
        # Convert list of embeddings to numpy array for efficient computation
        self.embeddings = np.array(embeddings_list)
        
    def _chunk_text(self, text: str, title: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """
        Split text into overlapping chunks to maintain context across chunks.
        
        This internal method divides a large text into smaller, manageable pieces with
        some overlap between consecutive chunks to preserve context around chunk boundaries.
        
        Args:
            text (str): The full text content to be split into chunks
            title (str): Title of the article (included with each chunk for reference)
            chunk_size (int): Maximum number of words per chunk
            overlap (int): Number of words to overlap between consecutive chunks
            
        Returns:
            List[Dict]: List of chunk dictionaries, each containing:
                        - text: The chunk text content
                        - title: The original article title
                        - start_idx: Word index where this chunk begins in the original text
        """
        words = text.split()
        chunks = []
        
        # Create chunks with specified size and overlap
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append({
                "text": chunk,
                "title": title,
                "start_idx": i
            })
            
        return chunks
        
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding vector for a piece of text using OpenAI's API.
        
        This internal method sends the text to OpenAI's embedding API and
        returns the resulting vector representation.
        
        Args:
            text (str): The text to generate an embedding for
            
        Returns:
            Optional[List[float]]: The embedding vector as a list of floats,
                                  or None if embedding generation failed
        """
        try:
            # Call OpenAI API to generate embedding
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
            
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Find the most relevant chunks for a given query.
        
        This method computes similarity between the query and all chunks,
        then returns the most similar chunks with their metadata.
        
        Args:
            query (str): The search query text
            top_k (int): Maximum number of results to return
            
        Returns:
            List[Dict]: List of the most relevant chunks, each containing:
                       - text: The chunk content
                       - title: The original article title
                       - similarity: Float score indicating relevance to query
                       
        Note:
            The method ensures that only one chunk is returned per article title
            to maximize diversity of results.
        """
        # Generate embedding for query
        query_embedding = self._generate_embedding(query)
        if not query_embedding:
            return []
            
        # Calculate cosine similarities between query and all chunks
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get indices of chunks with highest similarity scores
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        seen_titles = set()  # Track article titles to avoid duplicates
        
        for idx in top_indices:
            chunk = self.chunks[idx]
            title = chunk["title"]
            
            # Skip if we already have a chunk from this article
            if title in seen_titles:
                continue
                
            seen_titles.add(title)
            results.append({
                "text": chunk["text"],
                "title": title,
                "similarity": float(similarities[idx])
            })
            
        return results 