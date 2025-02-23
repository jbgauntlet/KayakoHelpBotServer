from typing import List, Dict, Optional
import json
from pathlib import Path
import html2text
from openai import OpenAI
import numpy as np
import logging
import re
import os

logger = logging.getLogger(__name__)

class RAGRetriever:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.model = "text-embedding-3-small"
        self.chunks = []
        self.embeddings = None
        
    def load_articles(self, file_paths: List[str], embeddings_file: Optional[str] = None):
        """Load articles and their embeddings"""
        if embeddings_file and os.path.exists(embeddings_file):
            self._load_precomputed_embeddings(embeddings_file)
            logger.info("Loaded pre-computed embeddings")
            return
            
        # Fall back to processing articles and computing embeddings if no pre-computed file
        articles = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    articles.extend(data if isinstance(data, list) else [data])
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        self.process_articles(articles)
        
    def _load_precomputed_embeddings(self, file_path: str):
        """Load pre-computed embeddings from file"""
        try:
            logger.info(f"Loading pre-computed embeddings from {file_path}")
            if not os.path.exists(file_path):
                logger.error(f"Embeddings file not found: {file_path}")
                raise FileNotFoundError(f"Embeddings file not found: {file_path}")
                
            data = np.load(file_path, allow_pickle=True)
            
            # Validate the loaded data
            if 'embeddings' not in data or 'chunks' not in data:
                logger.error("Invalid embeddings file format - missing required arrays")
                raise ValueError("Invalid embeddings file format - missing required arrays")
                
            embeddings = data['embeddings']
            chunks = json.loads(str(data['chunks']))
            
            if len(embeddings) == 0 or len(chunks) == 0:
                logger.error("Empty embeddings or chunks in file")
                raise ValueError("Empty embeddings or chunks in file")
                
            if embeddings.shape[1] != 1536:  # Expected embedding dimension for text-embedding-3-small
                logger.error(f"Invalid embedding dimension: {embeddings.shape[1]}, expected 1536")
                raise ValueError(f"Invalid embedding dimension: {embeddings.shape[1]}, expected 1536")
                
            self.embeddings = embeddings
            self.chunks = chunks
            logger.info(f"Successfully loaded {len(chunks)} chunks with embeddings of shape {embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Error loading pre-computed embeddings: {e}")
            # Don't set empty values, let the error propagate
            raise
            
    def process_articles(self, articles: List[Dict]):
        """Process articles and generate embeddings (only used if no pre-computed embeddings)"""
        self.chunks = []
        embeddings_list = []
        
        for article in articles:
            title = article.get("title", "Untitled")
            content = article.get("content", "")
            
            # Split into chunks
            chunks = self._chunk_text(content, title)
            
            # Generate embeddings for each chunk
            for chunk in chunks:
                embedding = self._generate_embedding(chunk["text"])
                if embedding:
                    self.chunks.append(chunk)
                    embeddings_list.append(embedding)
                    
        self.embeddings = np.array(embeddings_list)
        
    def _chunk_text(self, text: str, title: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append({
                "text": chunk,
                "title": title,
                "start_idx": i
            })
            
        return chunks
        
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a piece of text"""
        try:
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
        """Find most relevant chunks for a query"""
        # Generate embedding for query
        query_embedding = self._generate_embedding(query)
        if not query_embedding:
            return []
            
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        seen_titles = set()
        
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