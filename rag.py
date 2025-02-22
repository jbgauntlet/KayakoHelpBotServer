from typing import List, Dict
import json
from pathlib import Path
import html2text
from openai import OpenAI
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)

class RAGRetriever:
    def __init__(self, openai_client: OpenAI, chunk_size: int = 500, chunk_overlap: int = 50):
        # Initialize with configurable chunk parameters
        self.openai_client = openai_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.articles = []
        self.chunks = []  # Store chunks instead of full embeddings
        
    def load_articles(self, json_files: List[str]):
        """Load and process articles from JSON files"""
        for file in json_files:
            with open(file) as f:
                data = json.load(f)
                self.articles.extend(data['articles'])
                
    def _clean_text(self, text: str) -> str:
        # Clean text by removing special characters and extra whitespace
        text = text.replace("#", "").replace("*", "")
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = text.strip()
        return text
    
    def _split_into_chunks(self, text: str, title: str) -> List[Dict]:
        # Split text into overlapping chunks while preserving sentence boundaries
        chunks = []
        
        # Clean the text first
        text = self._clean_text(text)
        
        # Split into sentences (basic implementation - can be improved with nltk)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If single sentence is longer than chunk_size, split it
            if sentence_len > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        "title": title,
                        "text": current_chunk.strip(),
                        "embedding": None
                    })
                
                # Split long sentence into smaller pieces
                words = sentence.split()
                current_chunk = ""
                current_size = 0
                
                for word in words:
                    if current_size + len(word) + 1 > self.chunk_size:
                        chunks.append({
                            "title": title,
                            "text": current_chunk.strip(),
                            "embedding": None
                        })
                        current_chunk = word + " "
                        current_size = len(word) + 1
                    else:
                        current_chunk += word + " "
                        current_size += len(word) + 1
                        
            # Normal case: add sentence to current chunk or start new chunk
            elif current_size + sentence_len + 1 > self.chunk_size:
                chunks.append({
                    "title": title,
                    "text": current_chunk.strip(),
                    "embedding": None
                })
                current_chunk = sentence + " "
                current_size = sentence_len + 1
            else:
                current_chunk += sentence + " "
                current_size += sentence_len + 1
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append({
                "title": title,
                "text": current_chunk.strip(),
                "embedding": None
            })
            
        return chunks
        
    def process_articles(self):
        """Convert HTML to text and create embeddings for chunks"""
        h = html2text.HTML2Text()
        h.ignore_links = True
        
        # Process each article into chunks
        for article in self.articles:
            # Convert HTML to plain text
            text = h.handle(article['body'])
            
            # Split article into chunks
            article_chunks = self._split_into_chunks(text, article['title'])
            
            # Create embeddings for each chunk
            for chunk in article_chunks:
                try:
                    embedding = self.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=chunk["text"]
                    )
                    chunk["embedding"] = embedding.data[0].embedding
                    self.chunks.append(chunk)
                    logger.debug(f"Created embedding for chunk of size {len(chunk['text'])} from article: {chunk['title']}")
                except Exception as e:
                    logger.error(f"Error creating embedding for chunk: {e}")
            
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        # Retrieve most relevant chunks for a query
        query_embedding = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding
        
        # Calculate similarities with all chunks
        similarities = []
        for chunk in self.chunks:
            similarity = np.dot(query_embedding, chunk['embedding'])
            similarities.append((similarity, chunk))
            logger.debug(f"Chunk similarity: {similarity}, Title: {chunk['title']}, Preview: {chunk['text'][:100]}...")
            
        # Sort by similarity and get top results
        similarities.sort(reverse=True)
        
        # Deduplicate chunks from the same title while maintaining order
        seen_titles = set()
        unique_results = []
        
        for similarity, chunk in similarities:
            if len(unique_results) >= top_k:
                break
                
            if chunk['title'] not in seen_titles:
                seen_titles.add(chunk['title'])
                unique_results.append({
                    'title': chunk['title'],
                    'text': chunk['text'],
                    'similarity': similarity
                })
        
        return unique_results 