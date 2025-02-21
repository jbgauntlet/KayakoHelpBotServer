from typing import List, Dict
import json
from pathlib import Path
import html2text
from openai import OpenAI
import numpy as np
import logging

logger = logging.getLogger(__name__)

class RAGRetriever:
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client
        self.articles = []
        self.embeddings = []
        
    def load_articles(self, json_files: List[str]):
        """Load and process articles from JSON files"""
        for file in json_files:
            with open(file) as f:
                data = json.load(f)
                self.articles.extend(data['articles'])
                
    def process_articles(self):
        """Convert HTML to text and create embeddings"""
        h = html2text.HTML2Text()
        h.ignore_links = True
        
        for article in self.articles:
            # Convert HTML to plain text
            text = h.handle(article['body'])
            # Create embedding
            embedding = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            self.embeddings.append({
                'id': article['id'],
                'embedding': embedding.data[0].embedding,
                'title': article['title'],
                'text': text
            })
            
    def retrieve(self, query: str, top_k: int = 2) -> List[Dict]:
        """Retrieve most relevant articles for a query"""
        # Get query embedding
        query_embedding = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding
        
        # Calculate similarities
        similarities = []
        for doc in self.embeddings:
            similarity = np.dot(query_embedding, doc['embedding'])
            similarities.append((similarity, doc))
            logger.debug(f"Article: {doc['title']}, Similarity: {similarity}")
            
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Return top k results
        return [doc for _, doc in similarities[:top_k]] 