from openai import OpenAI
import json
import numpy as np
from dotenv import load_dotenv
import os
import logging
from typing import List, Dict
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EmbeddingGenerator:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "text-embedding-3-small"
        self.chunk_size = 500
        self.chunk_overlap = 50
        
    def load_articles(self, file_paths: List[str]) -> List[Dict]:
        """Load help articles from JSON files"""
        articles = []
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    articles.extend(data if isinstance(data, list) else [data])
                logger.info(f"Loaded {len(articles)} articles from {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        return articles
        
    def chunk_text(self, text: str, title: str) -> List[Dict]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append({
                "text": chunk,
                "title": title,
                "start_idx": i
            })
            
        return chunks
        
    async def generate_embedding(self, text: str) -> List[float]:
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
            
    def process_articles(self, articles: List[Dict]) -> Dict:
        """Process articles and generate embeddings"""
        all_chunks = []
        all_embeddings = []
        
        for article in articles:
            title = article.get("title", "Untitled")
            content = article.get("content", "")
            
            # Split into chunks
            chunks = self.chunk_text(content, title)
            
            # Generate embeddings for each chunk
            for chunk in chunks:
                logger.info(f"Generating embedding for chunk from article: {title}")
                embedding = self.generate_embedding(chunk["text"])
                if embedding:
                    all_chunks.append(chunk)
                    all_embeddings.append(embedding)
                time.sleep(0.1)  # Rate limiting
                
        return {
            "chunks": all_chunks,
            "embeddings": all_embeddings
        }
        
    def save_embeddings(self, data: Dict, output_file: str):
        """Save embeddings and chunks to file"""
        try:
            # Convert embeddings to numpy array for efficient storage
            embeddings_array = np.array(data["embeddings"])
            
            # Save to npz file
            np.savez(
                output_file,
                embeddings=embeddings_array,
                chunks=json.dumps(data["chunks"])
            )
            logger.info(f"Saved embeddings to {output_file}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")

def main():
    generator = EmbeddingGenerator()
    
    # Load articles
    articles = generator.load_articles(['sample-help.json', 'sample-help-2.json'])
    logger.info(f"Loaded {len(articles)} articles")
    
    # Process articles and generate embeddings
    data = generator.process_articles(articles)
    logger.info(f"Generated {len(data['embeddings'])} embeddings")
    
    # Save embeddings
    generator.save_embeddings(data, "help_embeddings.npz")
    
if __name__ == "__main__":
    main() 