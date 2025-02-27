from openai import OpenAI
import json
import numpy as np
from dotenv import load_dotenv
import os
import logging
from typing import List, Dict
import time
import html2text  # Add html2text for HTML processing

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EmbeddingGenerator:
    def __init__(self):
        key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)
        self.model = "text-embedding-3-small"
        self.chunk_size = 500
        self.chunk_overlap = 50
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        
    def load_articles(self, file_paths: List[str]) -> List[Dict]:
        """Load help articles from JSON files"""
        articles = []
        
        for file_path in file_paths:
            try:
                logger.info(f"Reading file: {file_path}")
                if not os.path.exists(file_path):
                    logger.error(f"File not found: {file_path}")
                    continue
                    
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Extract articles array from the JSON structure
                    if 'articles' in data:
                        file_articles = data['articles']
                        logger.info(f"Found {len(file_articles)} articles in {file_path}")
                        
                        for idx, article in enumerate(file_articles):
                            title = article.get('title', 'Untitled')
                            # Convert HTML body to plain text
                            body = article.get('body', '')
                            content = self.html_converter.handle(body).strip()
                            
                            logger.debug(f"Article {idx} title: {title}")
                            logger.debug(f"Article {idx} content length: {len(content)}")
                            
                            articles.append({
                                'title': title,
                                'content': content
                            })
                            
                        logger.info(f"Successfully processed {len(file_articles)} articles from {file_path}")
                    else:
                        logger.warning(f"No 'articles' array found in {file_path}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {file_path}: {e}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                
        logger.info(f"Total articles loaded and processed: {len(articles)}")
        return articles
        
    def chunk_text(self, text: str, title: str) -> List[Dict]:
        """Split text into overlapping chunks"""
        if not text:
            logger.warning(f"Empty content for article: {title}")
            return []
            
        logger.info(f"Chunking article '{title}' with content length: {len(text)}")
        words = text.split()
        logger.info(f"Article contains {len(words)} words")
        
        if not words:
            logger.warning(f"No words found in article: {title}")
            return []
            
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = " ".join(chunk_words)
            chunks.append({
                "text": chunk,
                "title": title,
                "start_idx": i
            })
            logger.debug(f"Created chunk {len(chunks)} with {len(chunk_words)} words")
            
        logger.info(f"Created {len(chunks)} chunks for article: {title}")
        return chunks
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a piece of text"""
        try:
            logger.debug(f"Generating embedding for text: {text[:100]}...")
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            if not response.data:
                logger.error("No embedding data returned from API")
                return None
            logger.debug("Successfully generated embedding")
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
            
    def process_articles(self, articles: List[Dict]) -> Dict:
        """Process articles and generate embeddings"""
        all_chunks = []
        all_embeddings = []
        
        logger.info(f"Processing {len(articles)} articles")
        for idx, article in enumerate(articles):
            title = article.get("title", "Untitled")
            content = article.get("content", "")
            
            logger.info(f"Processing article {idx + 1}/{len(articles)}: {title}")
            logger.info(f"Content length: {len(content)}")
            
            if not content:
                logger.warning(f"Empty content for article: {title}")
                continue
                
            # Split into chunks
            chunks = self.chunk_text(content, title)
            if not chunks:
                logger.warning(f"No chunks created for article: {title}")
                continue
                
            # Generate embeddings for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                logger.info(f"Generating embedding for chunk {chunk_idx + 1}/{len(chunks)} from article: {title}")
                logger.debug(f"Chunk text preview: {chunk['text'][:100]}...")
                
                embedding = self.generate_embedding(chunk["text"])
                if embedding:
                    all_chunks.append(chunk)
                    all_embeddings.append(embedding)
                    logger.debug(f"Successfully generated embedding for chunk {chunk_idx + 1}")
                else:
                    logger.warning(f"Failed to generate embedding for chunk {chunk_idx + 1} from article: {title}")
                
                # Rate limiting
                if chunk_idx < len(chunks) - 1:  # Don't sleep after the last chunk
                    logger.debug("Sleeping for rate limit...")
                    time.sleep(0.1)
                
        logger.info(f"Generated {len(all_embeddings)} embeddings from {len(articles)} articles")
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
    try:
        logger.info("Starting embedding generation process")
        generator = EmbeddingGenerator()
        
        # Load articles
        logger.info("Loading articles from JSON files...")
        articles = generator.load_articles(['sample-help.json', 'sample-help-2.json'])
        logger.info(f"Loaded {len(articles)} articles")
        
        # Validate articles
        if not articles:
            logger.error("No articles loaded. Check if JSON files exist and are not empty")
            return
            
        # Process articles and generate embeddings
        logger.info("Processing articles and generating embeddings...")
        data = generator.process_articles(articles)
        
        # Validate embeddings
        if not data['embeddings']:
            logger.error("No embeddings generated. Check OpenAI API key and rate limits")
            return
            
        logger.info(f"Generated {len(data['embeddings'])} embeddings")
        
        # Save embeddings
        logger.info("Saving embeddings to file...")
        generator.save_embeddings(data, "help_embeddings.npz")
        logger.info("Embedding generation process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)

if __name__ == "__main__":
    main() 