# FastAPI server for Kayako Help Center voice bot
# This is the main server file that handles voice calls and provides AI-powered responses

# Import necessary libraries
# FastAPI - for creating the web server
# OpenAI - for AI capabilities (chat, speech-to-text, embeddings)
# Twilio - for handling phone calls
# Other utilities for various functionalities
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import Response
from openai import OpenAI, RateLimitError
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from dotenv import load_dotenv
import json
import os
import logging
import time
import asyncio
from typing import Dict, Literal, Optional, List, AsyncGenerator
import base64
from collections import OrderedDict
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import websockets
import re

# Import our custom modules
from prompts import SYSTEM_PROMPT, INTENT_CLASSIFICATION_PROMPT
from rag import RAGRetriever

# Set up logging to help track what's happening in the application
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Create the FastAPI application
app = FastAPI()

# Get API keys from environment variables for security
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")

# Initialize our external service clients
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)  # For making phone calls
openai_client = OpenAI(api_key=OPENAI_API_KEY)         # For AI capabilities

# Set up our knowledge base system (RAG) with help articles
rag = RAGRetriever(openai_client)
rag.load_articles(['sample-help.json', 'sample-help-2.json'], embeddings_file='help_embeddings.npz')

# Dictionary to store conversation information for each active call
conversation_context: Dict[str, Dict] = {}

# Constants to control conversation flow and prevent infinite loops
MAX_INVALID_ATTEMPTS = 3    # How many times we'll allow unclear or off-topic responses
MAX_SILENCE_ATTEMPTS = 3    # How many times we'll allow silence before ending call
MAX_TOTAL_TURNS = 50       # Maximum back-and-forth exchanges in one call

# Define what types of user intents we can recognize
IntentType = Literal["END_CALL", "CONTINUE", "CONNECT_HUMAN", "UNCLEAR", "OFF_TOPIC"]

@dataclass
class AudioBuffer:
    """Represents a buffer of audio data with metadata"""
    chunks: List[bytes]
    last_activity: float
    transcript: str = ""
    is_final: bool = False
    state: str = "collecting"  # Add state tracking
    MAX_BUFFER_SIZE = 1024 * 1024  # 1MB maximum buffer size
    BACKPRESSURE_THRESHOLD = 768 * 1024  # 75% of max size
    current_size: int = 0
    needs_backpressure: bool = False

    def add_chunk(self, chunk: bytes) -> bool:
        """Add chunk to buffer if within size limits. Returns False if buffer is full."""
        chunk_size = len(chunk)
        
        # Check if adding this chunk would exceed max size
        if self.current_size + chunk_size > self.MAX_BUFFER_SIZE:
            self.needs_backpressure = True
            return False
            
        # Set backpressure flag if we're approaching the threshold
        self.needs_backpressure = (self.current_size + chunk_size) > self.BACKPRESSURE_THRESHOLD
        
        self.chunks.append(chunk)
        self.current_size += chunk_size
        return True

    def clear(self):
        """Clear the buffer"""
        self.chunks = []
        self.current_size = 0
        self.transcript = ""
        self.is_final = False
        self.state = "collecting"
        self.needs_backpressure = False

class LRUCache:
    """A cache that keeps the most recently used items and removes the least recently used ones"""
    
    def __init__(self, capacity: int = 1000):
        # Initialize the cache with a maximum size
        self.cache = OrderedDict()  # Special dictionary that remembers order of items
        self.capacity = capacity    # Maximum number of items we can store
        self.hits = 0              # Count of successful cache lookups
        self.misses = 0            # Count of unsuccessful cache lookups
        self.logger = logging.getLogger(__name__)
        
    def get(self, key: str) -> Optional[str]:
        # Try to get an item from the cache
        if key in self.cache:
            self.hits += 1  # We found it! Count this as a success
            self.cache.move_to_end(key)  # Move this item to "most recently used"
            self.logger.debug(f"Cache hit for key: {key[:50]}...")
            return self.cache[key]
        self.misses += 1  # We didn't find it. Count this as a miss
        self.logger.debug(f"Cache miss for key: {key[:50]}...")
        return None
        
    def put(self, key: str, value: str):
        # Add or update an item in the cache
        if key in self.cache:
            self.cache.move_to_end(key)  # If it exists, move it to "most recently used"
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # If we're over capacity, remove the least recently used item
            self.cache.popitem(last=False)
            
    def get_stats(self) -> dict:
        # Get statistics about how well the cache is performing
        total = self.hits + self.misses
        hit_rate = (self.hits / total) * 100 if total > 0 else 0
        return {
            "size": len(self.cache),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%"
        }

class ResponseCache:
    """A system to manage multiple caches for different types of responses"""
    
    def __init__(self):
        # Create two separate caches:
        self.knowledge_cache = LRUCache(capacity=1000)  # For storing retrieved knowledge
        self.response_cache = LRUCache(capacity=1000)   # For storing formatted responses
        self.logger = logging.getLogger(__name__)
        
    def _generate_knowledge_key(self, query: str) -> str:
        # Create a unique key for storing knowledge results
        # We normalize the query (lowercase, no extra spaces) to increase cache hits
        return query.lower().strip()
        
    def _generate_response_key(self, query: str, knowledge: str) -> str:
        # Create a unique key for storing formatted responses
        # We combine the query and a hash of the knowledge to ensure uniqueness
        return f"{query.lower().strip()}:{hash(knowledge)}"
        
    async def get_knowledge(self, query: str) -> Optional[str]:
        # Try to get cached knowledge for a query, or retrieve new knowledge if not found
        key = self._generate_knowledge_key(query)
        if cached := self.knowledge_cache.get(key):
            return cached
        
        # If not in cache, get new knowledge and store it
        knowledge = retrieve_data(query)
        if knowledge:
            self.knowledge_cache.put(key, knowledge)
        return knowledge
        
    async def get_response(self, query: str, knowledge: str) -> Optional[str]:
        # Try to get a cached response, or generate new one if not found
        key = self._generate_response_key(query, knowledge)
        if cached := self.response_cache.get(key):
            return cached
            
        # If not in cache, generate new response and store it
        response = await format_response(knowledge, query)
        if response:
            self.response_cache.put(key, response)
        return response
        
    def get_stats(self) -> dict:
        # Get performance statistics from both caches
        knowledge_stats = self.knowledge_cache.get_stats()
        response_stats = self.response_cache.get_stats()
        return {
            "knowledge_cache": knowledge_stats,
            "response_cache": response_stats
        }
        
    async def prewarm(self, common_queries: List[str]):
        # Fill the cache with responses to common questions before they're needed
        self.logger.info("Starting cache prewarming...")
        start_time = time.time()
        
        for query in common_queries:
            try:
                # First get and cache the knowledge
                knowledge = await self.get_knowledge(query)
                if knowledge:
                    # Then get and cache the response
                    await self.get_response(query, knowledge)
                    self.logger.debug(f"Prewarmed cache for query: {query[:50]}...")
            except Exception as e:
                self.logger.error(f"Error prewarming cache for query '{query}': {e}")
                
        duration = time.time() - start_time
        self.logger.info(f"Cache prewarming completed in {duration:.2f} seconds")
        
    def clear(self):
        # Reset both caches to empty state
        self.knowledge_cache = LRUCache(capacity=self.knowledge_cache.capacity)
        self.response_cache = LRUCache(capacity=self.response_cache.capacity)
        self.logger.info("Caches cleared")

# List of common questions we'll prepare answers for
COMMON_QUERIES = [
    "How do I set up Single Sign-On (SSO) for admin users?",
    "How do I manage administrator roles and permissions?",
    "How do I customize the branding of my AdvocateHub?",
    "How do I add or upgrade users to administrator accounts?",
    "How do I customize the sign-in page?",
    "How do I change the name of my AdvocateHub?",
    "How do I edit administrator profiles?",
    "How do I contact the admin team?",
    "How do I switch between administrator and advocate accounts?",
    "How do I customize the CSS and styling of my hub?"
]

# Create our response cache system
response_cache = ResponseCache()

# Add startup event to prewarm cache
@app.on_event("startup")
async def startup_event():
    """Initialize and prewarm cache on startup"""
    try:
        await response_cache.prewarm(COMMON_QUERIES)
    except Exception as e:
        logger.error(f"Error during cache prewarming: {e}")

# Add endpoint to manually trigger cache prewarming
@app.post("/admin/cache/prewarm")
async def prewarm_cache():
    """Manually trigger cache prewarming"""
    try:
        # Clear existing cache first
        response_cache.clear()
        # Prewarm with common queries
        await response_cache.prewarm(COMMON_QUERIES)
        return {"status": "success", "message": "Cache prewarming completed"}
    except Exception as e:
        logger.error(f"Error during manual cache prewarming: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class WebSocketRetry:
    """Handles WebSocket connection retries with exponential backoff"""
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    async def connect(self, url: str, headers: dict) -> websockets.WebSocketClientProtocol:
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    delay = self.base_delay * (2 ** (attempt - 1))
                    self.logger.info(f"Retrying WebSocket connection in {delay} seconds (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(delay)
                
                return await websockets.connect(url, extra_headers=headers)
                
            except websockets.exceptions.WebSocketException as e:
                last_exception = e
                self.logger.warning(f"WebSocket connection attempt {attempt + 1} failed: {e}")
                continue
                
        raise last_exception or RuntimeError("Failed to establish WebSocket connection")

class ResponseState:
    """Tracks state of response generation"""
    def __init__(self):
        self.sentences: List[str] = []
        self.current_sentence: List[str] = []
        self.position: int = 0
        self.is_complete: bool = False
        self.last_token: Optional[str] = None
        
    def add_content(self, content: str):
        """Add content and track sentence completion"""
        self.current_sentence.append(content)
        self.last_token = content
        
        # Check if we completed a sentence
        text = ''.join(self.current_sentence)
        if any(text.rstrip().endswith(char) for char in '.!?'):
            self.sentences.append(text)
            self.current_sentence = []
            self.position = len(self.sentences)
            
    def get_current_state(self) -> dict:
        """Get current state for recovery"""
        return {
            "sentences": self.sentences.copy(),
            "current": ''.join(self.current_sentence),
            "position": self.position,
            "is_complete": self.is_complete
        }
        
    @classmethod
    def from_state(cls, state: dict) -> 'ResponseState':
        """Recreate state from saved state"""
        response_state = cls()
        response_state.sentences = state["sentences"]
        response_state.current_sentence = list(state["current"])
        response_state.position = state["position"]
        response_state.is_complete = state["is_complete"]
        return response_state

class StreamingProcessor:
    """Handles real-time processing of audio from phone calls"""
    
    def __init__(self, 
                 silence_threshold: float = 0.4,
                 max_buffer_age: float = 2.0,
                 min_transcript_len: int = 10):
        self.buffers: Dict[str, AudioBuffer] = {}
        self.active_streams: Dict[str, Dict] = {}  # Track active response streams
        self.response_states: Dict[str, ResponseState] = {}  # Track response states
        self.silence_threshold = silence_threshold
        self.max_buffer_age = max_buffer_age
        self.min_transcript_len = min_transcript_len
        self.logger = logging.getLogger(__name__)
        self.openai_client = OpenAI()
        
    async def handle_interrupt(self, call_sid: str):
        """Handle user interruption of response"""
        if call_sid in self.active_streams:
            # Save current state before interrupting
            if call_sid in self.response_states:
                self.active_streams[call_sid]["last_state"] = \
                    self.response_states[call_sid].get_current_state()
            
            # Signal interruption
            self.active_streams[call_sid]["interrupt"].set()
            # Wait for cleanup
            await asyncio.sleep(0.1)
            # Remove from active streams
            self.active_streams.pop(call_sid, None)
            
    async def start_response_stream(self, call_sid: str, websocket: WebSocket, knowledge: str, query: str):
        """Start a new response stream with interruption handling"""
        # Create interruption event
        interrupt_event = asyncio.Event()
        
        # Store stream info
        self.active_streams[call_sid] = {
            "interrupt": interrupt_event,
            "start_time": time.time()
        }
        
        try:
            await format_and_stream_response(
                websocket=websocket,
                knowledge=knowledge,
                query=query,
                interrupt_event=interrupt_event,
                call_sid=call_sid,
                streaming_processor=self
            )
        finally:
            # Clean up stream tracking
            self.active_streams.pop(call_sid, None)
        
    async def process_chunk(self, call_sid: str, chunk: bytes, websocket: WebSocket) -> Optional[str]:
        try:
            # Get or create buffer for this call
            if call_sid not in self.buffers:
                self.buffers[call_sid] = AudioBuffer(
                    chunks=[],
                    last_activity=time.time(),
                    state="collecting"
                )
            buffer = self.buffers[call_sid]
            
            # Add new chunk and update activity
            if not buffer.add_chunk(chunk):
                # Buffer is full, signal backpressure
                await websocket.send_json({
                    "event": "backpressure",
                    "action": "pause",
                    "reason": "Buffer full"
                })
                return None
            
            # Signal backpressure if we're approaching threshold
            if buffer.needs_backpressure:
                await websocket.send_json({
                    "event": "backpressure",
                    "action": "slow",
                    "reason": "Buffer nearly full"
                })
            
            buffer.last_activity = time.time()
            
            # Check if we should process based on our rules
            if await self._should_process_buffer(buffer):
                # Process accumulated audio using OpenAI's WebSocket with retries
                async with asyncio.timeout(10.0):  # 10-second timeout for audio processing
                    async with WebSocketRetry(max_retries=3) as retry:
                        ws = await retry.connect(
                            'wss://api.openai.com/v1/audio/speech',
                            headers={
                                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                                "Content-Type": "application/json"
                            }
                        )
                        async with ws:
                            # Send audio data
                            await ws.send(json.dumps({
                                "model": "whisper-1",
                                "language": "en",
                                "audio_format": "mulaw",
                                "sample_rate": 8000
                            }))
                            
                            # Send audio chunks
                            for audio_chunk in buffer.chunks:
                                await ws.send(audio_chunk)
                            
                            # Get transcription
                            result = await ws.recv()
                            transcript_data = json.loads(result)
                            buffer.transcript = transcript_data.get("text", "")
                            
                            # Clear buffer and signal ready for more
                            transcript = await self._process_buffer(call_sid)
                            if transcript:
                                await websocket.send_json({
                                    "event": "backpressure",
                                    "action": "resume",
                                    "reason": "Buffer processed"
                                })
                            return transcript
            
            return None
            
        except asyncio.TimeoutError:
            self.logger.error("Audio processing timed out")
            return None
        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {e}")
            raise
            
    async def _should_process_buffer(self, buffer: AudioBuffer) -> bool:
        # Decide if we have enough audio to process
        current_time = time.time()
        
        # Calculate how long since last audio
        buffer_age = current_time - buffer.last_activity
        has_minimum_content = len(buffer.transcript.strip()) >= self.min_transcript_len
        
        # Process if we have either:
        # 1. A natural pause and enough content
        # 2. Been holding the audio too long but have some content
        return (buffer_age >= self.silence_threshold and has_minimum_content) or \
               (buffer_age >= self.max_buffer_age and has_minimum_content)
               
    async def _process_buffer(self, call_sid: str) -> Optional[str]:
        # Convert our collected audio into text
        try:
            buffer = self.buffers[call_sid]
            
            if not buffer.transcript:
                return None
                
            # Get the final text version
            transcript = buffer.transcript.strip()
            
            # Clear the buffer for next time
            self.buffers[call_sid] = AudioBuffer(
                chunks=[],
                last_activity=time.time()
            )
            
            self.logger.info(f"Processed transcript: {transcript}")
            return transcript
            
        except Exception as e:
            self.logger.error(f"Error processing buffer: {e}")
            return None
            
    async def cleanup_old_buffers(self):
        # Remove any audio buffers that haven't been used in a while
        current_time = time.time()
        to_remove = []
        
        for call_sid, buffer in self.buffers.items():
            if current_time - buffer.last_activity > self.max_buffer_age * 2:
                to_remove.append(call_sid)
                
        for call_sid in to_remove:
            del self.buffers[call_sid]
            
    def get_buffer_status(self, call_sid: str) -> dict:
        # Get information about a specific call's audio buffer
        if call_sid not in self.buffers:
            return {"status": "no_buffer"}
            
        buffer = self.buffers[call_sid]
        return {
            "chunks": len(buffer.chunks),
            "last_activity": datetime.fromtimestamp(buffer.last_activity).isoformat(),
            "transcript_length": len(buffer.transcript),
            "is_final": buffer.is_final
        }

# Create our audio processor
streaming_processor = StreamingProcessor()

def classify_intent(response: str, context: str) -> IntentType:
    # Figure out what the user wants to do based on their response
    try:
        # Ask OpenAI to classify the user's intent
        classification = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": INTENT_CLASSIFICATION_PROMPT},
                {"role": "user", "content": f"Context: {context}\nUser's response: {response}"}
            ],
            temperature=0,  # Use consistent responses
            max_tokens=10   # We only need a short answer
        )
        intent = classification.choices[0].message.content.strip()
        logger.debug(f"Classified intent: {intent} for response: {response}")
        
        # Make sure we got a valid intent type
        if intent in ["END_CALL", "CONTINUE", "CONNECT_HUMAN", "UNCLEAR", "OFF_TOPIC"]:
            return intent
        return "UNCLEAR"
    except Exception as e:
        logger.error(f"Error in intent classification: {e}")
        return "UNCLEAR"

@app.post("/voice")
async def handle_call():
    # This is the entry point for new phone calls
    # It answers the call and starts the conversation with Media Streams
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Start>
        <Stream url="{os.getenv('SERVER_URL')}/media" mode="full-duplex"/>
    </Start>
    <Pause length="120"/>
    <Redirect method="POST">{os.getenv('SERVER_URL')}/voice</Redirect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.websocket("/media")
async def media_stream(websocket: WebSocket):
    logger.info("New WebSocket connection attempt")
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    call_sid = None
    heartbeat_task = None
    stream_sid = None
    
    async def send_heartbeat():
        try:
            while True:
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                await websocket.send_json({"event": "heartbeat"})
                logger.debug("Heartbeat sent")
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket closed during heartbeat")
        except Exception as e:
            logger.error(f"Error in heartbeat: {e}")
    
    try:
        # Wait for the initial connected message from Twilio
        logger.info("Waiting for initial connected message")
        data = await websocket.receive_text()
        msg = json.loads(data)
        if msg.get("event") != "connected":
            logger.error(f"Expected 'connected' event, got: {msg.get('event')}")
            raise ValueError("Expected 'connected' event")
        logger.info("Received connected event")
        
        # Wait for the start message which contains call metadata
        logger.info("Waiting for start message")
        data = await websocket.receive_text()
        msg = json.loads(data)
        if msg.get("event") != "start":
            logger.error(f"Expected 'start' event, got: {msg.get('event')}")
            raise ValueError("Expected 'start' event")
            
        # Extract important metadata
        start_data = msg.get("start", {})
        call_sid = start_data.get("callSid")
        stream_sid = start_data.get("streamSid")
        if not call_sid or not stream_sid:
            logger.error(f"Missing metadata - callSid: {call_sid}, streamSid: {stream_sid}")
            raise ValueError("Missing callSid or streamSid")
        
        logger.info(f"Started streaming session for call {call_sid}")
        
        # Initialize conversation context
        conversation_context[call_sid] = {
            "last_query": None,
            "current_question": None,
            "invalid_attempts": 0,
            "silence_attempts": 0,
            "total_turns": 0
        }
        logger.info("Conversation context initialized")
        
        # Send initial greeting
        logger.info("Sending initial greeting")
        greeting = "Thank you for calling the Kayako Help Center today. How may I assist you?"
        greeting_payload = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {
                "payload": base64.b64encode(greeting.encode('utf-8')).decode('utf-8'),
                "contentType": "text/plain",
                "encoding": "base64"
            }
        }
        logger.info(f"Greeting payload prepared: {greeting_payload}")
        await websocket.send_json(greeting_payload)
        logger.info("Initial greeting sent")
        
        # Send mark to track when greeting is done
        await websocket.send_json({
            "event": "mark",
            "streamSid": stream_sid,
            "mark": {
                "name": "greeting_complete"
            }
        })
        logger.info("Greeting mark sent")
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(send_heartbeat())
        logger.info("Heartbeat task started")
        
        while True:
            try:
                logger.debug("Waiting for next message")
                data = await websocket.receive_text()
                msg = json.loads(data)
                logger.debug(f"Received message type: {msg.get('event')}")
                
                if msg["event"] == "media":
                    if not call_sid:
                        logger.error("Received media without proper initialization")
                        continue
                    
                    # Process audio chunk
                    chunk = base64.b64decode(msg["media"]["payload"])
                    logger.debug(f"Processing audio chunk of size: {len(chunk)}")
                    if transcript := await streaming_processor.process_chunk(call_sid, chunk, websocket):
                        logger.info(f"Processed transcript: {transcript}")
                        # Get conversation context
                        context = conversation_context.get(call_sid)
                        if not context:
                            logger.error(f"No context found for call {call_sid}")
                            continue
                        
                        # Process the transcript and generate response
                        response = await process_user_input(transcript, context, stream_sid, websocket)
                        
                        # Send mark to track when response is complete
                        await websocket.send_json({
                            "event": "mark",
                            "streamSid": stream_sid,
                            "mark": {
                                "name": f"response_{context['total_turns']}_complete"
                            }
                        })
                
                elif msg["event"] == "mark":
                    # Handle mark events (track when our audio finishes playing)
                    logger.info(f"Received mark: {msg.get('mark', {}).get('name')}")
                
                elif msg["event"] == "stop":
                    logger.info(f"Stopping stream for call {call_sid}")
                    break
                
            except asyncio.TimeoutError:
                # Send ping to check connection
                logger.warning("Message timeout, sending ping")
                try:
                    await websocket.send_json({"event": "ping"})
                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"WebSocket connection lost for call {call_sid}")
                    break
                    
    except Exception as e:
        logger.error(f"Error in media stream: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "event": "error",
                "message": str(e)
            })
        except websockets.exceptions.ConnectionClosed:
            logger.error("Could not send error message - connection closed")
    finally:
        # Clean up
        logger.info("Starting cleanup")
        if call_sid and call_sid in streaming_processor.active_streams:
            await streaming_processor.handle_interrupt(call_sid)
            logger.info("Handled stream interrupt")
        if call_sid:
            await streaming_processor.cleanup_old_buffers()
            conversation_context.pop(call_sid, None)
            logger.info("Cleaned up conversation context and buffers")
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
                logger.info("Heartbeat task cancelled")
            except asyncio.CancelledError:
                pass
        try:
            await websocket.close()
            logger.info("WebSocket connection closed")
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket was already closed")

async def process_user_input(transcript: str, context: dict, stream_sid: str, websocket: WebSocket) -> None:
    """Process user input and send appropriate response"""
    # Update conversation turns
    context["total_turns"] = context.get("total_turns", 0) + 1
    
    # Classify user intent
    intent = classify_intent(
        transcript,
        f"Current question: {context.get('current_question', 'How may I assist you?')}"
    )
    
    # Generate appropriate response
    response_text = await generate_response(transcript, intent, context)
    
    # Send response as media
    await websocket.send_json({
        "event": "media",
        "streamSid": stream_sid,
        "media": {
            "payload": base64.b64encode(response_text.encode()).decode()
        }
    })

async def generate_response(transcript: str, intent: str, context: dict) -> str:
    """Generate appropriate response based on intent and context"""
    if context.get("total_turns") > MAX_TOTAL_TURNS:
        return "I notice this has been a long conversation. To ensure quality service, let me connect you with a human agent who can provide more comprehensive assistance."
        
    if context.get("current_question") == "Do you need help with anything else?" and intent == "END_CALL":
        return "Thank you for calling Kayako Help Center. Have a great day!"
        
    if intent in ["OFF_TOPIC", "UNCLEAR"]:
        context["invalid_attempts"] = context.get("invalid_attempts", 0) + 1
        if context["invalid_attempts"] >= MAX_INVALID_ATTEMPTS:
            return "I apologize, but I'm having trouble understanding your needs. Let me connect you with a human agent who can better assist you."
        return "I can only assist with questions related to Kayako and its services. Please ask a Kayako-related question." if intent == "OFF_TOPIC" else "I'm not sure I understood. Could you please rephrase your question about Kayako?"
    
    # For valid queries, get knowledge and generate response
    context["invalid_attempts"] = 0
    context["last_query"] = transcript
    
    knowledge = await response_cache.get_knowledge(transcript)
    response = await format_response(knowledge, transcript)
    
    # Update current question
    context["current_question"] = "Do you need help with anything else?"
    if "connect you with a human" in knowledge:
        context["current_question"] = "Would you like me to connect you with a human support agent?"
        
    return response

def retrieve_data(query: str) -> str:
    # Get relevant information from our knowledge base
    #
    # Args:
    #     query: The user's question
    #        
    # Returns:
    #     Information from our help articles that answers their question
    
    # Get the 3 most relevant pieces of information
    results = rag.retrieve(query, top_k=3)
    
    # Format the information nicely
    context = []
    total_chars = 0
    max_chars = 2000  # Keep responses a reasonable length
    
    for result in results:
        # Add the article title and content
        content = result['text']
        
        # Make sure we don't make the response too long
        chunk_length = len(content) + len(result['title']) + 10
        if total_chars + chunk_length > max_chars:
            logger.debug(f"Skipping chunk due to length limit. Current total: {total_chars}, Chunk size: {chunk_length}")
            break
            
        context.append(f"Title: {result['title']}\n\n{content}")
        total_chars += chunk_length
        logger.debug(f"Retrieved chunk from article: {result['title']}, Similarity: {result['similarity']:.3f}")
        logger.debug(f"Content preview: {content[:200]}...")
        logger.debug(f"Estimated tokens: {total_chars // 4}")
        
    return "\n\n---\n\n".join(context)

async def format_response(knowledge: str, query: str) -> AsyncGenerator[str, None]:
    # Create a natural-sounding response to the user's question
    #
    # Args:
    #     knowledge: Information from our help articles
    #     query: The user's question
    #        
    # Yields:
    #     Pieces of the response, one at a time
    
    max_retries = 3
    base_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # First check if we have relevant information
            analysis = await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing whether documentation is relevant to a user's question. Respond with 'RELEVANT' or 'NOT_RELEVANT' followed by a brief reason."},
                    {"role": "user", "content": f"Question: {query}\n\nDocumentation:\n{knowledge}"}
                ],
                temperature=0,
                max_tokens=100
            )
            relevance = analysis.choices[0].message.content
            logger.debug(f"Relevance analysis: {relevance}")
            
            # If we don't have relevant info, let them know
            if relevance.startswith("NOT_RELEVANT"):
                yield "I'm sorry, but I'm not familiar with that aspect of Kayako."
                return
            
            # Generate a response one piece at a time
            stream = await openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"User's question: {query}\n\nRelevant documentation:\n{knowledge}"}
                ],
                max_tokens=500,
                stream=True
            )
            
            # Build up sentences one word at a time
            current_sentence = []
            async for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if not content:
                    continue
                
                # Add the new word to our current sentence
                current_sentence.append(content)
                
                # When we have a complete sentence, send it
                text = ''.join(current_sentence)
                if any(text.rstrip().endswith(char) for char in '.!?'):
                    yield text
                    current_sentence = []
            
            # Send any remaining words
            if current_sentence:
                yield ''.join(current_sentence)
            return
            
        except RateLimitError as e:
            # Handle hitting OpenAI's rate limits
            if attempt == max_retries - 1:
                logger.error(f"OpenAI rate limit exceeded after {max_retries} attempts")
                yield "I apologize, but I'm currently experiencing high demand. Please try again in a moment."
                return
            delay = base_delay * (2 ** attempt)  # Wait longer between each retry
            await asyncio.sleep(delay)
        except Exception as e:
            # Handle any other errors
            logger.error(f"Error in OpenAI API call: {e}")
            yield "I apologize, but I'm having trouble processing your request right now."
            return

@app.get("/monitor/cache")
async def get_cache_stats():
    # Get information about how well our caching is working
    return response_cache.get_stats()

async def format_and_stream_response(websocket: WebSocket, knowledge: str, query: str, interrupt_event: asyncio.Event, 
                                   call_sid: str, streaming_processor: StreamingProcessor):
    """Format and stream response with proper sentence tracking and recovery"""
    stream = None
    try:
        # Set a timeout for the entire response generation
        async with asyncio.timeout(30):  # 30-second timeout
            # Check if we have a saved state to recover from
            response_state = ResponseState()
            if call_sid in streaming_processor.active_streams and \
               "last_state" in streaming_processor.active_streams[call_sid]:
                saved_state = streaming_processor.active_streams[call_sid]["last_state"]
                response_state = ResponseState.from_state(saved_state)
                
                # Send any previously completed sentences
                for sentence in response_state.sentences:
                    await websocket.send_json({
                        "event": "response",
                        "text": sentence.strip(),
                        "is_final": False,
                        "position": response_state.position
                    })
            
            streaming_processor.response_states[call_sid] = response_state
            
            # Start response generation
            stream = await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"User's question: {query}\n\nRelevant documentation:\n{knowledge}"}
                ],
                max_tokens=500,
                stream=True
            )
            
            async for chunk in stream:
                # Check for interruption
                if interrupt_event.is_set():
                    logger.info("Response generation interrupted by user")
                    break
                
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if not content:
                    continue
                
                # Update response state
                response_state.add_content(content)
                
                # Stream completed sentences
                if response_state.sentences:
                    last_sentence = response_state.sentences[-1]
                    await websocket.send_json({
                        "event": "response",
                        "text": last_sentence.strip(),
                        "is_final": False,
                        "position": response_state.position
                    })
                    response_state.sentences.clear()  # Clear sent sentences
            
            # Mark the last content as final if we weren't interrupted
            if not interrupt_event.is_set():
                response_state.is_complete = True
                final_text = ''.join(response_state.current_sentence)
                if final_text:
                    await websocket.send_json({
                        "event": "response",
                        "text": final_text.strip(),
                        "is_final": True,
                        "position": response_state.position + 1
                    })
                
    except asyncio.TimeoutError:
        logger.error("Response generation timed out")
        await websocket.send_json({
            "event": "error",
            "message": "Response generation timed out",
            "type": "timeout"
        })
    except websockets.exceptions.ConnectionClosed:
        logger.error("WebSocket connection closed unexpectedly")
        raise
    except Exception as e:
        logger.error(f"Error in response streaming: {e}")
        raise
    finally:
        # Ensure stream is properly closed
        if stream:
            await stream.aclose()
        # Clean up response state if complete
        if call_sid in streaming_processor.response_states and \
           streaming_processor.response_states[call_sid].is_complete:
            del streaming_processor.response_states[call_sid]
