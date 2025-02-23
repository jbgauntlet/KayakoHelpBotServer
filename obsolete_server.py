# # FastAPI server for Kayako Help Center voice bot
# # This is the main server file that handles voice calls and provides AI-powered responses

# # Import necessary libraries
# # FastAPI - for creating the web server
# # OpenAI - for AI capabilities (chat, speech-to-text, embeddings)
# # Twilio - for handling phone calls
# # Other utilities for various functionalities
# from fastapi import FastAPI, HTTPException, Request, WebSocket
# from fastapi.responses import Response, HTMLResponse
# from openai import OpenAI, RateLimitError
# from twilio.rest import Client
# from twilio.base.exceptions import TwilioRestException
# from dotenv import load_dotenv
# import json
# import os
# import logging
# import time
# import asyncio
# from typing import Dict, Literal, Optional, List, AsyncGenerator
# import base64
# from collections import OrderedDict
# import numpy as np
# from dataclasses import dataclass
# import websockets
# import re
# from twilio.twiml.voice_response import VoiceResponse, Start, Connect
# from fastapi import WebSocketDisconnect

# # Import our custom modules
# from prompts import SYSTEM_PROMPT, INTENT_CLASSIFICATION_PROMPT
# from rag import RAGRetriever

# # Set up logging to help track what's happening in the application
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # Load environment variables from .env file
# load_dotenv()

# # Create the FastAPI application
# app = FastAPI()

# # Get API keys from environment variables for security
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# TWILIO_SID = os.getenv("TWILIO_SID")
# TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
# TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")

# # Initialize our external service clients
# twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)  # For making phone calls
# openai_client = OpenAI(api_key=OPENAI_API_KEY)         # For AI capabilities

# # Set up our knowledge base system (RAG) with help articles
# rag = RAGRetriever(openai_client)
# rag.load_articles(['sample-help.json', 'sample-help-2.json'], embeddings_file='help_embeddings.npz')

# # Dictionary to store conversation information for each active call
# conversation_context: Dict[str, Dict] = {}

# # Constants to control conversation flow and prevent infinite loops
# MAX_INVALID_ATTEMPTS = 3    # How many times we'll allow unclear or off-topic responses
# MAX_SILENCE_ATTEMPTS = 3    # How many times we'll allow silence before ending call
# MAX_TOTAL_TURNS = 50       # Maximum back-and-forth exchanges in one call

# # Define what types of user intents we can recognize
# IntentType = Literal["END_CALL", "CONTINUE", "CONNECT_HUMAN", "UNCLEAR", "OFF_TOPIC"]

# @dataclass
# class AudioBuffer:
#     """Represents a buffer of audio data with metadata"""
#     chunks: List[bytes]
#     last_activity: float
#     transcript: str = ""
#     is_final: bool = False
#     state: str = "collecting"  # Add state tracking
#     MAX_BUFFER_SIZE = 1024 * 1024  # 1MB maximum buffer size
#     BACKPRESSURE_THRESHOLD = 768 * 1024  # 75% of max size
#     current_size: int = 0
#     needs_backpressure: bool = False

#     def add_chunk(self, chunk: bytes) -> bool:
#         """Add chunk to buffer if within size limits. Returns False if buffer is full."""
#         chunk_size = len(chunk)
        
#         # Check if adding this chunk would exceed max size
#         if self.current_size + chunk_size > self.MAX_BUFFER_SIZE:
#             self.needs_backpressure = True
#             return False
            
#         # Set backpressure flag if we're approaching the threshold
#         self.needs_backpressure = (self.current_size + chunk_size) > self.BACKPRESSURE_THRESHOLD
        
#         self.chunks.append(chunk)
#         self.current_size += chunk_size
#         return True

#     def clear(self):
#         """Clear the buffer"""
#         self.chunks = []
#         self.current_size = 0
#         self.transcript = ""
#         self.is_final = False
#         self.state = "collecting"
#         self.needs_backpressure = False

# class LRUCache:
#     """A cache that keeps the most recently used items and removes the least recently used ones"""
    
#     def __init__(self, capacity: int = 1000):
#         # Initialize the cache with a maximum size
#         self.cache = OrderedDict()  # Special dictionary that remembers order of items
#         self.capacity = capacity    # Maximum number of items we can store
#         self.hits = 0              # Count of successful cache lookups
#         self.misses = 0            # Count of unsuccessful cache lookups
#         self.logger = logging.getLogger(__name__)
        
#     def get(self, key: str) -> Optional[str]:
#         # Try to get an item from the cache
#         if key in self.cache:
#             self.hits += 1  # We found it! Count this as a success
#             self.cache.move_to_end(key)  # Move this item to "most recently used"
#             self.logger.debug(f"Cache hit for key: {key[:50]}...")
#             return self.cache[key]
#         self.misses += 1  # We didn't find it. Count this as a miss
#         self.logger.debug(f"Cache miss for key: {key[:50]}...")
#         return None
        
#     def put(self, key: str, value: str):
#         # Add or update an item in the cache
#         if key in self.cache:
#             self.cache.move_to_end(key)  # If it exists, move it to "most recently used"
#         self.cache[key] = value
#         if len(self.cache) > self.capacity:
#             # If we're over capacity, remove the least recently used item
#             self.cache.popitem(last=False)
            
#     def get_stats(self) -> dict:
#         # Get statistics about how well the cache is performing
#         total = self.hits + self.misses
#         hit_rate = (self.hits / total) * 100 if total > 0 else 0
#         return {
#             "size": len(self.cache),
#             "capacity": self.capacity,
#             "hits": self.hits,
#             "misses": self.misses,
#             "hit_rate": f"{hit_rate:.2f}%"
#         }

# class ResponseCache:
#     """A system to manage multiple caches for different types of responses"""
    
#     def __init__(self):
#         # Create two separate caches:
#         self.knowledge_cache = LRUCache(capacity=1000)  # For storing retrieved knowledge
#         self.response_cache = LRUCache(capacity=1000)   # For storing formatted responses
#         self.logger = logging.getLogger(__name__)
        
#     def _generate_knowledge_key(self, query: str) -> str:
#         # Create a unique key for storing knowledge results
#         # We normalize the query (lowercase, no extra spaces) to increase cache hits
#         return query.lower().strip()
        
#     def _generate_response_key(self, query: str, knowledge: str) -> str:
#         # Create a unique key for storing formatted responses
#         # We combine the query and a hash of the knowledge to ensure uniqueness
#         return f"{query.lower().strip()}:{hash(knowledge)}"
        
#     async def get_knowledge(self, query: str) -> Optional[str]:
#         # Try to get cached knowledge for a query, or retrieve new knowledge if not found
#         key = self._generate_knowledge_key(query)
#         if cached := self.knowledge_cache.get(key):
#             return cached
        
#         # If not in cache, get new knowledge and store it
#         knowledge = retrieve_data(query)
#         if knowledge:
#             self.knowledge_cache.put(key, knowledge)
#         return knowledge
        
#     async def get_response(self, query: str, knowledge: str) -> Optional[str]:
#         # Try to get a cached response, or generate new one if not found
#         key = self._generate_response_key(query, knowledge)
#         if cached := self.response_cache.get(key):
#             return cached
            
#         # If not in cache, generate new response and store it
#         response = await format_response(knowledge, query)
#         if response:
#             self.response_cache.put(key, response)
#         return response
        
#     def get_stats(self) -> dict:
#         # Get performance statistics from both caches
#         knowledge_stats = self.knowledge_cache.get_stats()
#         response_stats = self.response_cache.get_stats()
#         return {
#             "knowledge_cache": knowledge_stats,
#             "response_cache": response_stats
#         }
        
#     async def prewarm(self, common_queries: List[str]):
#         # Fill the cache with responses to common questions before they're needed
#         self.logger.info("Starting cache prewarming...")
#         start_time = time.time()
        
#         for query in common_queries:
#             try:
#                 # First get and cache the knowledge
#                 knowledge = await self.get_knowledge(query)
#                 if knowledge:
#                     # Then get and cache the response
#                     await self.get_response(query, knowledge)
#                     self.logger.debug(f"Prewarmed cache for query: {query[:50]}...")
#             except Exception as e:
#                 self.logger.error(f"Error prewarming cache for query '{query}': {e}")
                
#         duration = time.time() - start_time
#         self.logger.info(f"Cache prewarming completed in {duration:.2f} seconds")
        
#     def clear(self):
#         # Reset both caches to empty state
#         self.knowledge_cache = LRUCache(capacity=self.knowledge_cache.capacity)
#         self.response_cache = LRUCache(capacity=self.response_cache.capacity)
#         self.logger.info("Caches cleared")

# # List of common questions we'll prepare answers for
# COMMON_QUERIES = [
#     "How do I set up Single Sign-On (SSO) for admin users?",
#     "How do I manage administrator roles and permissions?",
#     "How do I customize the branding of my AdvocateHub?",
#     "How do I add or upgrade users to administrator accounts?",
#     "How do I customize the sign-in page?",
#     "How do I change the name of my AdvocateHub?",
#     "How do I edit administrator profiles?",
#     "How do I contact the admin team?",
#     "How do I switch between administrator and advocate accounts?",
#     "How do I customize the CSS and styling of my hub?"
# ]

# # Create our response cache system
# response_cache = ResponseCache()

# # Add startup event to prewarm cache
# @app.on_event("startup")
# async def startup_event():
#     """Initialize and prewarm cache on startup"""
#     try:
#         await response_cache.prewarm(COMMON_QUERIES)
#     except Exception as e:
#         logger.error(f"Error during cache prewarming: {e}")

# # Add endpoint to manually trigger cache prewarming
# @app.post("/admin/cache/prewarm")
# async def prewarm_cache():
#     """Manually trigger cache prewarming"""
#     try:
#         # Clear existing cache first
#         response_cache.clear()
#         # Prewarm with common queries
#         await response_cache.prewarm(COMMON_QUERIES)
#         return {"status": "success", "message": "Cache prewarming completed"}
#     except Exception as e:
#         logger.error(f"Error during manual cache prewarming: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

# class WebSocketRetry:
#     """Handles WebSocket connection retries with exponential backoff"""
#     def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
#         self.max_retries = max_retries
#         self.base_delay = base_delay
#         self.logger = logging.getLogger(__name__)
        
#     async def __aenter__(self):
#         return self
        
#     async def __aexit__(self, exc_type, exc_val, exc_tb):
#         pass
        
#     async def connect(self, url: str, headers: dict) -> websockets.WebSocketClientProtocol:
#         last_exception = None
#         for attempt in range(self.max_retries):
#             try:
#                 if attempt > 0:
#                     delay = self.base_delay * (2 ** (attempt - 1))
#                     self.logger.info(f"Retrying WebSocket connection in {delay} seconds (attempt {attempt + 1}/{self.max_retries})")
#                     await asyncio.sleep(delay)
                
#                 return await websockets.connect(url, headers=headers)
                
#             except websockets.exceptions.WebSocketException as e:
#                 last_exception = e
#                 self.logger.warning(f"WebSocket connection attempt {attempt + 1} failed: {e}")
#                 continue
                
#         raise last_exception or RuntimeError("Failed to establish WebSocket connection")

# class ResponseState:
#     """Tracks state of response generation"""
#     def __init__(self):
#         self.sentences: List[str] = []
#         self.current_sentence: List[str] = []
#         self.position: int = 0
#         self.is_complete: bool = False
#         self.last_token: Optional[str] = None
        
#     def add_content(self, content: str):
#         """Add content and track sentence completion"""
#         self.current_sentence.append(content)
#         self.last_token = content
        
#         # Check if we completed a sentence
#         text = ''.join(self.current_sentence)
#         if any(text.rstrip().endswith(char) for char in '.!?'):
#             self.sentences.append(text)
#             self.current_sentence = []
#             self.position = len(self.sentences)
            
#     def get_current_state(self) -> dict:
#         """Get current state for recovery"""
#         return {
#             "sentences": self.sentences.copy(),
#             "current": ''.join(self.current_sentence),
#             "position": self.position,
#             "is_complete": self.is_complete
#         }
        
#     @classmethod
#     def from_state(cls, state: dict) -> 'ResponseState':
#         """Recreate state from saved state"""
#         response_state = cls()
#         response_state.sentences = state["sentences"]
#         response_state.current_sentence = list(state["current"])
#         response_state.position = state["position"]
#         response_state.is_complete = state["is_complete"]
#         return response_state

# class StreamingProcessor:
#     """Handles real-time processing of audio from phone calls"""
    
#     def __init__(self, 
#                  silence_threshold: float = 0.4,
#                  max_buffer_age: float = 2.0,
#                  min_transcript_len: int = 10):
#         self.buffers: Dict[str, AudioBuffer] = {}
#         self.active_streams: Dict[str, Dict] = {}  # Track active response streams
#         self.response_states: Dict[str, ResponseState] = {}  # Track response states
#         self.silence_threshold = silence_threshold
#         self.max_buffer_age = max_buffer_age
#         self.min_transcript_len = min_transcript_len
#         self.logger = logging.getLogger(__name__)
#         self.openai_client = OpenAI()
        
#     async def handle_interrupt(self, call_sid: str):
#         """Handle user interruption of response"""
#         if call_sid in self.active_streams:
#             # Save current state before interrupting
#             if call_sid in self.response_states:
#                 self.active_streams[call_sid]["last_state"] = \
#                     self.response_states[call_sid].get_current_state()
            
#             # Signal interruption
#             self.active_streams[call_sid]["interrupt"].set()
#             # Wait for cleanup
#             await asyncio.sleep(0.1)
#             # Remove from active streams
#             self.active_streams.pop(call_sid, None)
            
#     async def start_response_stream(self, call_sid: str, websocket: WebSocket, knowledge: str, query: str):
#         """Start a new response stream with interruption handling"""
#         # Create interruption event
#         interrupt_event = asyncio.Event()
        
#         # Store stream info
#         self.active_streams[call_sid] = {
#             "interrupt": interrupt_event,
#             "start_time": time.time()
#         }
        
#         try:
#             await format_and_stream_response(
#                 websocket=websocket,
#                 knowledge=knowledge,
#                 query=query,
#                 interrupt_event=interrupt_event,
#                 call_sid=call_sid,
#                 streaming_processor=self
#             )
#         finally:
#             # Clean up stream tracking
#             self.active_streams.pop(call_sid, None)
        
#     async def process_chunk(self, call_sid: str, chunk: bytes, websocket: WebSocket) -> Optional[str]:
#         try:
#             # Get or create buffer for this call
#             if call_sid not in self.buffers:
#                 self.buffers[call_sid] = AudioBuffer(
#                     chunks=[],
#                     last_activity=time.time(),
#                     state="collecting"
#                 )
#             buffer = self.buffers[call_sid]
            
#             # Add new chunk and update activity
#             if not buffer.add_chunk(chunk):
#                 # Buffer is full, signal backpressure
#                 await websocket.send_json({
#                     "event": "backpressure",
#                     "action": "pause",
#                     "reason": "Buffer full"
#                 })
#                 return None
            
#             # Signal backpressure if we're approaching threshold
#             if buffer.needs_backpressure:
#                 await websocket.send_json({
#                     "event": "backpressure",
#                     "action": "slow",
#                     "reason": "Buffer nearly full"
#                 })
            
#             buffer.last_activity = time.time()
            
#             # Check if we should process based on our rules
#             if await self._should_process_buffer(buffer):
#                 # Process accumulated audio using OpenAI's WebSocket with retries
#                 async with asyncio.timeout(10.0):  # 10-second timeout for audio processing
#                     async with WebSocketRetry(max_retries=3) as retry:
#                         ws = await retry.connect(
#                             'wss://api.openai.com/v1/audio/speech',
#                             headers={
#                                 "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
#                                 "Content-Type": "application/json"
#                             }
#                         )
#                         async with ws:
#                             # Send audio data
#                             await ws.send(json.dumps({
#                                 "model": "whisper-1",
#                                 "language": "en",
#                                 "audio_format": "mulaw",
#                                 "sample_rate": 8000
#                             }))
                            
#                             # Send audio chunks
#                             for audio_chunk in buffer.chunks:
#                                 await ws.send(audio_chunk)
                            
#                             # Get transcription
#                             result = await ws.recv()
#                             transcript_data = json.loads(result)
#                             buffer.transcript = transcript_data.get("text", "")
                            
#                             # Clear buffer and signal ready for more
#                             transcript = await self._process_buffer(call_sid)
#                             if transcript:
#                                 await websocket.send_json({
#                                     "event": "backpressure",
#                                     "action": "resume",
#                                     "reason": "Buffer processed"
#                                 })
#                             return transcript
            
#             return None
            
#         except asyncio.TimeoutError:
#             self.logger.error("Audio processing timed out")
#             return None
#         except Exception as e:
#             self.logger.error(f"Error processing audio chunk: {e}")
#             raise
            
#     async def _should_process_buffer(self, buffer: AudioBuffer) -> bool:
#         # Decide if we have enough audio to process
#         current_time = time.time()
        
#         # Calculate how long since last audio
#         buffer_age = current_time - buffer.last_activity
#         has_minimum_content = len(buffer.transcript.strip()) >= self.min_transcript_len
        
#         # Process if we have either:
#         # 1. A natural pause and enough content
#         # 2. Been holding the audio too long but have some content
#         return (buffer_age >= self.silence_threshold and has_minimum_content) or \
#                (buffer_age >= self.max_buffer_age and has_minimum_content)
               
#     async def _process_buffer(self, call_sid: str) -> Optional[str]:
#         # Convert our collected audio into text
#         try:
#             buffer = self.buffers[call_sid]
            
#             if not buffer.transcript:
#                 return None
                
#             # Get the final text version
#             transcript = buffer.transcript.strip()
            
#             # Clear the buffer for next time
#             self.buffers[call_sid] = AudioBuffer(
#                 chunks=[],
#                 last_activity=time.time()
#             )
            
#             self.logger.info(f"Processed transcript: {transcript}")
#             return transcript
            
#         except Exception as e:
#             self.logger.error(f"Error processing buffer: {e}")
#             return None
            
#     async def cleanup_old_buffers(self):
#         # Remove any audio buffers that haven't been used in a while
#         current_time = time.time()
#         to_remove = []
        
#         for call_sid, buffer in self.buffers.items():
#             if current_time - buffer.last_activity > self.max_buffer_age * 2:
#                 to_remove.append(call_sid)
                
#         for call_sid in to_remove:
#             del self.buffers[call_sid]
            
#     def get_buffer_status(self, call_sid: str) -> dict:
#         # Get information about a specific call's audio buffer
#         if call_sid not in self.buffers:
#             return {"status": "no_buffer"}
            
#         buffer = self.buffers[call_sid]
#         return {
#             "chunks": len(buffer.chunks),
#             "last_activity": datetime.fromtimestamp(buffer.last_activity).isoformat(),
#             "transcript_length": len(buffer.transcript),
#             "is_final": buffer.is_final
#         }

# # Create our audio processor
# streaming_processor = StreamingProcessor()

# def classify_intent(response: str, context: str) -> IntentType:
#     # Figure out what the user wants to do based on their response
#     try:
#         # Ask OpenAI to classify the user's intent
#         classification = openai_client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": INTENT_CLASSIFICATION_PROMPT},
#                 {"role": "user", "content": f"Context: {context}\nUser's response: {response}"}
#             ],
#             temperature=0,  # Use consistent responses
#             max_tokens=10   # We only need a short answer
#         )
#         intent = classification.choices[0].message.content.strip()
#         logger.debug(f"Classified intent: {intent} for response: {response}")
        
#         # Make sure we got a valid intent type
#         if intent in ["END_CALL", "CONTINUE", "CONNECT_HUMAN", "UNCLEAR", "OFF_TOPIC"]:
#             return intent
#         return "UNCLEAR"
#     except Exception as e:
#         logger.error(f"Error in intent classification: {e}")
#         return "UNCLEAR"

# @app.post("/incoming-call")
# async def handle_incoming_call(request: Request):
#     """Handle incoming call and return TwiML response to connect to Media Stream."""
#     logger.info("Received incoming call request")
#     response = VoiceResponse()
#     # <Say> punctuation to improve text-to-speech flow
#     response.say("Please wait while we connect your call to the A. I. voice assistant, powered by Twilio and the Open-A.I. Realtime API")
#     response.pause(length=1)
#     response.say("O.K. you can start talking!")

#     host = request.url.hostname
#     logger.info(f"Setting up WebSocket connection with host: {host}")
#     connect = Connect()
#     connect.stream(url=f'wss://{host}/media-stream')
#     response.append(connect)
#     logger.info("TwiML response created with WebSocket stream connection")
#     return HTMLResponse(content=str(response), media_type="application/xml")

# @app.websocket("/media-stream")
# async def handle_media_stream(websocket: WebSocket):
#     """Handle WebSocket connections with agentic capabilities."""
#     agent_state = AgentState()
    
#     try:
#         await websocket.accept()
#         logger.info("WebSocket connection accepted")

#         # Create connection to OpenAI's realtime API
#         openai_headers = {
#                 "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
#                 "OpenAI-Beta": "realtime=v1"
#             }
#         async with websockets.connect(
#             'wss://api.openai.com/v1/realtime?model=gpt-4-turbo-preview',
#             additional_headers=openai_headers
#         ) as openai_ws:
#             await initialize_session(openai_ws)

#             # Set up state tracking
#             stream_sid = None
#             latest_transcript = ""
            
#             async def process_speech_to_text():
#                 nonlocal latest_transcript
#                 try:
#                     async for message in websocket.iter_text():
#                         data = json.loads(message)
#                         if data['event'] == 'media':
#                             # Process audio chunk and get transcript
#                             transcript = await streaming_processor.process_chunk(
#                                 stream_sid, 
#                                 data['media']['payload'],
#                                 websocket
#                             )
#                             if transcript:
#                                 latest_transcript = transcript
#                                 # Process with agent
#                                 action = await agent_state.process_user_input(transcript)
#                                 await handle_agent_action(action, openai_ws, websocket)
                                
#                 except WebSocketDisconnect:
#                     logger.error("WebSocket disconnected")
                    
#             async def handle_agent_action(action: dict, openai_ws, websocket):
#                 """Handle different agent actions."""
#                 if action["action"] == "respond":
#                     # Use our existing response streaming
#                     await format_and_stream_response(
#                         websocket=websocket,
#                         knowledge=action["knowledge"],
#                         query=action["query"],
#                         interrupt_event=asyncio.Event(),
#                         call_sid=stream_sid,
#                         streaming_processor=streaming_processor
#                     )
#                 elif action["action"] in ["transfer", "end", "clarify"]:
#                     # Send simple response
#                     response = {
#                         "type": "conversation.item.create",
#                         "item": {
#                             "type": "message",
#                             "role": "assistant",
#                             "content": action["message"]
#                         }
#                     }
#                     await openai_ws.send(json.dumps(response))
                    
#                     if action["action"] == "transfer":
#                         # Implement call transfer logic
#                         await transfer_to_human_agent(stream_sid)
#                     elif action["action"] == "end":
#                         await websocket.close()
                        
#             # Start processing loops
#             await asyncio.gather(
#                 process_speech_to_text(),
#                 handle_openai_responses(openai_ws, websocket)
#             )
            
#                 except Exception as e:
#         logger.error(f"Error in handle_media_stream: {str(e)}", exc_info=True)
#     finally:
#         logger.info("WebSocket connection closed")

# async def initialize_session(openai_ws):
#     """Initialize session with OpenAI including our agentic capabilities."""
#     try:
#         logger.info("Initializing OpenAI session")
        
#         # Load initial RAG context
#         initial_context = await rag.get_initial_context()
        
#         session_update = {
#             "type": "session.update",
#             "session": {
#                 "turn_detection": {"type": "server_vad"},
#                 "input_audio_format": "g711_ulaw",
#                 "output_audio_format": "g711_ulaw",
#                 "voice": "alloy",
#                 "instructions": SYSTEM_PROMPT,  # Use our comprehensive system prompt
#                 "context": {
#                     "rag_knowledge": initial_context,
#                     "conversation_history": [],
#                     "user_intents": []
#                 },
#                 "modalities": ["text", "audio"],
#                 "temperature": 0.7,
#             }
#         }
#         await openai_ws.send(json.dumps(session_update))
#         logger.info("Session update sent successfully")

#         await send_initial_conversation_item(openai_ws)
#     except Exception as e:
#         logger.error(f"Error in initialize_session: {str(e)}", exc_info=True)
#         raise

# async def send_initial_conversation_item(openai_ws):
#     """Send initial conversation item for AI greeting."""
#     try:
#         logger.info("Sending initial conversation item")
#         initial_conversation_item = {
#             "type": "conversation.item.create",
#             "item": {
#                 "type": "message",
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "input_text",
#                         "text": "Greet the user and ask how you can help them with Kayako today."
#                     }
#                 ]
#             }
#         }
#         await openai_ws.send(json.dumps(initial_conversation_item))
#         logger.info("Initial conversation item sent")
#         await openai_ws.send(json.dumps({"type": "response.create"}))
#         logger.info("Response creation triggered")
#     except Exception as e:
#         logger.error(f"Error in send_initial_conversation_item: {str(e)}", exc_info=True)
#         raise

# @app.get("/monitor/cache")
# async def get_cache_stats():
#     # Get information about how well our caching is working
#     return response_cache.get_stats()

# async def format_and_stream_response(websocket: WebSocket, knowledge: str, query: str, interrupt_event: asyncio.Event, 
#                                    call_sid: str, streaming_processor: StreamingProcessor):
#     """Format and stream response with proper sentence tracking and recovery"""
#     stream = None
#     try:
#         # Set a timeout for the entire response generation
#         async with asyncio.timeout(30):  # 30-second timeout
#             # Check if we have a saved state to recover from
#             response_state = ResponseState()
#             if call_sid in streaming_processor.active_streams and \
#                "last_state" in streaming_processor.active_streams[call_sid]:
#                 saved_state = streaming_processor.active_streams[call_sid]["last_state"]
#                 response_state = ResponseState.from_state(saved_state)
                
#                 # Send any previously completed sentences
#                 for sentence in response_state.sentences:
#                     await websocket.send_json({
#                         "event": "response",
#                         "text": sentence.strip(),
#                         "is_final": False,
#                         "position": response_state.position
#                     })
            
#             streaming_processor.response_states[call_sid] = response_state
            
#             # Start response generation
#             stream = await openai_client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "system", "content": SYSTEM_PROMPT},
#                     {"role": "user", "content": f"User's question: {query}\n\nRelevant documentation:\n{knowledge}"}
#                 ],
#                 max_tokens=500,
#                 stream=True
#             )
            
#             async for chunk in stream:
#                 # Check for interruption
#                 if interrupt_event.is_set():
#                     logger.info("Response generation interrupted by user")
#                     break
                
#                 if not chunk.choices:
#                     continue
#                 content = chunk.choices[0].delta.content
#                 if not content:
#                     continue
                
#                 # Update response state
#                 response_state.add_content(content)
                
#                 # Stream completed sentences
#                 if response_state.sentences:
#                     last_sentence = response_state.sentences[-1]
#                     await websocket.send_json({
#                         "event": "response",
#                         "text": last_sentence.strip(),
#                         "is_final": False,
#                         "position": response_state.position
#                     })
#                     response_state.sentences.clear()  # Clear sent sentences
            
#             # Mark the last content as final if we weren't interrupted
#             if not interrupt_event.is_set():
#                 response_state.is_complete = True
#                 final_text = ''.join(response_state.current_sentence)
#                 if final_text:
#                     await websocket.send_json({
#                         "event": "response",
#                         "text": final_text.strip(),
#                         "is_final": True,
#                         "position": response_state.position + 1
#                     })
                
#     except asyncio.TimeoutError:
#         logger.error("Response generation timed out")
#         await websocket.send_json({
#             "event": "error",
#             "message": "Response generation timed out",
#             "type": "timeout"
#         })
#     except websockets.exceptions.ConnectionClosed:
#         logger.error("WebSocket connection closed unexpectedly")
#         raise
#     except Exception as e:
#         logger.error(f"Error in response streaming: {e}")
#         raise
#     finally:
#         # Ensure stream is properly closed
#         if stream:
#             await stream.aclose()
#         # Clean up response state if complete
#         if call_sid in streaming_processor.response_states and \
#            streaming_processor.response_states[call_sid].is_complete:
#             del streaming_processor.response_states[call_sid]

# class AgentState:
#     def __init__(self):
#         self.conversation_history = []
#         self.current_intent = None
#         self.invalid_attempts = 0
#         self.silence_attempts = 0
#         self.total_turns = 0
#         self.last_rag_context = None
        
#     async def process_user_input(self, text: str) -> dict:
#         """Process user input and return next action."""
#         self.total_turns += 1
        
#         # Classify intent
#         intent = classify_intent(text, "\n".join(self.conversation_history))
#         self.current_intent = intent
        
#         # Get relevant knowledge if continuing conversation
#         if intent == "CONTINUE":
#             knowledge = await rag.get_relevant_articles(text)
#             self.last_rag_context = knowledge
#             return {
#                 "action": "respond",
#                 "knowledge": knowledge,
#                 "query": text
#             }
#         elif intent == "CONNECT_HUMAN":
#             return {
#                 "action": "transfer",
#                 "message": "Connecting you to a human agent..."
#             }
#         elif intent == "END_CALL":
#             return {
#                 "action": "end",
#                 "message": "Thank you for calling. Goodbye!"
#             }
#         elif intent in ["UNCLEAR", "OFF_TOPIC"]:
#             self.invalid_attempts += 1
#             if self.invalid_attempts >= MAX_INVALID_ATTEMPTS:
#                 return {
#                     "action": "transfer",
#                     "message": "I'm having trouble understanding. Let me connect you with a human agent."
#                 }
#             return {
#                 "action": "clarify",
#                 "message": "I'm not sure I understood. Could you please rephrase that?"
#             }

# import os
# import json
# import base64
# import asyncio
# import websockets
# from fastapi import FastAPI, WebSocket, Request
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.websockets import WebSocketDisconnect
# from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
# from dotenv import load_dotenv
# from openai import OpenAI
# from rag import RAGRetriever

# load_dotenv()

# # Configuration
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# PORT = int(os.getenv('PORT', 5050))

# # Initialize OpenAI client and RAG retriever
# openai_client = OpenAI(api_key=OPENAI_API_KEY)
# rag_retriever = RAGRetriever(openai_client)

# # Load help articles and embeddings
# HELP_ARTICLES = ['sample-help.json', 'sample-help-2.json']
# EMBEDDINGS_FILE = 'help_embeddings.npz'
# rag_retriever.load_articles(HELP_ARTICLES, EMBEDDINGS_FILE)

# SYSTEM_MESSAGE = """You are a helpful customer support agent for Kayako. You specialize in providing clear, actionable answers based on Kayako's documentation.

# When responding:
# 1. Be concise and clear - suitable for phone conversation
# 2. Use a natural, conversational tone
# 3. Focus on providing specific, actionable steps
# 4. If the documentation contains relevant information, even if partial, use it to help the user
# 5. Never suggest connecting to a human agent - the system will handle that automatically

# Evaluate the provided documentation:
# - If it contains ANY relevant information to answer the question, use it to provide specific guidance
# - If it's completely unrelated or doesn't help answer the question at all, respond with:
# "I'm sorry, but I'm not familiar with that aspect of Kayako."

# When providing instructions:
# - Convert any technical steps into natural spoken language
# - Focus on the "what" and "how" rather than technical details
# - Keep steps sequential and clear
# - Avoid technical jargon unless necessary

# Keep responses under 3-4 sentences when possible, but ensure all critical steps are included."""

# VOICE = 'alloy'
# LOG_EVENT_TYPES = [
#     'error', 'response.content.done', 'rate_limits.updated',
#     'response.done', 'input_audio_buffer.committed',
#     'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
#     'session.created'
# ]
# SHOW_TIMING_MATH = False

# app = FastAPI()

# if not OPENAI_API_KEY:
#     raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

# @app.get("/", response_class=JSONResponse)
# async def index_page():
#     return {"message": "Twilio Media Stream Server is running!"}

# @app.api_route("/incoming-call", methods=["GET", "POST"])
# async def handle_incoming_call(request: Request):
#     """Handle incoming call and return TwiML response to connect to Media Stream."""
#     response = VoiceResponse()
#     # <Say> punctuation to improve text-to-speech flow
#     response.say("Please wait while we connect your call to the A. I. voice assistant, powered by Twilio and the Open-A.I. Realtime API")
#     response.pause(length=1)
#     response.say("O.K. you can start talking!")
#     host = request.url.hostname
#     connect = Connect()
#     connect.stream(url=f'wss://{host}/media-stream')
#     response.append(connect)
#     return HTMLResponse(content=str(response), media_type="application/xml")

# @app.websocket("/media-stream")
# async def handle_media_stream(websocket: WebSocket):
#     """Handle WebSocket connections between Twilio and OpenAI."""
#     await websocket.accept()
#     stream_sid = None
#     mark_queue = []
#     last_assistant_item = None
#     response_start_timestamp_twilio = None
#     latest_media_timestamp = 0

#     async def process_user_query(query: str) -> str:
#         """Process user query using RAG retriever to get relevant context."""
#         # Get relevant chunks from the knowledge base
#         relevant_chunks = rag_retriever.retrieve(query)
        
#         if not relevant_chunks:
#             return "I'm sorry, but I'm not familiar with that aspect of Kayako."
            
#         # Format context from relevant chunks
#         context = "\n\n".join([
#             f"From article '{chunk['title']}':\n{chunk['text']}"
#             for chunk in relevant_chunks
#         ])
        
#         return context

#     async def receive_from_twilio():
#         """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
#         nonlocal stream_sid, latest_media_timestamp
#         try:
#             async for message in websocket.iter_text():
#                 data = json.loads(message)
#                 if data['event'] == 'media' and openai_client.open:
#                     latest_media_timestamp = int(data['media']['timestamp'])
#                     audio_append = {
#                         "type": "input_audio_buffer.append",
#                         "audio": data['media']['payload']
#                     }
#                     await openai_client.send(json.dumps(audio_append))
#                 elif data['event'] == 'start':
#                     stream_sid = data['start']['streamSid']
#                     print(f"Incoming stream has started {stream_sid}")
#                     response_start_timestamp_twilio = None
#                     latest_media_timestamp = 0
#                     last_assistant_item = None
#                 elif data['event'] == 'mark':
#                     if mark_queue:
#                         mark_queue.pop(0)
#                 elif data['event'] == 'stop':
#                     print(f"Call ended, stream {stream_sid} stopped")
#                     if openai_client.open:
#                         await openai_client.close()
#                     await websocket.close()
#                     return
#         except WebSocketDisconnect:
#             print("Client disconnected.")
#             if openai_client.open:
#                 await openai_client.close()

#     async def send_to_twilio():
#         """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
#         nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio, openai_client
#         try:
#             async for openai_message in openai_client:
#                 response = json.loads(openai_message)
#                 if response['type'] in LOG_EVENT_TYPES:
#                     print(f"Received event: {response['type']}", response)

#                 # If we receive transcribed text, use RAG to get relevant context
#                 if response.get('type') == 'text' and response.get('text'):
#                     user_query = response['text']
#                     context = await process_user_query(user_query)
                    
#                     # Send context and user query to OpenAI
#                     conversation_item = {
#                         "type": "conversation.item.create",
#                         "item": {
#                             "type": "message",
#                             "role": "user",
#                             "content": [
#                                 {
#                                     "type": "text",
#                                     "text": f"Context from knowledge base:\n{context}\n\nUser query: {user_query}\n\nRespond to the user query using the context provided. If the context is not relevant, inform the user that you're not familiar with that aspect of Kayako."
#                                 }
#                             ]
#                         }
#                     }
#                     await openai_client.send(json.dumps(conversation_item))
#                     await openai_client.send(json.dumps({"type": "response.create"}))

#                 if response.get('type') == 'response.audio.delta' and 'delta' in response:
#                     audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
#                     audio_delta = {
#                         "event": "media",
#                         "streamSid": stream_sid,
#                         "media": {
#                             "payload": audio_payload
#                         }
#                     }
#                     await websocket.send_json(audio_delta)

#                     if response_start_timestamp_twilio is None:
#                         response_start_timestamp_twilio = latest_media_timestamp
#                         if SHOW_TIMING_MATH:
#                             print(f"Setting start timestamp for new response: {response_start_timestamp_twilio}ms")

#                     # Update last_assistant_item safely
#                     if response.get('item_id'):
#                         last_assistant_item = response['item_id']

#                     await send_mark(websocket, stream_sid)

#                 # Trigger an interruption. Your use case might work better using `input_audio_buffer.speech_stopped`, or combining the two.
#                 if response.get('type') == 'input_audio_buffer.speech_started':
#                     print("Speech started detected.")
#                     if last_assistant_item:
#                         print(f"Interrupting response with id: {last_assistant_item}")
#                         await handle_speech_started_event()
#         except Exception as e:
#             print(f"Error in send_to_twilio: {e}")
#             raise

#     async def handle_speech_started_event():
#         """Handle interruption when the caller's speech starts."""
#         nonlocal response_start_timestamp_twilio, last_assistant_item, openai_client
#         print("Handling speech started event.")
#         if mark_queue and response_start_timestamp_twilio is not None:
#             elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
#             if SHOW_TIMING_MATH:
#                 print(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

#             if last_assistant_item:
#                 if SHOW_TIMING_MATH:
#                     print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

#                 truncate_event = {
#                     "type": "conversation.item.truncate",
#                     "item_id": last_assistant_item,
#                     "content_index": 0,
#                     "audio_end_ms": elapsed_time
#                 }
#                 await openai_client.send(json.dumps(truncate_event))

#             await websocket.send_json({
#                 "event": "clear",
#                 "streamSid": stream_sid
#             })

#             mark_queue.clear()
#             last_assistant_item = None
#             response_start_timestamp_twilio = None

#     async def send_mark(connection, stream_sid):
#         if stream_sid:
#             mark_event = {
#                 "event": "mark",
#                 "streamSid": stream_sid,
#                 "mark": {"name": "responsePart"}
#             }
#             await connection.send_json(mark_event)
#             mark_queue.append('responsePart')

#     try:
#         async with websockets.connect(
#             'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
#             extra_headers={
#                 "Authorization": f"Bearer {OPENAI_API_KEY}",
#                 "OpenAI-Beta": "realtime=v1"
#             }
#         ) as openai_ws:
#             await initialize_session(openai_ws)
#             await send_initial_conversation_item(openai_ws)

#             # Create tasks for receiving from Twilio and sending to Twilio
#             receive_task = asyncio.create_task(receive_from_twilio())
#             send_task = asyncio.create_task(send_to_twilio())

#             # Wait for either task to complete (or raise an exception)
#             await asyncio.gather(receive_task, send_task)
#     except Exception as e:
#         print(f"Error in WebSocket handler: {e}")
#         # Ensure WebSocket connections are closed
#         await websocket.close()
#         raise  # Re-raise the exception for proper error handling
#     finally:
#         print("WebSocket connection closed")

# async def send_initial_conversation_item(openai_ws):
#     """Send initial conversation item if AI talks first."""
#     initial_conversation_item = {
#         "type": "conversation.item.create",
#         "item": {
#             "type": "message",
#             "role": "user",
#             "content": [
#                 {
#                     "type": "input_text",
#                     "text": "Greet the user with 'Hello! I am Kayako's help center assistant. I can help you with questions about Kayako's features and functionality. How can I assist you today?'"
#                 }
#             ]
#         }
#     }
#     await openai_ws.send(json.dumps(initial_conversation_item))
#     await openai_ws.send(json.dumps({"type": "response.create"}))

# async def initialize_session(openai_ws):
#     """Control initial session with OpenAI."""
#     session_update = {
#         "type": "session.update",
#         "session": {
#             "turn_detection": {"type": "server_vad"},
#             "input_audio_format": "g711_ulaw",
#             "output_audio_format": "g711_ulaw",
#             "voice": VOICE,
#             "instructions": SYSTEM_MESSAGE,
#             "modalities": ["text", "audio"],
#             "temperature": 0.8,
#         }
#     }
#     print('Sending session update:', json.dumps(session_update))
#     await openai_ws.send(json.dumps(session_update))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=PORT)