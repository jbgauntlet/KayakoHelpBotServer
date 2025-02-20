from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    AudioSource
)
from openai import OpenAI
import json
import os
from twilio.rest import Client
from dotenv import load_dotenv
import base64
import audioop
import time
import logging

from prompts import SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# API Keys from environment variables
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")

# Initialize Twilio Client
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Initialize Deepgram Client
deepgram = DeepgramClient(DEEPGRAM_API_KEY)

# Initialize OpenAI Client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

@app.websocket("/stream")
async def call_center_bot(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection accepted")
    
    call_sid = None
    audio_buffer = bytearray()
    text_buffer = []
    last_speech_time = None
    message_count = 0

    try:
        while True:
            message = await websocket.receive()
            message_count += 1
            logger.debug(f"Received message type: {message['type']}")
            
            if message["type"] == "websocket.disconnect":
                logger.info("Received disconnect message")
                break
                
            if message["type"] == "websocket.receive" and "text" in message:
                # Handle JSON messages (start/stop/media events)
                data = json.loads(message["text"])
                logger.debug(f"Received message event: {data['event']}")
                
                if data["event"] == "start":
                    call_sid = data["start"]["callSid"]
                    logger.info(f"Started streaming call {call_sid}")
                
                elif data["event"] == "media":
                    logger.debug("Processing media chunk")
                    # Decode base64 audio data
                    chunk = base64.b64decode(data["media"]["payload"])
                    # Convert from mulaw to PCM
                    chunk = audioop.ulaw2lin(chunk, 2)
                    # Add to buffer
                    audio_buffer.extend(chunk)
                    logger.debug(f"Current audio buffer size: {len(audio_buffer)}")
                    
                    # Process audio in chunks (every ~2 seconds)
                    if len(audio_buffer) >= 32000:  # 8000 samples/sec * 2 seconds * 2 bytes/sample for PCM
                        logger.info("Processing audio chunk of sufficient size")
                        # Convert audio to text
                        user_text = await transcribe_audio(bytes(audio_buffer))
                        logger.debug(f"Transcription result: '{user_text}'")
                        
                        if user_text.strip():
                            logger.info(f"Transcribed chunk: {user_text}")
                            text_buffer.append(user_text)
                            last_speech_time = time.time()
                        elif last_speech_time and time.time() - last_speech_time > 2.0:
                            if text_buffer:
                                complete_utterance = " ".join(text_buffer)
                                logger.info(f"Processing complete utterance: {complete_utterance}")
                                
                                knowledge = retrieve_data(complete_utterance)
                                llm_response = format_response(knowledge)
                                logger.info(f"AI Response: {llm_response}")
                                send_twilio_tts(llm_response, call_sid)
                                
                                text_buffer = []
                                last_speech_time = None
                        
                        # Clear audio buffer for next chunk
                        audio_buffer = bytearray()
                
                elif data["event"] == "stop":
                    logger.info("Received stop event")
                    # Process any remaining text before ending
                    if text_buffer:
                        complete_utterance = " ".join(text_buffer)
                        logger.info(f"Processing final utterance: {complete_utterance}")
                        knowledge = retrieve_data(complete_utterance)
                        llm_response = format_response(knowledge)
                        logger.info(f"Final AI Response: {llm_response}")
                        send_twilio_tts(llm_response, call_sid)
                    logger.info(f"Call ended after processing {message_count} messages")
                    break
    
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}", exc_info=True)
    finally:
        logger.info(f"Closing WebSocket connection. Processed {message_count} total messages")
        await websocket.close()

@app.post("/start-call")
async def start_call():
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Thank you for calling the Kayako Help Center today. How may I assist you?</Say>
    <Connect>
        <Stream url="wss://kayakohelpbotserver.fly.dev/stream" track="inbound_track">
            <Parameter name="format" value="audio/x-mulaw;rate=8000" />
        </Stream>
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

# Deepgram STT Function
async def transcribe_audio(audio_chunk):
    try:
        logger.debug(f"Attempting to transcribe audio chunk of size {len(audio_chunk)}")
        source = {
            "buffer": audio_chunk,
            "mimetype": "audio/l16"
        }
        options = PrerecordedOptions(
            smart_format=True,
            model="nova-2",
            encoding="linear16",
            sample_rate=8000
        )
        
        logger.debug("Sending request to Deepgram")
        response = await deepgram.listen.prerecorded.transcribe(
            source,
            options
        )
        
        if response and response.results and response.results.channels:
            transcript = response.results.channels[0].alternatives[0].transcript
            logger.debug(f"Transcription successful: {transcript}")
            return transcript
        logger.warning("No transcription result from Deepgram")
        return ""
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return ""

# RAG Knowledge Retrieval
def retrieve_data(query):
    # Placeholder: Replace with real RAG retrieval
    return f"Relevant info for: {query}"

# OpenAI LLM Response Formatting
def format_response(knowledge):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=knowledge, query="")},
            {"role": "user", "content": knowledge}
        ]
    )
    return response.choices[0].message.content

# Twilio Say API (TTS)
def send_twilio_tts(response_text, call_sid):
    twilio_client.calls(call_sid).update(twiml=f'<Response><Say>{response_text}</Say></Response>')
