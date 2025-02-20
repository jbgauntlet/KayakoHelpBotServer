from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
)
import openai
import json
import os
from twilio.rest import Client
from dotenv import load_dotenv
import base64
import audioop
import time

from prompts import SYSTEM_PROMPT

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

@app.websocket("/stream")
async def call_center_bot(websocket: WebSocket):
    await websocket.accept()
    print("Call started")
    
    call_sid = None
    audio_buffer = bytearray()
    text_buffer = []
    last_speech_time = None

    try:
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                break
                
            if message["type"] == "websocket.receive":
                if "text" in message:
                    # Handle JSON messages (start/stop events)
                    data = json.loads(message["text"])
                    if data["event"] == "start":
                        call_sid = data["start"]["callSid"]
                        print(f"Started streaming call {call_sid}")
                    elif data["event"] == "stop":
                        # Process any remaining text before ending
                        if text_buffer:
                            complete_utterance = " ".join(text_buffer)
                            print(f"Final utterance: {complete_utterance}")
                            knowledge = retrieve_data(complete_utterance)
                            llm_response = format_response(knowledge)
                            print(f"Final AI Response: {llm_response}")
                            send_twilio_tts(llm_response, call_sid)
                        print("Call ended")
                        break
                
                elif "bytes" in message:
                    # Handle binary messages (audio data)
                    data = json.loads(message["bytes"])
                    if data["event"] == "media":
                        # Decode base64 audio data
                        chunk = base64.b64decode(data["media"]["payload"])
                        # Convert from mulaw to PCM
                        chunk = audioop.ulaw2lin(chunk, 2)
                        # Add to buffer
                        audio_buffer.extend(chunk)
                        
                        # Process audio in chunks (every ~2 seconds)
                        if len(audio_buffer) >= 32000:  # 16000 samples/sec * 2 seconds * 2 bytes/sample
                            # Convert audio to text
                            user_text = await transcribe_audio(bytes(audio_buffer))
                            print(f"Processing audio chunk...")
                            
                            if user_text.strip():  # Only process if we got some text
                                print(f"Transcribed chunk: {user_text}")
                                text_buffer.append(user_text)
                                last_speech_time = time.time()
                            elif last_speech_time and time.time() - last_speech_time > 2.0:  # 2 second silence
                                # Process complete utterance
                                if text_buffer:
                                    complete_utterance = " ".join(text_buffer)
                                    print(f"Complete utterance: {complete_utterance}")
                                    
                                    # Now process the complete utterance
                                    knowledge = retrieve_data(complete_utterance)
                                    llm_response = format_response(knowledge)
                                    print(f"AI Response: {llm_response}")
                                    send_twilio_tts(llm_response, call_sid)
                                    
                                    # Clear the text buffer for next utterance
                                    text_buffer = []
                                    last_speech_time = None
                            
                            # Clear audio buffer for next chunk
                            audio_buffer = bytearray()
    
    except Exception as e:
        print(f"Error in WebSocket handler: {e}")
    finally:
        print("Closing WebSocket connection")
        await websocket.close()

@app.post("/start-call")
async def start_call():
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Thank you for calling the Kayako Help Center today. How may I assist you?</Say>
    <Connect>
        <Stream url="wss://kayakohelpbotserver.fly.dev/stream" />
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

# Deepgram STT Function
async def transcribe_audio(audio_chunk):
    try:
        options = PrerecordedOptions(
            smart_format=True,
            model="nova-2",
            encoding="linear16",
            sample_rate=8000
        )
        payload = {
            "buffer": audio_chunk,
        }
        response = await deepgram.transcription.prerecorded(payload, options)
        if response and response.results and response.results.channels:
            return response.results.channels[0].alternatives[0].transcript
        return ""
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

# RAG Knowledge Retrieval
def retrieve_data(query):
    # Placeholder: Replace with real RAG retrieval
    return f"Relevant info for: {query}"

# OpenAI LLM Response Formatting
def format_response(knowledge):
    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=knowledge, query="")},
            {"role": "user", "content": knowledge}
        ]
    )
    return response['choices'][0]['message']['content']

# Twilio Say API (TTS)
def send_twilio_tts(response_text, call_sid):
    twilio_client.calls(call_sid).update(twiml=f'<Response><Say>{response_text}</Say></Response>')
