from fastapi import FastAPI, WebSocket
from deepgram import Deepgram
import openai
import json
import os
from twilio.rest import Client
from dotenv import load_dotenv

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

@app.websocket("/stream")
async def call_center_bot(websocket: WebSocket):
    await websocket.accept()
    print("Call started")

    async for message in websocket.iter_text():
        data = json.loads(message)

        # Convert audio to text (Deepgram STT)
        user_text = transcribe_audio(data["audio"])
        print(f"User said: {user_text}")

        # Retrieve knowledge (RAG System)
        knowledge = retrieve_data(user_text)

        # Use OpenAI to format response
        llm_response = format_response(knowledge)
        print(f"AI Response: {llm_response}")

        # Send response to Twilio TTS (Say API)
        send_twilio_tts(llm_response, data["call_sid"])

    print("Call ended")
    await websocket.close()

# Deepgram STT Function
def transcribe_audio(audio_chunk):
    deepgram = Deepgram(DEEPGRAM_API_KEY)
    response = deepgram.transcription.prerecorded({
        "buffer": audio_chunk,
        "mimetype": "audio/wav"
    })
    return response['results']['channels'][0]['alternatives'][0]['transcript']

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
