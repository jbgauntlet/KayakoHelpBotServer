import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from openai import OpenAI
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv
from rag import RAGRetriever

load_dotenv()

# Load knowledge base
with open('knowledge_base.json', 'r') as f:
    knowledge_base = json.load(f)

# Format knowledge base articles into a string
knowledge_base_text = "\n\n".join([
    f"Article: {article['title']}\n"
    f"Category: {article['category']}\n"
    f"Summary: {article['summary']}\n"
    f"Content: {json.dumps(article['content'], indent=2)}"
    for article in knowledge_base['articles']
])

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))
SYSTEM_MESSAGE = f"""
You are a helpful and bubbly customer support agent for Kayako who loves to chat to customers. 
You specialize in providing clear, actionable answers based on Kayako's documentation.

When responding:
1. Be concise and clear - suitable for phone conversation
2. Use a natural, conversational tone
3. Focus on providing specific, actionable steps
4. If the documentation contains relevant information, even if partial, use it to help the user
5. Never suggest connecting to a human agent - the system will handle that automatically

Evaluate the provided documentation:
- If it contains ANY relevant information to answer the question, use it to provide specific guidance
- If it's completely unrelated or doesn't help answer the question at all, respond with:
"I'm sorry, but I'm not sure I can help you with that."
- After any response to a user query that is not a question, ask "Is there anything else I can help you with?"

When providing instructions:
- Convert any technical steps into natural spoken language
- Focus on the "what" and "how" rather than technical details
- Keep steps sequential and clear
- Avoid technical jargon unless necessary

Keep responses under 3-4 sentences when possible, but ensure all critical steps are included.

KNOWLEDGE BASE DOCUMENTATION:
{knowledge_base_text}
"""
# SYSTEM_MESSAGE = (
#     "You are a helpful and bubbly AI assistant who loves to chat about "
#     "anything the user is interested in and is prepared to offer them facts. "
#     "You have a penchant for dad jokes, owl jokes, and rickrolling – subtly. "
#     "Always stay positive, but work in a joke when appropriate."
# )
VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created'
]
SHOW_TIMING_MATH = False

app = FastAPI()

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize the RAG retriever
retriever = RAGRetriever(openai_client)

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    # <Say> punctuation to improve text-to-speech flow
    # response.say("Please wait while we connect your call to the A. I. voice assistant, powered by Twilio and the Open-A.I. Realtime API")
    # response.pause(length=1)
    # response.say("O.K. you can start talking!")
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    await websocket.accept()

    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01',
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        await initialize_session(openai_ws)

        # Connection specific state
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None
        
        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, latest_media_timestamp
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        latest_media_timestamp = int(data['media']['timestamp'])
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": data['media']['payload']
                        }
                        await openai_ws.send(json.dumps(audio_append))
                    elif data['event'] == 'start':
                        stream_sid = data['start']['streamSid']
                        print(f"Incoming stream has started {stream_sid}")
                        response_start_timestamp_twilio = None
                        latest_media_timestamp = 0
                        last_assistant_item = None
                    elif data['event'] == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)
                    elif data['event'] == 'stop':
                        print(f"Call ended, stream {stream_sid} stopped")
                        if openai_ws.open:
                            await openai_ws.close()
                        await websocket.close()
                        return
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.open:
                    await openai_ws.close()

        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)
                    print(f"Received response: {response}")

                    # if response.get('type') == 'conversation.item.input_audio_transcription.completed':
                    #     print(f"Received transcription: {response}")
                    #     transcription_text = response.get('transcription')
                    #     if transcription_text:
                    #         print("User transcription:", transcription_text)
                    #         # Retrieve relevant context based on the transcription
                    #         relevant_info = await retriever.retrieve(transcription_text)
                    #         # Inject this context into the conversation
                    #         add_context_event = {
                    #             "type": "conversation.item.create",
                    #             "item": {
                    #                 "type": "context",
                    #                 "role": "system",
                    #                 "content": relevant_info
                    #             }
                    #         }
                    #         await openai_ws.send(json.dumps(add_context_event))

                    if response['type'] in LOG_EVENT_TYPES:
                        print(f"Received event: {response['type']}", response)

                    if response.get('type') == 'response.content.done':
                        print(f"Response content done: {response}")
                        
                    elif response.get('type') == 'response.done':
                        print(f"Response done: {response}")
                    elif response.get('type') == 'response.audio.delta':
                        print(f"Response audio delta: {response}")

                    if response.get('type') == 'response.audio.delta' and 'delta' in response:
                        audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                        audio_delta = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": audio_payload
                            }
                        }
                        await websocket.send_json(audio_delta)

                        if response_start_timestamp_twilio is None:
                            response_start_timestamp_twilio = latest_media_timestamp
                            if SHOW_TIMING_MATH:
                                print(f"Setting start timestamp for new response: {response_start_timestamp_twilio}ms")

                        # Update last_assistant_item safely
                        if response.get('item_id'):
                            last_assistant_item = response['item_id']

                        await send_mark(websocket, stream_sid)

                    # Trigger an interruption. Your use case might work better using `input_audio_buffer.speech_stopped`, or combining the two.
                    if response.get('type') == 'input_audio_buffer.speech_started':
                        print("Speech started detected.")
                        if last_assistant_item:
                            print(f"Interrupting response with id: {last_assistant_item}")
                            await handle_speech_started_event()
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        async def handle_speech_started_event():
            """Handle interruption when the caller's speech starts."""
            nonlocal response_start_timestamp_twilio, last_assistant_item
            print("Handling speech started event.")
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                if SHOW_TIMING_MATH:
                    print(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

                if last_assistant_item:
                    if SHOW_TIMING_MATH:
                        print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time
                    }
                    await openai_ws.send(json.dumps(truncate_event))

                await websocket.send_json({
                    "event": "clear",
                    "streamSid": stream_sid
                })

                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_mark(connection, stream_sid):
            if stream_sid:
                mark_event = {
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "responsePart"}
                }
                await connection.send_json(mark_event)
                mark_queue.append('responsePart')

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

async def send_initial_conversation_item(openai_ws):
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with 'Hello! I am Kayako's help center assistant. How can I assist you today?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {
                "type": "server_vad",
                # "mode": "quality",  # Use quality mode for better noise filtering
                "threshold": 0.7,   # Higher threshold means less sensitive to background noise (default is 0.5)
                # "min_speech_duration_ms": 200,  # Minimum duration to consider something as speech
                # "min_silence_duration_ms": 400  # Minimum silence duration before considering speech ended
            },
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Uncomment the next line to have the AI speak first
    await send_initial_conversation_item(openai_ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)