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
from twilio.rest import Client

# Load environment variables from .env file
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
- If the user responds with no or any variant of that to the question "Is there anything else I can help you with?", say "Thank you for calling Kayako's help center. Have a great day!"
- If at any point the user indicates that they are done or want to end the call, say "Thank you for calling Kayako's help center. Have a great day!"
- Only use this predefined termination message "Thank you for calling Kayako's help center. Have a great day!" at any point where you would want to end the call.
- If the user's transcription is empty or incoherent, say "I'm sorry, I didn't catch that. Could you please repeat that again?"

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

# Voice configuration for text-to-speech
VOICE = 'alloy'

# Event types to log during conversation
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created'
]

# Flag to control display of timing calculations in logs
SHOW_TIMING_MATH = False

# Add these to your environment variables
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')

# Initialize Twilio client
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

# Initialize FastAPI application
app = FastAPI()

# Validate OpenAI API key presence
if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

# Initialize OpenAI client with API key
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize the RAG retriever
retriever = RAGRetriever(openai_client)

@app.get("/", response_class=JSONResponse)
async def index_page():
    """Return a simple health check message for the server."""
    return {"message": "Twilio Media Stream Server is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    # Initialize TwiML response object
    response = VoiceResponse()
    # <Say> punctuation to improve text-to-speech flow
    # response.say("Please wait while we connect your call to the A. I. voice assistant, powered by Twilio and the Open-A.I. Realtime API")
    # response.pause(length=1)
    # response.say("O.K. you can start talking!")
    
    # Get the host from the request URL
    host = request.url.hostname
    # Create a connection to the media stream
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    # Add the connection to the response
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    # Accept the WebSocket connection
    await websocket.accept()

    # Add conversation transcript variable
    conversation_transcript = []

    # Establish WebSocket connection to OpenAI's realtime API
    async with websockets.connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17',
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    ) as openai_ws:
        # Initialize the conversation session
        await initialize_session(openai_ws)

        # Initialize connection state variables
        stream_sid = None  # Unique identifier for the stream
        call_sid = None  # Add this variable
        latest_media_timestamp = 0  # Track the most recent media timestamp
        last_assistant_item = None  # Track the last assistant response
        mark_queue = []  # Queue for tracking response parts
        response_start_timestamp_twilio = None  # Track when responses start
        
        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, latest_media_timestamp, call_sid  # Add call_sid to nonlocal
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
                        call_sid = data['start']['callSid']
                        print(f"Incoming stream has started {stream_sid} and call_sid {call_sid}")
                        response_start_timestamp_twilio = None
                        latest_media_timestamp = 0
                        last_assistant_item = None
                    elif data['event'] == 'mark':
                        if mark_queue:
                            mark_queue.pop(0)
                    elif data['event'] == 'stop':
                        print("\n=== Full Conversation Transcript ===")
                        for entry in conversation_transcript:
                            print(f"{entry['role']}: {entry['text']}")
                        print("================================\n")
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

                    print(f"Received response: {response.get('type')}")
                    # if response.get('type') != 'response.audio.delta':
                    #     # print(f"Received response: {response}")

                    if response.get('type') == 'conversation.item.input_audio_transcription.completed':
                        print(f"Received transcription: {response}")
                        transcription_text = response.get('transcript')
                        if transcription_text:
                            conversation_transcript.append({
                                'role': 'User',
                                'text': transcription_text
                            })
                            print("User transcription:", transcription_text)
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

                    # if response['type'] in LOG_EVENT_TYPES:
                    #     # print(f"Received event: {response['type']}", response)
                    #
                    # if response.get('type') == 'response.content.done':
                    #     # print(f"Response content done: {response}")
                    #
                    # elif response.get('type') == 'response.done':
                    #     # print(f"Response done: {response}")
                    
                    if response.get('type') == 'response.audio_transcript.done':
                        transcript = response.get('transcript')
                        if "Thank you for calling Kayako's help center. Have a great day!" in transcript:
                            print("Termination command detected. Ending call.")
                            await asyncio.sleep(5)
                            await end_call(call_sid)
                            # await websocket.close()
                            print("WebSocket closed")
                            if openai_ws.open:
                                await openai_ws.close()
                            return
                    #
                    # if response.get("type") == "conversation.item.created":
                    #     # print(f"Received conversation item created: {response}")

                    # Ensure we have a function call event.
                    # response.function_call_arguments.done
                    if response.get("type") == "response.function_call_arguments.done":
                        print(f"Received function call arguments: {response}")

                    if response.get("type") == "conversation.item.created" and response["item"].get("type") == "function_call":
                        func_name = response["item"].get("name")
                        params = response["item"].get("parameters", {})
                        call_id = response["item"].get("call_id")
                        result = None
                        print(f"Received function call: {func_name} with parameters: {params} and call_id: {call_id}")
                        # print(f"Weather Conversation item created response: {response}")

                        # Inspect the function name and call the corresponding function.
                        if func_name == "get_weather":
                            location = params.get("location")
                            units = params.get("units")
                            result = get_weather(location, units)
                        else:
                            result = {"error": "Unknown function"}

                        # Prepare a function call output event to send back to OpenAI.
                        response_event = {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": call_id,  # This associates the output with the original call.
                                "output": json.dumps(result)
                            }
                        }

                        # Send the response event back through your WebSocket.
                        await openai_ws.send(json.dumps(response_event))

                        # Trigger AI to generate a response
                        trigger_response = {
                            "type": "response.create"
                        }
                        await openai_ws.send(json.dumps(trigger_response))

                    # Check for termination condition in the AI's response
                    if response.get('type') == 'response.audio.delta' and 'delta' in response:
                        # Decode the audio delta to check for termination command
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
    # Create the initial greeting message
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
    # Send the greeting message
    await openai_ws.send(json.dumps(initial_conversation_item))
    # Trigger response generation
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
    # Configure session parameters including VAD settings
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {
                "type": "server_vad",
                # "mode": "quality",  # Use quality mode for better noise filtering
                "threshold": 0.7,   # Higher threshold means less sensitive to background noise (default is 0.5)
                # "min_speech_duration_ms": 200,  # Minimum duration to consider something as speech
                # "min_silence_duration_ms": 500,  # Wait longer before considering speech as ended
                # "speech_pad_ms": 400  # Add padding to avoid cutting off speech too early
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
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Fetches the current weather for a given location. Use this function when the user asks for weather updates.",
                    "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                        "type": "string",
                        "description": "The city or region to retrieve the weather for."
                        },
                        "units": {
                        "type": "string",
                        "enum": ["metric", "imperial"],
                        "description": "Unit system for temperature (metric for Celsius, imperial for Fahrenheit)."
                        }
                    },
                    "required": ["location", "units"]
                    }
                }
            ]
        }
    }
    # Log and send session configuration
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Initialize the conversation with a greeting
    await send_initial_conversation_item(openai_ws)

def get_weather(location, units):
    # Your custom logic to fetch weather information for the given location.
    # For demonstration, we'll return a dummy result.
    print(f"Fetching weather for {location} in {units} units")
    return {
        "temperature": 22,
        "description": "Partly cloudy",
        "units": units,
        "location": location
    }

async def end_call(call_sid):
    """End a Twilio call using the REST API."""
    try:
        twilio_client.calls(call_sid).update(status='completed')
        print(f"Successfully ended call {call_sid}")
    except Exception as e:
        print(f"Error ending call: {e}")

# Start the server if running as main script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)