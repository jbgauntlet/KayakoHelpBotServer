import os
import json
import base64
import asyncio
import websockets
import requests
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

# Load knowledge base from the knowledge_base.json file
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

# Kayako API credentials
KAYAKO_API_USERNAME = os.getenv('KAYAKO_API_USERNAME')
KAYAKO_API_PASSWORD = os.getenv('KAYAKO_API_PASSWORD')
KAYAKO_API_URL = os.getenv('KAYAKO_API_URL')

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

# Voice configuration for text-to-speech
VOICE = 'alloy'

# Show interruption timing math
SHOW_TIMING_MATH = False

# Fetch Twilio environment variables
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

# Health check endpoint, root route
@app.get("/", response_class=JSONResponse)
async def index_page():
    """Return a simple health check message for the server."""
    return {"message": "Twilio Media Stream Server is running!"}

# Handle incoming call and return TwiML response to connect to Media Stream, entry point for Twilio
@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    # Initialize TwiML response object
    response = VoiceResponse()   
    # Get the host from the request URL
    host = request.url.hostname
    # Create a connection to the media stream
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    # Add the connection to the response
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

# Handle WebSocket connections between Twilio and OpenAI, entry point for OpenAI and handles the conversation
@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI."""
    print("Client connected")
    # Accept the WebSocket connection
    await websocket.accept()

    # Add conversation transcript variable
    conversation_transcript = []

    # Initialize call recording file
    call_recording_file = None
    
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
        end_call_triggered = False  # Track if the end call function has been triggered
        
        # Receive audio data from Twilio and send it to the OpenAI Realtime API
        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, latest_media_timestamp, call_sid, end_call_triggered, call_recording_file
            try:
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data['event'] == 'media' and openai_ws.open:
                        # Get the audio payload
                        audio_payload = data['media']['payload']
                        decoded_audio = base64.b64decode(audio_payload)
                        
                        # Open recording file if not already open
                        if call_sid and not call_recording_file:
                            call_recording_file = open(f'call_{call_sid}.ulaw', 'wb')
                        
                        # Write user audio with a marker
                        if call_recording_file:
                            # You could add a marker here if needed
                            call_recording_file.write(decoded_audio)
                        
                        # Continue with regular processing
                        latest_media_timestamp = int(data['media']['timestamp'])
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": audio_payload
                        }))
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
                        # Close the recording file if open
                        if call_recording_file:
                            call_recording_file.close()
                            call_recording_file = None
                        
                        # Create ticket with the call recording
                        await create_call_summary_ticket(conversation_transcript, call_sid)
                        
                        if openai_ws.open:
                            await openai_ws.close()
                        await websocket.close()
                        return
            except WebSocketDisconnect:
                print("Client disconnected.")
                if openai_ws.open:
                    await openai_ws.close()
            except Exception as e:
                print(f"Error in receive_from_twilio: {e}")
                if call_recording_file:
                    call_recording_file.close()

        # Send events to Twilio
        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio, end_call_triggered, call_recording_file
            try:
                async for openai_message in openai_ws:
                    response = json.loads(openai_message)

                    # Log the response type
                    print(f"Received response: {response.get('type')}")

                    # If the response is a user transcription, add it to the conversation transcript
                    if response.get('type') == 'conversation.item.input_audio_transcription.completed':
                        print(f"Received transcription: {response}")
                        transcription_text = response.get('transcript')
                        if transcription_text:
                            conversation_transcript.append({
                                'role': 'User',
                                'text': transcription_text
                            })
                            print("User transcription:", transcription_text)

                    # If the response is a help bot transcription, add it to the conversation transcript
                    if response.get('type') == 'response.audio_transcript.done':
                        transcription_text = response.get('transcript')
                        if transcription_text:
                            conversation_transcript.append({
                                'role': 'Help Bot',
                                'text': transcription_text
                            })
                            print("User transcription:", transcription_text)

                        if "Thank you for calling Kayako's help center. Have a great day!" in transcription_text:
                            print("Termination command detected. Marking call as ready to end.")
                            end_call_triggered = True
                            await asyncio.sleep(5)
                            await end_call(call_sid)
                            return

                    # Ensure we have a function call event with all the necessary arguments
                    if response.get("type") == "response.function_call_arguments.done":
                        func_name = response.get("name")
                        args_str = response.get("arguments", "{}")
                        call_id = response.get("call_id")
                        result = None
                        print(f"Received function call: {func_name} with call_id: {call_id}")
                        print(f"Received function call arguments: {response}")

                        try:
                            # Parse the arguments string into a dictionary
                            if isinstance(args_str, str):
                                params = json.loads(args_str)
                            else:
                                params = args_str
                            print(params)
                            
                            # # DO NOT REMOVE THIS COMMENTED CODE
                            # if response.get("type") == "conversation.item.created" and response["item"].get("type") == "function_call":
                            #     func_name = response["item"].get("name")
                            #     params = response["item"].get("parameters", {})
                            #     call_id = response["item"].get("call_id")
                            #     result = None
                            #     print(f"Received function call: {func_name} with parameters: {params} and call_id: {call_id}")
                            #     # print(f"Weather Conversation item created response: {response}")

                            # Inspect the function name and call the corresponding function.
                            if func_name == "get_weather":
                                location = params.get("location", "")
                                units = params.get("units", "metric")
                                result = get_weather(location, units)
                            else:
                                result = {"error": "Unknown function"}
                        except json.JSONDecodeError as e:
                            print(f"Error parsing function arguments: {e}, args: {args_str}")
                            result = {"error": "Invalid arguments format"}
                        except Exception as e:
                            print(f"Error handling function call: {e}")
                            result = {"error": str(e)}

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

                        # Trigger response generation
                        await openai_ws.send(json.dumps({"type": "response.create"}))

                    if response.get('type') == 'response.audio.delta' and 'delta' in response:
                        audio_data = base64.b64decode(response['delta'])
                        
                        # Write bot audio to the same file
                        if call_recording_file:
                            # You could add a marker here if needed
                            call_recording_file.write(audio_data)
                        
                        # Continue with regular processing
                        audio_payload = base64.b64encode(audio_data).decode('utf-8')
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
                    # Only trigger interruptions if the call is not ready to end
                    if response.get('type') == 'input_audio_buffer.speech_started' and not end_call_triggered:
                        print("Speech started detected.")
                        if last_assistant_item:
                            print(f"Interrupting response with id: {last_assistant_item}")
                            await handle_speech_started_event()
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")

        # Handle interruption when the caller's speech starts
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

        # Send a mark event to Twilio to indicate the start of a new response
        async def send_mark(connection, stream_sid):
            if stream_sid:
                mark_event = {
                    "event": "mark",
                    "streamSid": stream_sid,
                    "mark": {"name": "responsePart"}
                }
                await connection.send_json(mark_event)
                mark_queue.append('responsePart')

        # Run the receive and send functions concurrently
        await asyncio.gather(receive_from_twilio(), send_to_twilio())

# Send initial conversation item
async def send_initial_conversation_item(openai_ws):
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

# Initialize the session with OpenAI
async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
    # Configure session parameters including VAD settings
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.8,   # Higher threshold means less sensitive to background noise (default is 0.8)
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

# Get weather information for a given location
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

# End a Twilio call using the REST API
async def end_call(call_sid):
    """End a Twilio call using the REST API."""
    try:
        twilio_client.calls(call_sid).update(status='completed')
        print(f"Successfully ended call {call_sid}")
    except Exception as e:
        print(f"Error ending call: {e}")

# Create a call summary ticket
async def create_call_summary_ticket(transcript, call_sid=None):
    """Create a ticket with call transcript and audio attachment."""
    # Format transcript into a string with new line separation
    print("\n=== Full Conversation Transcript ===")
    formatted_transcript = ""
    for entry in transcript:
        print(f"{entry['role']}: {entry['text']}")
        formatted_transcript += f"{entry['role']}: {entry['text']}\n"
    print("================================\n")

    # Prepare wav file path if we have a call_sid
    wav_path = None
    if call_sid:
        try:
            # Check if the raw audio file exists
            ulaw_path = f'call_{call_sid}.ulaw'
            wav_path = f'call_{call_sid}.wav'
            
            if os.path.exists(ulaw_path):
                print(f"Converting audio for call {call_sid}...")
                
                try:
                    # Python-based audio conversion instead of ffmpeg
                    import wave
                    import numpy as np

                    # Function to convert u-law to linear PCM
                    def ulaw2linear(u_law_data):
                        # u-law decoding table
                        u = 255
                        u_law_data = np.frombuffer(u_law_data, dtype=np.uint8)
                        # Convert to signed integers
                        sign = np.where(u_law_data < 128, 1, -1)
                        # Remove sign bit
                        u_law_data = np.where(u_law_data < 128, u_law_data, 255 - u_law_data)
                        # Decode using u-law formula
                        linear_data = sign * (((u_law_data / u) ** (1/1.5)) * (2**15 - 1))
                        return linear_data.astype(np.int16)
                    
                    # Read u-law data
                    with open(ulaw_path, 'rb') as f:
                        u_law_data = f.read()
                    
                    # Convert to linear PCM
                    linear_data = ulaw2linear(u_law_data)
                    
                    # Create WAV file
                    with wave.open(wav_path, 'wb') as wav_file:
                        wav_file.setnchannels(1)  # Mono
                        wav_file.setsampwidth(2)  # 2 bytes (16 bits) per sample
                        wav_file.setframerate(8000)  # 8 kHz sampling rate for u-law
                        wav_file.writeframes(linear_data.tobytes())
                    
                    print(f"Successfully converted audio to {wav_path}")
                except Exception as e:
                    print(f"Audio conversion failed: {e}")
                    # Fallback message
                    print("Consider installing ffmpeg for better audio conversion support")
                    wav_path = None
            else:
                print(f"No audio file found at {ulaw_path}")
                wav_path = None
        except Exception as e:
            print(f"Error processing audio: {e}")
            wav_path = None

    # API endpoint with query parameters
    url = f"{KAYAKO_API_URL}"

    # Create data payload according to API documentation
    data = {
        "subject": f"[GAUNTLET AI TEST] Call Summary - Call {call_sid}",
        "contents": f"""<div style="font-family: Arial, sans-serif; line-height: 1.6;">
            <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
                <strong>📋 SUBJECT</strong><br>
                <p style="margin-top: 10px; margin-bottom: 0; font-size: 16px;">Kayako Help Center Call</p>
            </div>

            <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
                <strong>📝 SUMMARY</strong><br>
                <p style="margin-top: 10px; margin-bottom: 0;">Call assistance with Kayako help center.</p>
            </div>

            <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
                <strong>📞 CALL TRANSCRIPT</strong><br>
                <pre style="background: #f5f5f5; padding: 15px; border-radius: 4px; margin: 10px 0; white-space: pre-wrap; font-family: monospace;">
{formatted_transcript}
                </pre>
            </div>
        </div>""",
        "channel": "MAIL",
        "channel_id": "1",
        "tags": "gauntlet-ai",
        "type_id": "7",
        "status_id": "1",
        "priority_id": "1",
        "assigned_agent_id": "309",
        "assigned_team_id": "1",
        "requester_id": "309",
        "form_id": "1",
    }

    try:
        # Prepare the multipart/form-data request
        files = {}
        
        # Authenticate to Kayako
        auth = (KAYAKO_API_USERNAME, KAYAKO_API_PASSWORD)

        # Add the audio file to the request if it exists
        if wav_path and os.path.exists(wav_path):
            with open(wav_path, 'rb') as f:
                files['attachment'] = (os.path.basename(wav_path), f, 'audio/wav')
                # Make the POST request
                
                print(f"Creating case with Kayako API at {url}")
                response = requests.post(
                    url,
                    auth=auth,
                    data=data,
                    files=files
                )
                
                # Check if request was successful
                if response.status_code in [200, 201]:
                    result = response.json()
                    print(f"Successfully created ticket: {result.get('data', {}).get('id')}")
                    return result
                else:
                    print(f"Error creating ticket: {response.status_code} - {response.text}")
                    return None
            print(f"Attaching audio file: {wav_path}")
        else:
            print("No audio file found")

            # Make the POST request
            print(f"Creating case with Kayako API at {url}")
            response = requests.post(
                url,
                auth=auth,
                data=data,
                files=files
            )
            
            # Check if request was successful
            if response.status_code in [200, 201]:
                result = response.json()
                print(f"Successfully created ticket: {result.get('data', {}).get('id')}")
                return result
            else:
                print(f"Error creating ticket: {response.status_code} - {response.text}")
                return None
    
    except Exception as e:
        print(f"Error creating ticket: {e}")
        return None

# Start the server if running as main script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)