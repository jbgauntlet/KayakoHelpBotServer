import asyncio
import base64
import json
import os

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from openai import OpenAI
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect
from kayako_functions import create_call_summary_ticket
from realtime_functions import (
    get_article_search_results,
    connect_customer_to_agent,
    create_custom_support_ticket,
)
from utility_functions import end_call

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

# System message for the help bot
SYSTEM_MESSAGE = f"""
You are a helpful and bubbly customer support agent for Kayako who loves to chat to customers. 
You specialize in providing actionable answers based on Kayako's documentation.

When responding:
1. Be concise and clear - suitable for phone conversation
2. Use a natural, conversational tone
3. Focus on providing specific, actionable steps
4. If the documentation contains relevant information, even if partial, use it to help the user
6. Always use the get_article_search_results function on the user's problem or question, to search for articles in Kayako and see if any relevant information is available.

Evaluate the provided documentation:
- Always use the get_article_search_results function on the user's problem or question, to search for articles in Kayako and see if any relevant information is available.
- If it contains ANY relevant information to answer the question, use it to provide specific guidance
- If it's completely unrelated or doesn't help answer the question at all, respond with:
"I'm sorry, but I'm not sure I can help you with that."
- After any response to a user query that is not a question, ask "Is there anything else I can help you with?"
- If the user responds with no or any variant of that to the question "Is there anything else I can help you with?", say "Thank you for calling Kayako's help center. Have a great day!"
- If at any point the user indicates that they are done or want to end the call, say "Thank you for calling Kayako's help center. Have a great day!"
- Only use this predefined termination message "Thank you for calling Kayako's help center. Have a great day!" at any point where you would want to end the call.
- If the user's transcription is empty or incoherent, say "I'm sorry, I didn't catch that. Could you please repeat that again?"
- If the user asks to speak to a agent or accepts the offer to speak to a agent, accept the request and use the connect_customer_to_agent function to connect the customer to an agent.
- If the user's question is related to Kayako but you cannot answer it based on the documentation, offer to connect them to a human agent.
- For any tool call, make sure to ask the user for parameters if needed. And if added clarity is needed, ask for more information.
- If the transcript is ever empty, prompt the user again with "I'm sorry, I didn't catch that. Could you please repeat that again?"

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
VOICE = 'sage'

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
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio, end_call_triggered, call_recording_file, call_sid
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
                            await asyncio.sleep(4) # Wait for 4 seconds to ensure the message is not interrupted
                            await end_call(call_sid, twilio_client)
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
                            
                            # DO NOT REMOVE THIS COMMENTED CODE
                            if response.get("type") == "conversation.item.created" and response["item"].get("type") == "function_call":
                                func_name = response["item"].get("name")
                                params = response["item"].get("parameters", {})
                                call_id = response["item"].get("call_id")
                                result = None
                                print(f"Received function call: {func_name} with parameters: {params} and call_id: {call_id}")

                            # Inspect the function name and call the corresponding function.
                            if func_name == "connect_customer_to_agent":
                                name = params.get("name", "")
                                email = params.get("email", "")
                                result = connect_customer_to_agent(name, email, call_sid, twilio_client)
                            elif func_name == "create_custom_support_ticket":
                                name = params.get("name", "")
                                email = params.get("email", "")
                                subject = params.get("subject", "")
                                description = params.get("description", "")
                                priority = params.get("priority", 1)
                                result = create_custom_support_ticket(name, email, subject, description, priority, conversation_transcript)
                            elif func_name == "get_article_search_results":
                                query = params.get("query", "")
                                result = get_article_search_results(query)
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
                "silence_duration_ms": 1000,
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
                    "name": "connect_customer_to_agent",
                    "description": """Queries the customer for the necessary details to connect them to an agent. 
                                      Determines whether an agent is available to take the call. 
                                      If no agent is available, offer to create a support ticket. 
                                      If the user accepts, use the create_custom_support_ticket function.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the customer."
                            },
                            "email": {
                                "type": "string",
                                "description": "The email of the customer."
                            }
                        },
                        "required": ["name", "email"]
                    }
                },
                {
                    "type": "function",
                    "name": "create_custom_support_ticket",
                    "description": """Creates a custom support ticket for the customer.
                                      You come up with the subject and description of the ticket based on the what the user has said previously.
                                      If the user has not specified any details, and only asked for an agent then Prompt the user to share what their issue is so that you can create a ticket.
                                      Based on the user's issue determine the priority of the ticket and set the priority to 1, 2, 3 or 4 where 4 is the highest priority and 1 is the lowest.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the customer."
                            },
                            "email": {
                                "type": "string",
                                "description": "The email of the customer."
                            },
                            "subject": {
                                "type": "string",
                                "description": "The subject of the support ticket."
                            },
                            "description": {
                                "type": "string",
                                "description": "Summary of the user's request."
                            },
                            "priority": {
                                "type": "integer",
                                "description": "The priority of the support ticket."
                            }
                        },
                        "required": ["name", "email", "subject", "description", "priority"]
                    }
                },
                {
                    "type": "function",
                    "name": "get_article_search_results",
                    "description": """Searches for articles in Kayako and returns the results.
                                      Should be called on every user query if there is relevant information.""",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The user's problem or question to search for."
                            }
                        },
                        "required": ["query"]
                    }
                },
            ]
        }
    }
    # Log and send session configuration
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Initialize the conversation with a greeting
    await send_initial_conversation_item(openai_ws)

@app.api_route("/redirect-to-agent", methods=["POST"])
async def redirect_to_agent(request: Request):
    """
    Handle call redirection to a human agent.
    
    This endpoint generates TwiML to smoothly transition the customer from the AI assistant
    to a human agent. It's called by Twilio when a call redirection is initiated.
    
    The endpoint performs the following:
    1. Creates a TwiML response with a transitional message for the customer
    2. Dials out to the agent's phone number (configured in environment variables)
    3. Returns the TwiML for Twilio to execute
    
    Args:
        request (Request): The FastAPI request object containing information about the call
        
    Returns:
        Response: A FastAPI response containing TwiML instructions for Twilio
        
    Note:
        The REDIRECT_PHONE_NUMBER must be properly configured in the environment 
        variables for the agent connection to succeed.
    """
    # Create TwiML response object for Twilio to execute
    response = VoiceResponse()
    
    # Add a message explaining the transfer to improve customer experience
    # This message is spoken to the caller while the transfer is being set up
    response.say("Transferring you to an available agent. Please hold.")
    
    # Retrieve the destination phone number from environment variables
    # This could be a direct line, call center number, or SIP address
    REDIRECT_PHONE_NUMBER = os.getenv('REDIRECT_PHONE_NUMBER')
    
    # Dial instruction initiates the outbound call to the agent
    # When answered, the customer's call will be connected to this new call
    response.dial(REDIRECT_PHONE_NUMBER)
    
    # Return the TwiML response with proper content type
    # Twilio requires XML format with specific content-type header
    return Response(content=str(response), media_type="text/xml")

# Start the server if running as main script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)