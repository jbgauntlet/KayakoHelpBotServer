# FastAPI server for Kayako Help Center voice bot
# Handles voice calls using Twilio and provides responses using OpenAI and RAG

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from openai import OpenAI, RateLimitError
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException
from dotenv import load_dotenv
import json
import os
import logging
import time
from typing import Dict, Literal

from prompts import SYSTEM_PROMPT, INTENT_CLASSIFICATION_PROMPT
from rag import RAGRetriever

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# API Keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")

# Initialize external service clients
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize RAG system with help articles
rag = RAGRetriever(openai_client)
rag.load_articles(['sample-help.json', 'sample-help-2.json'])
rag.process_articles()

# Store conversation state for each active call
conversation_context: Dict[str, Dict] = {}

# Constants for conversation control and limits
MAX_INVALID_ATTEMPTS = 3    # Maximum number of off-topic or unclear responses
MAX_SILENCE_ATTEMPTS = 3    # Maximum number of silent/no-input attempts
MAX_TOTAL_TURNS = 50       # Maximum number of conversation turns (much higher as these are valid turns)

# Valid intents for user responses
IntentType = Literal["END_CALL", "CONTINUE", "CONNECT_HUMAN", "UNCLEAR", "OFF_TOPIC"]

def classify_intent(response: str, context: str) -> IntentType:
    # Use LLM to classify user's intent
    try:
        classification = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": INTENT_CLASSIFICATION_PROMPT},
                {"role": "user", "content": f"Context: {context}\nUser's response: {response}"}
            ],
            temperature=0,  # Use low temperature for consistent classification
            max_tokens=10   # We only need a single word response
        )
        intent = classification.choices[0].message.content.strip()
        logger.debug(f"Classified intent: {intent} for response: {response}")
        if intent in ["END_CALL", "CONTINUE", "CONNECT_HUMAN", "UNCLEAR", "OFF_TOPIC"]:
            return intent
        return "UNCLEAR"
    except Exception as e:
        logger.error(f"Error in intent classification: {e}")
        return "UNCLEAR"

@app.post("/voice")
async def handle_call():
    # Initial endpoint for new calls. Provides welcome message and starts conversation.
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Thank you for calling the Kayako Help Center today. How may I assist you?</Say>
    <Gather input="speech" action="/handle-input" method="POST" speechTimeout="1" language="en-US"/>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.post("/handle-input")
async def handle_input(request: Request):
    # Main conversation handler. Processes user input and generates appropriate responses.
    
    # Extract speech and call data from the request
    form_data = await request.form()
    speech_result = form_data.get("SpeechResult", "")
    call_sid = form_data.get("CallSid", "")
    logger.info(f"Received speech: {speech_result}")

    # Initialize or retrieve conversation context for this call
    context = conversation_context.get(call_sid, {
        "last_query": None,          # Store the most recent user question
        "current_question": None,     # Track what we last asked the user
        "invalid_attempts": 0,        # Count off-topic/unclear responses
        "silence_attempts": 0,        # Count no-input responses
        "total_turns": 0             # Track total conversation length
    })

    # Handle silence or no speech input
    if not speech_result:
        context["silence_attempts"] = context.get("silence_attempts", 0) + 1
        
        # If too many silent attempts, end call politely
        if context["silence_attempts"] >= MAX_SILENCE_ATTEMPTS:
            twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I'm having trouble with processing your response. Please try calling back, and we'll be happy to help you with your Kayako questions. Have a great day!</Say>
    <Hangup/>
</Response>"""
            conversation_context.pop(call_sid, None)  # Clean up context
            return Response(content=twiml, media_type="application/xml")
            
        # Otherwise, prompt for retry
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I didn't catch that. Could you please try your question again?</Say>
    <Gather input="speech" action="/handle-input" method="POST" speechTimeout="1" language="en-US"/>
</Response>"""
        conversation_context[call_sid] = context
        return Response(content=twiml, media_type="application/xml")

    # Reset silence counter on valid response
    context["silence_attempts"] = 0
    
    # Track conversation length
    context["total_turns"] = context.get("total_turns", 0) + 1
    
    # End conversation if it's getting too long
    if context["total_turns"] > MAX_TOTAL_TURNS:
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I notice this has been a long conversation. To ensure quality service, let me connect you with a human agent who can provide more comprehensive assistance.</Say>
    <Hangup/>
</Response>"""
        conversation_context.pop(call_sid, None)
        return Response(content=twiml, media_type="application/xml")

    # Determine user's intent based on their response
    intent = classify_intent(
        speech_result,
        f"Current question: {context.get('current_question', 'How may I assist you?')}"
    )
    
    # Handle end of conversation
    if context.get("current_question") == "Do you need help with anything else?":
        if intent == "END_CALL":
            twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Thank you for calling Kayako Help Center. Have a great day!</Say>
    <Hangup/>
</Response>"""
            conversation_context.pop(call_sid, None)
            return Response(content=twiml, media_type="application/xml")
    
    # Handle human agent connection requests
    if context.get("current_question") == "Would you like me to connect you with a human support agent?":
        if intent == "CONNECT_HUMAN":
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I'll connect you with a human agent regarding your question about {context['last_query']}. Please hold.</Say>
    <Hangup/>
</Response>"""
            conversation_context.pop(call_sid, None)
            return Response(content=twiml, media_type="application/xml")
        elif intent == "END_CALL":
            twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Alright, what else can I help you with?</Say>
    <Gather input="speech" action="/handle-input" method="POST" speechTimeout="1" language="en-US"/>
</Response>"""
            context["current_question"] = "What else can I help you with?"
            conversation_context[call_sid] = context
            return Response(content=twiml, media_type="application/xml")
    
    # Handle unclear or off-topic responses
    if intent in ["OFF_TOPIC", "UNCLEAR"]:
        context["invalid_attempts"] = context.get("invalid_attempts", 0) + 1
        
        # Transfer to human after too many invalid attempts
        if context["invalid_attempts"] >= MAX_INVALID_ATTEMPTS:
            twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I apologize, but I'm having trouble understanding your needs. Let me connect you with a human agent who can better assist you.</Say>
    <Hangup/>
</Response>"""
            conversation_context.pop(call_sid, None)
            return Response(content=twiml, media_type="application/xml")
        
        # Handle off-topic vs unclear responses differently
        if intent == "OFF_TOPIC":
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I can only assist with questions related to Kayako and its services. Please ask a Kayako-related question.</Say>
    <Gather input="speech" action="/handle-input" method="POST" speechTimeout="1" language="en-US"/>
</Response>"""
        else:  # UNCLEAR
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I'm not sure I understood. Could you please rephrase your question about Kayako?</Say>
    <Gather input="speech" action="/handle-input" method="POST" speechTimeout="1" language="en-US"/>
</Response>"""
        
        context["current_question"] = "Please ask a Kayako-related question"
        conversation_context[call_sid] = context
        return Response(content=twiml, media_type="application/xml")
    
    # Reset invalid attempts counter for valid responses
    context["invalid_attempts"] = 0
    
    # Store the current query for context
    context["last_query"] = speech_result
    
    # Generate response using RAG and LLM
    knowledge = retrieve_data(speech_result)
    llm_response = format_response(knowledge, speech_result)
    logger.info(f"AI Response: {llm_response}")

    # Determine follow-up question based on response
    current_question = "Do you need help with anything else?"
    if "connect you with a human" in llm_response:
        current_question = "Would you like me to connect you with a human support agent?"
    
    context["current_question"] = current_question
    conversation_context[call_sid] = context

    # Deliver response and gather next input
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>{llm_response}</Say>
    <Gather input="speech" action="/handle-input" method="POST" speechTimeout="1" language="en-US">
        <Say>{current_question}</Say>
    </Gather>
</Response>"""

    return Response(content=twiml, media_type="application/xml")

def retrieve_data(query: str) -> str:
    # Retrieve relevant articles from the knowledge base using RAG
    #
    # Args:
    #     query: User's question or request
    #        
    # Returns:
    #     Formatted string containing relevant article content
    
    # Get relevant articles
    results = rag.retrieve(query, top_k=2)
    
    # Format the context
    context = []
    for result in results:
        # Take more content and format it better
        content = result['text'][:1500]  # Increased from 500 to 1500 characters
        # Clean up the content
        content = content.replace("#", "").replace("*", "").strip()
        context.append(f"Title: {result['title']}\n\n{content}")
        logger.debug(f"Retrieved article: {result['title']}\nContent preview: {content[:200]}...")
        
    return "\n\n---\n\n".join(context)

def format_response(knowledge: str, query: str) -> str:
    # Generate a natural language response using OpenAI based on retrieved knowledge
    #
    # Args:
    #     knowledge: Retrieved article content
    #     query: User's original question
    #        
    # Returns:
    #     Natural language response suitable for text-to-speech
    
    max_retries = 3
    base_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # First, analyze if the retrieved content is relevant
            analysis = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing whether documentation is relevant to a user's question. Respond with 'RELEVANT' or 'NOT_RELEVANT' followed by a brief reason."},
                    {"role": "user", "content": f"Question: {query}\n\nDocumentation:\n{knowledge}"}
                ],
                temperature=0
            )
            relevance = analysis.choices[0].message.content
            logger.debug(f"Relevance analysis: {relevance}")
            
            # Handle cases where no relevant information is found
            if relevance.startswith("NOT_RELEVANT"):
                return "I'm sorry, but I'm not familiar with that aspect of Kayako."
            
            # Generate natural response from relevant content
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"User's question: {query}\n\nRelevant documentation:\n{knowledge}"}
                ]
            )
            return response.choices[0].message.content
            
        except RateLimitError as e:
            # Handle rate limits with exponential backoff
            if attempt == max_retries - 1:
                logger.error(f"OpenAI rate limit exceeded after {max_retries} attempts")
                return "I apologize, but I'm currently experiencing high demand. Please try again in a moment."
            delay = base_delay * (2 ** attempt)  # exponential backoff
            time.sleep(delay)
        except Exception as e:
            # Handle other API errors
            logger.error(f"Error in OpenAI API call: {e}")
            return "I apologize, but I'm having trouble processing your request right now."
