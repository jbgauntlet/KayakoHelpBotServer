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

# Initialize clients
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize RAG
rag = RAGRetriever(openai_client)
rag.load_articles(['sample-help.json', 'sample-help-2.json'])
rag.process_articles()

# Conversation context storage
conversation_context: Dict[str, Dict] = {}

# Constants for conversation control
MAX_INVALID_ATTEMPTS = 3    # Maximum number of off-topic or unclear responses
MAX_SILENCE_ATTEMPTS = 3    # Maximum number of silent/no-input attempts
MAX_TOTAL_TURNS = 50       # Maximum number of conversation turns (much higher as these are valid turns)

IntentType = Literal["END_CALL", "CONTINUE", "CONNECT_HUMAN", "UNCLEAR", "OFF_TOPIC"]

def classify_intent(response: str, context: str) -> IntentType:
    """Use LLM to classify user's intent"""
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
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Thank you for calling the Kayako Help Center today. How may I assist you?</Say>
    <Gather input="speech" action="/handle-input" method="POST" speechTimeout="1" language="en-US"/>
</Response>"""
    return Response(content=twiml, media_type="application/xml")

@app.post("/handle-input")
async def handle_input(request: Request):
    form_data = await request.form()
    speech_result = form_data.get("SpeechResult", "")
    call_sid = form_data.get("CallSid", "")
    logger.info(f"Received speech: {speech_result}")

    # Get or initialize conversation context
    context = conversation_context.get(call_sid, {
        "last_query": None,
        "current_question": None,
        "invalid_attempts": 0,    # Counter for off-topic/unclear responses
        "silence_attempts": 0,    # Counter for no-input responses
        "total_turns": 0         # Counter for total conversation turns
    })

    if not speech_result:
        context["silence_attempts"] = context.get("silence_attempts", 0) + 1
        
        if context["silence_attempts"] >= MAX_SILENCE_ATTEMPTS:
            twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I haven't been able to hear you clearly. Please call back when you're ready to discuss your Kayako-related questions. Have a great day!</Say>
    <Hangup/>
</Response>"""
            conversation_context.pop(call_sid, None)
            return Response(content=twiml, media_type="application/xml")
            
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I'm having trouble hearing you. Could you please speak your question clearly?</Say>
    <Gather input="speech" action="/handle-input" method="POST" speechTimeout="1" language="en-US"/>
</Response>"""
        conversation_context[call_sid] = context
        return Response(content=twiml, media_type="application/xml")

    # Reset silence counter when we get a response
    context["silence_attempts"] = 0
    
    # Increment total turns only for valid speech input
    context["total_turns"] = context.get("total_turns", 0) + 1
    
    # Check if we've exceeded maximum turns
    if context["total_turns"] > MAX_TOTAL_TURNS:
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I notice this has been a long conversation. To ensure quality service, let me connect you with a human agent who can provide more comprehensive assistance.</Say>
    <Hangup/>
</Response>"""
        conversation_context.pop(call_sid, None)
        return Response(content=twiml, media_type="application/xml")

    # Classify user's intent based on context
    intent = classify_intent(
        speech_result,
        f"Current question: {context.get('current_question', 'How may I assist you?')}"
    )
    
    # Handle different intents
    if context.get("current_question") == "Do you need help with anything else?":
        if intent == "END_CALL":
            twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Thank you for calling Kayako Help Center. Have a great day!</Say>
    <Hangup/>
</Response>"""
            conversation_context.pop(call_sid, None)
            return Response(content=twiml, media_type="application/xml")
    
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
    
    # Handle invalid responses (OFF_TOPIC or UNCLEAR)
    if intent in ["OFF_TOPIC", "UNCLEAR"]:
        context["invalid_attempts"] = context.get("invalid_attempts", 0) + 1
        
        # Check if we've exceeded maximum invalid attempts
        if context["invalid_attempts"] >= MAX_INVALID_ATTEMPTS:
            twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I apologize, but I'm having trouble understanding your needs. Let me connect you with a human agent who can better assist you.</Say>
    <Hangup/>
</Response>"""
            conversation_context.pop(call_sid, None)
            return Response(content=twiml, media_type="application/xml")
        
        if intent == "OFF_TOPIC":
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I can only assist with questions related to Kayako and its services. This is attempt {context["invalid_attempts"]} of {MAX_INVALID_ATTEMPTS}. Please ask a Kayako-related question.</Say>
    <Gather input="speech" action="/handle-input" method="POST" speechTimeout="1" language="en-US"/>
</Response>"""
        else:  # UNCLEAR
            twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I'm not sure I understood. This is attempt {context["invalid_attempts"]} of {MAX_INVALID_ATTEMPTS}. Could you please rephrase your question about Kayako?</Say>
    <Gather input="speech" action="/handle-input" method="POST" speechTimeout="1" language="en-US"/>
</Response>"""
        
        context["current_question"] = "Please ask a Kayako-related question"
        conversation_context[call_sid] = context
        return Response(content=twiml, media_type="application/xml")
    
    # Reset invalid attempts counter for valid responses
    context["invalid_attempts"] = 0
    
    # Store the current query
    context["last_query"] = speech_result
    
    # Get relevant knowledge
    knowledge = retrieve_data(speech_result)
    # Generate response
    llm_response = format_response(knowledge, speech_result)
    logger.info(f"AI Response: {llm_response}")

    # Set the current question for context in next interaction
    current_question = "Do you need help with anything else?"
    if "connect you with a human" in llm_response:
        current_question = "Would you like me to connect you with a human support agent?"
    
    context["current_question"] = current_question
    conversation_context[call_sid] = context

    # Respond and continue listening
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>{llm_response}</Say>
    <Gather input="speech" action="/handle-input" method="POST" speechTimeout="1" language="en-US">
        <Say>{current_question}</Say>
    </Gather>
</Response>"""

    return Response(content=twiml, media_type="application/xml")

# RAG Knowledge Retrieval
def retrieve_data(query):
    # Get relevant articles
    results = rag.retrieve(query, top_k=2)
    
    # Format the context
    context = []
    for result in results:
        context.append(f"Article: {result['title']}\n{result['text'][:500]}...")
        
    return "\n\n".join(context) if context else "NO_ARTICLES_FOUND"

# OpenAI LLM Response Formatting
def format_response(knowledge: str, query: str):
    max_retries = 3
    base_delay = 1  # seconds
    
    # If no articles found
    if knowledge == "NO_ARTICLES_FOUND":
        return "I apologize, but I don't have any information about that in my knowledge base. Would you like me to connect you with a human support agent?"
    
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(context=knowledge)},
                    {"role": "user", "content": f"Question: {query}\n\nAvailable documentation:\n{knowledge}"}
                ]
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            if attempt == max_retries - 1:
                logger.error(f"OpenAI rate limit exceeded after {max_retries} attempts")
                return "I apologize, but I'm currently experiencing high demand. Please try again in a moment."
            delay = base_delay * (2 ** attempt)  # exponential backoff
            time.sleep(delay)
        except Exception as e:
            logger.error(f"Error in OpenAI API call: {e}")
            return "I apologize, but I'm having trouble processing your request right now."
