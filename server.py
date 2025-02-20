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

from prompts import SYSTEM_PROMPT
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
    logger.info(f"Received speech: {speech_result}")

    if speech_result:
        # Get relevant knowledge
        knowledge = retrieve_data(speech_result)
        # Generate response
        llm_response = format_response(knowledge)
        logger.info(f"AI Response: {llm_response}")

        # Respond and continue listening
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>{llm_response}</Say>
    <Gather input="speech" action="/handle-input" method="POST" speechTimeout="1" language="en-US">
        <Say>Is there anything else I can help you with?</Say>
    </Gather>
</Response>"""
    else:
        twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>I didn't catch that. Could you please repeat your question?</Say>
    <Gather input="speech" action="/handle-input" method="POST" speechTimeout="1" language="en-US"/>
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
        
    return "\n\n".join(context)

# OpenAI LLM Response Formatting
def format_response(knowledge):
    max_retries = 3
    base_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(context=knowledge, query="")},
                    {"role": "user", "content": knowledge}
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
