import asyncio
import os

from twilio.rest import Client

from kayako_functions import create_article_search_results_request, create_custom_support_ticket_request
from utility_functions import redirect_call

# Load environment variables
KAYAKO_API_USERNAME = os.getenv('KAYAKO_API_USERNAME')
KAYAKO_API_PASSWORD = os.getenv('KAYAKO_API_PASSWORD')
KAYAKO_API_URL = os.getenv('KAYAKO_API_URL')
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')

# Initialize Twilio client
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)


def get_article_search_results(query):
    """Search for articles in Kayako and return the results."""
    return create_article_search_results_request(query)


def connect_customer_to_agent(name, email, call_sid, twilio_client):
    """Connect a customer to an agent."""

    # 50/50 dice roll to determine if an agent is available
    is_agent_available = True # random.choice([True, False])
    
    print(f"Agent availability check for {name} ({email}): {'Available' if is_agent_available else 'Unavailable'}")
    
    if is_agent_available:
        # Get the current call SID from your global/nonlocal variable
        # You'll need to make call_sid available here
        if call_sid:  
            # Schedule the redirect (use asyncio.create_task since we can't await directly in this sync function)
            asyncio.create_task(redirect_call(call_sid, twilio_client))
            
        return {
            "status": "success",
            "message": "Agent is available. Connecting customer to agent..."
        }
    else:
        return {
            "status": "error",
            "message": "No agent is available. Offering to create a support ticket."
        }


def create_custom_support_ticket(name, email, subject, description, transcript=None):
    """Create a custom support ticket."""
    is_ticket_created = create_custom_support_ticket_request(name, email, subject, description, transcript)
    if is_ticket_created:
        return {
            "status": "success",
            "message": "Ticket created successfully."
        }
    else:
        return {
            "status": "error",
            "message": "Failed to create ticket."
        }

