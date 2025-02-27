import asyncio
import os

from twilio.rest import Client

from kayako_functions import create_article_search_results_request, create_custom_support_ticket_request
from utility_functions import redirect_call

# Load environment variables
KAYAKO_API_USERNAME = os.getenv('KAYAKO_API_USERNAME')  # Kayako API authentication username
KAYAKO_API_PASSWORD = os.getenv('KAYAKO_API_PASSWORD')  # Kayako API authentication password
KAYAKO_API_URL = os.getenv('KAYAKO_API_URL')            # Base URL for Kayako API endpoints
TWILIO_SID = os.getenv('TWILIO_SID')                    # Twilio account SID for authentication
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')      # Twilio's authentication token

# Initialize Twilio client for making API calls to Twilio services
twilio_client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)


def get_article_search_results(query):
    """
    Search for articles in Kayako's knowledge base using the provided query.
    
    This function delegates to the Kayako API function to perform the search and
    return relevant articles that match the user's query.
    
    Args:
        query (str): The search query text to use for finding relevant articles
        
    Returns:
        dict: The JSON response from Kayako's article search API containing matching articles
    """
    return create_article_search_results_request(query)


def connect_customer_to_agent(name, email, call_sid, twilio_client):
    """
    Connect a customer to a live agent through the Twilio call.
    
    This function handles the logic of determining if an agent is available and
    initiating the call transfer process when possible. In a production environment,
    this would check agent availability through a queue management system.
    
    Args:
        name (str): Customer's name for identification
        email (str): Customer's email for identification and follow-up
        call_sid (str): Twilio Call SID identifier for the active call
        twilio_client (twilio.rest.Client): Initialized Twilio client for API calls
        
    Returns:
        dict: Status information about the connection attempt
            - status: "success" or "error"
            - message: Description of the outcome
    """

    # 50/50 dice roll to determine if an agent is available
    is_agent_available = False # random.choice([True, False])
    
    # Log the availability check for monitoring
    print(f"Agent availability check for {name} ({email}): {'Available' if is_agent_available else 'Unavailable'}")
    
    if is_agent_available:
        # If we have an active call SID, initiate the call redirect process
        if call_sid:  
            # Use asyncio.create_task since we can't await directly in this sync function
            # This schedules the redirect without blocking the current execution
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


def create_custom_support_ticket(name, email, subject, description, priority, transcript=None):
    """
    Create a custom support ticket in the Kayako system.
    
    This function handles the creation of a new support ticket when a customer
    needs assistance but cannot be connected to a live agent.
    
    Args:
        name (str): Customer's name for the ticket
        email (str): Customer's email for notifications and identification
        subject (str): Brief description of the customer's issue
        description (str): Detailed explanation of the customer's problem
        priority (int): The priority of the support ticket
        transcript (list, optional): List of conversation exchanges between the bot and customer
        
    Returns:
        dict: Status information about the ticket creation
            - status: "success" or "error"
            - message: Description of the outcome
    """
    is_ticket_created = create_custom_support_ticket_request(name, email, subject, description, priority, transcript)
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

