import os
from dotenv import load_dotenv
import requests

from utility_functions import cleanup_files, convert_call_audio_to_wav

# Load environment variables from .env file
load_dotenv()

# Extract Kayako API credentials from environment variables
KAYAKO_API_USERNAME = os.getenv('KAYAKO_API_USERNAME')  # Username for Kayako API authentication
KAYAKO_API_PASSWORD = os.getenv('KAYAKO_API_PASSWORD')  # Password for Kayako API authentication
KAYAKO_API_URL = os.getenv('KAYAKO_API_URL')            # Base URL for all Kayako API endpoints


def create_article_search_results_request(query):
    """
    Perform a search for knowledge base articles in Kayako using the API.
    
    This function sends a POST request to the Kayako search API endpoint
    to find articles that match the provided query string.
    
    Args:
        query (str): The search query text to find relevant knowledge base articles
        
    Returns:
        dict or None: JSON response from the API containing search results,
                     or None if the request failed
        
    Note:
        The function logs both the request and response for debugging purposes.
        API authentication is handled using the credentials from environment variables.
    """
    # API endpoint with query parameters
    url = f"{KAYAKO_API_URL}/helpcenter/search/articles.json"

    # Prepare the request data
    data = {
        "query": query
    }

    # Set up authentication using environment credentials
    auth = (KAYAKO_API_USERNAME, KAYAKO_API_PASSWORD)

    # Log the search query for debugging
    print(f"Searching for articles with query: '{query}'")
    
    try:
        # Make the API request with authentication
        response = requests.post(url, auth=auth, json=data)

        # Check if the request was successful (raises an exception if not)
        response.raise_for_status()

        # Log and return the JSON response
        print(f"Response: {response.json()}")
        return response.json()

    except requests.exceptions.RequestException as e:
        # Log detailed error information for debugging
        print(f"Error making request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None


def create_custom_support_ticket_request(name, email, subject, description, priority, transcript=None):
    """
    Create a new support ticket in the Kayako system via the API.
    
    This function handles formatting the ticket data, including conversation transcript
    if provided, and submits it to the Kayako API to create a new ticket.
    
    Args:
        name (str): Customer's name for identification
        email (str): Customer's email for notifications and identification
        subject (str): Brief description of the issue for the ticket title
        description (str): Detailed explanation of the customer's problem
        priority (int): Priority level of the ticket (1=Low, 2=Normal, 3=High, 4=Urgent)
        transcript (list, optional): List of conversation entries, each with 'role' and 'text' keys
        
    Returns:
        bool: True if ticket creation was successful, False otherwise
        
    Note:
        The ticket is styled with HTML formatting for better readability in the Kayako interface.
        The function includes a [GAUNTLET AI TEST] prefix in the subject for identification.
    """
    # Format transcript into a string with new line separation if provided
    formatted_transcript = ""
    if transcript:
        print("\n=== Full Conversation Transcript ===")
        for entry in transcript:
            print(f"{entry['role']}: {entry['text']}")
            formatted_transcript += f"{entry['role']}: {entry['text']}\n"
        print("================================\n")

    # API endpoint for case creation
    url = f"{KAYAKO_API_URL}/cases"

    # Generate individual transcript items with styling
    transcript_html = ""
    if transcript:
        for entry in transcript:
            role = entry['role']
            text = entry['text']
            
            # Style differently based on role
            if role.lower() in ['user', 'customer']:
                # Customer styling (purple background)
                transcript_html += f"""
                <div style="margin-bottom: 10px; padding: 8px; background-color: #f3e5f5; border-radius: 4px;">
                    <span style="color: #805ad5; font-weight: bold;">
                        Customer:
                    </span>
                    <span>
                        {text}
                    </span>
                </div>
                """
            else:
                # Agent styling (blue background)
                transcript_html += f"""
                <div style="margin-bottom: 10px; padding: 8px; background-color: #e3f2fd; border-radius: 4px;">
                    <span style="color: #2c5282; font-weight: bold;">
                        Agent:
                    </span>
                    <span>
                        {text}
                    </span>
                </div>
                """
    
    # Set priority display based on priority level
    priority_level = int(priority) if isinstance(priority, (int, str)) and str(priority).isdigit() else 1
    
    # Define styling and text for each priority level
    priority_styles = {
        1: {
            "bg_color": "#e6ffed",  # Light green
            "border_color": "#28a745",  # Green
            "text": "LOW",
            "description": "Will be handled when higher priorities are addressed"
        },
        2: {
            "bg_color": "#f1f8ff",  # Light blue
            "border_color": "#0366d6",  # Blue
            "text": "NORMAL",
            "description": "Standard response time expected"
        },
        3: {
            "bg_color": "#fff8c5",  # Light yellow
            "border_color": "#f9c513",  # Yellow
            "text": "HIGH",
            "description": "Should be addressed soon"
        },
        4: {
            "bg_color": "#ffeef0",  # Light red
            "border_color": "#d73a49",  # Red
            "text": "URGENT",
            "description": "Requires immediate attention"
        }
    }
    
    # Get the style for the given priority level (default to level 1 if invalid)
    style = priority_styles.get(priority_level, priority_styles[1])
    
    # Generate priority HTML section with appropriate styling
    priority_html = f"""
    <div style="margin-bottom: 20px; background-color: {style['bg_color']}; border-left: 4px solid {style['border_color']}; padding: 15px; border-radius: 4px;">
        <strong>🔔 PRIORITY: {style['text']}</strong><br>
        <p style="margin-top: 10px; margin-bottom: 0;">
            {style['description']}
        </p>
    </div>
    """

    # Create enhanced HTML content following sample.html format
    html_content = f"""
        <div style="font-family: Arial, sans-serif; line-height: 1.6;">

            <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
                <strong>📋 USER DETAILS</strong><br>
                <p style="margin-top: 10px; margin-bottom: 0; font-size: 16px;">Email: {email}</p>
                <p style="margin-top: 10px; margin-bottom: 0; font-size: 16px;">Name: {name}</p>
            </div>
            
            <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
                <strong>📋 SUBJECT</strong><br>
                <p style="margin-top: 10px; margin-bottom: 0; font-size: 16px;">{subject}</p>
            </div>
            
            <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
                <strong>📝 SUMMARY</strong><br>
                <p style="margin-top: 10px; margin-bottom: 0;">{description}</p>
            </div>

            {priority_html}

            <div style="margin-bottom: 20px;">
                <strong>📞 CALL TRANSCRIPT</strong><br>
                <div style="background: #f5f5f5; padding: 15px; border-radius: 4px; margin: 10px 0; max-height: 400px; overflow-y: auto;">
                    {transcript_html}
                </div>
            </div>

        </div>"""

    # Create data payload according to API documentation
    # HTML formatting is used to improve readability in the Kayako interface
    data = {
        "subject": f"[GAUNTLET AI TEST] Customer Support Ticket - {subject}",
        "contents": html_content,
        "channel": "MAIL",
        "channel_id": "1",
        "tags": "gauntlet-ai",
        "type_id": "7",
        "status_id": "1",
        "priority_id": "4",
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
            return True
        else:
            print(f"Error creating ticket: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error creating ticket: {e}")
        return False


# Create a call summary ticket
async def create_call_summary_ticket(transcript, call_sid=None):
    """
    Create a support ticket with call transcript and audio recording attachment.
    
    This asynchronous function creates a comprehensive ticket in Kayako that includes
    the full conversation transcript and, if available, the audio recording of the call.
    The audio file is converted from µ-law format to WAV for better compatibility.
    
    Args:
        transcript (list): List of conversation entries, each with 'role' and 'text' keys
        call_sid (str, optional): Twilio Call SID used to identify the call recording
        
    Returns:
        dict or None: JSON response from the Kayako API if successful, None otherwise
        
    Note:
        This function handles audio conversion using ffmpeg when available, with a Python
        fallback implementation. Temporary files are cleaned up after ticket creation.
    """
    # Format transcript into a string with new line separation
    print("\n=== Full Conversation Transcript ===")
    formatted_transcript = ""
    for entry in transcript:
        print(f"{entry['role']}: {entry['text']}")
        formatted_transcript += f"{entry['role']}: {entry['text']}\n"
    print("================================\n")

    paths = convert_call_audio_to_wav(call_sid)
    ulaw_path = paths[0]
    wav_path = paths[1]

    # Generate individual transcript items with styling
    transcript_html = ""
    if transcript:
        for entry in transcript:
            role = entry['role']
            text = entry['text']
            
            # Style differently based on role
            if role.lower() in ['user', 'customer']:
                # Customer styling (purple background)
                transcript_html += f"""
                <div style="margin-bottom: 10px; padding: 8px; background-color: #f3e5f5; border-radius: 4px;">
                    <span style="color: #805ad5; font-weight: bold;">
                        Customer:
                    </span>
                    <span>
                        {text}
                    </span>
                </div>
                """
            else:
                # Agent styling (blue background)
                transcript_html += f"""
                <div style="margin-bottom: 10px; padding: 8px; background-color: #e3f2fd; border-radius: 4px;">
                    <span style="color: #2c5282; font-weight: bold;">
                        Agent:
                    </span>
                    <span>
                        {text}
                    </span>
                </div>
                """
                
    # Define priority styling - using priority level 1 (LOW) for call summaries
    priority_style = {
        "bg_color": "#e6ffed",  # Light green
        "border_color": "#28a745",  # Green
        "text": "LOW",
        "description": "Routine call summary for review"
    }
    
    # Generate priority HTML section
    priority_html = f"""
    <div style="margin-bottom: 20px; background-color: {priority_style['bg_color']}; border-left: 4px solid {priority_style['border_color']}; padding: 15px; border-radius: 4px;">
        <strong>🔔 PRIORITY: {priority_style['text']}</strong><br>
        <p style="margin-top: 10px; margin-bottom: 0;">
            {priority_style['description']}
        </p>
    </div>
    """

    # Define the Kayako API endpoint for case creation
    url = f"{KAYAKO_API_URL}/cases"

    # Create formatted HTML payload for the ticket
    # This uses styled HTML to improve readability in the Kayako interface
    data = {
        "subject": f"[GAUNTLET AI TEST] Call Summary - Call {call_sid}",
        "contents": f"""
        <div style="font-family: Arial, sans-serif; line-height: 1.6;">
            <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
                <strong>📋 SUBJECT</strong><br>
                <p style="margin-top: 10px; margin-bottom: 0; font-size: 16px;">Kayako Help Center Call</p>
            </div>

            <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
                <strong>📝 SUMMARY</strong><br>
                <p style="margin-top: 10px; margin-bottom: 0;">Call assistance with Kayako help center.</p>
            </div>
            
            {priority_html}

            <div style="margin-bottom: 20px;">
                <strong>📞 CALL TRANSCRIPT</strong><br>
                <div style="background: #f5f5f5; padding: 15px; border-radius: 4px; margin: 10px 0; max-height: 400px; overflow-y: auto;">
                    {transcript_html}
                </div>
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

    result = None
    try:
        # Initialize empty files dictionary for multipart/form-data request
        files = {}

        # Set up authentication credentials for Kayako API
        auth = (KAYAKO_API_USERNAME, KAYAKO_API_PASSWORD)

        # Add the audio file to the request if it exists
        if wav_path and os.path.exists(wav_path):
            # Open the WAV file in binary mode and add it to the request
            with open(wav_path, 'rb') as f:
                files['attachment'] = (os.path.basename(wav_path), f, 'audio/wav')
                
                # Make the POST request with the audio attachment
                print(f"Creating case with Kayako API at {url}")
                response = requests.post(
                    url,
                    auth=auth,
                    data=data,
                    files=files
                )

                # Check if request was successful and log the result
                if response.status_code in [200, 201]:
                    result = response.json()
                    print(f"Successfully created ticket: {result.get('data', {}).get('id')}")
                else:
                    print(f"Error creating ticket: {response.status_code} - {response.text}")
            print(f"Attaching audio file: {wav_path}")
        else:
            # Make the POST request without audio attachment
            print("No audio file found")
            print(f"Creating case with Kayako API at {url}")
            response = requests.post(
                url,
                auth=auth,
                data=data,
                files=files
            )

            # Check if request was successful and log the result
            if response.status_code in [200, 201]:
                result = response.json()
                print(f"Successfully created ticket: {result.get('data', {}).get('id')}")
            else:
                print(f"Error creating ticket: {response.status_code} - {response.text}")

    except Exception as e:
        # Log any errors that occur during ticket creation
        print(f"Error creating ticket: {e}")

    finally:
        # Always clean up temporary files, regardless of success or failure
        # This prevents accumulation of temporary files on the server
        cleanup_files(ulaw_path, wav_path)

    return result
