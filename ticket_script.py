import requests
import base64
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API endpoint
KAYAKO_API_URL = os.getenv('KAYAKO_API_URL', 'https://doug-test.kayako.com/api/v1/cases')
url = f"{KAYAKO_API_URL}?include=channel,last_public_channel,mailbox,facebook_page,facebook_account,twitter_account,user,organization,sla_metric,sla_version_target,sla_version,identity_email,identity_domain,identity_facebook,identity_twitter,identity_phone,case_field,read_marker"

# Credentials
KAYAKO_API_USERNAME = os.getenv('KAYAKO_API_USERNAME')
KAYAKO_API_PASSWORD = os.getenv('KAYAKO_API_PASSWORD')

# Create data payload
data = {
    "field_values": {
        "product": "80"
    },
    "status_id": "1",
    "attachment_file_ids": "",
    "tags": "gauntlet-ai",
    "type_id": 7,
    "channel": "MAIL",
    "subject": "[GAUNTLET AI TEST] Call Summary - Johnny Test Email",
    "contents": """<div style="font-family: Arial, sans-serif; line-height: 1.6;">
    <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
        <strong>📋 SUBJECT</strong><br>
        <p style="margin-top: 10px; margin-bottom: 0; font-size: 16px;">Direct-fire recirculations for objectives</p>
    </div>

    <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
        <strong>📧 CUSTOMER EMAIL</strong><br>
        <p style="margin-top: 10px; margin-bottom: 0; font-size: 16px;">billy@gmail.com</p>
    </div>

    <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
        <strong>📝 SUMMARY</strong><br>
        <p style="margin-top: 10px; margin-bottom: 0;">The customer contacted support about Kayako services. The AI provided assistance with Technical Support. </p>
    </div>

    <div style="margin-bottom: 20px; background-color: #fff8c5; border-left: 4px solid #f9c513; padding: 15px; border-radius: 4px;">
        <strong>🔔 PRIORITY: MEDIUM</strong><br>
        <p style="margin-top: 10px; margin-bottom: 0;">
            Should be addressed soon
        </p>
    </div>
    
    <div style="margin-bottom: 20px; background-color: #fff8c5; border-left: 4px solid #f9c513; padding: 15px; border-radius: 4px;">
            <strong>⚠️ FOLLOW-UP REQUIRED</strong><br>
            <p style="margin-top: 10px; margin-bottom: 0;">
                Reason: General follow-up required
            </p>
        </div>
    
    
    

    
    <div style="margin-bottom: 20px;">
        <strong>📞 CALL TRANSCRIPT</strong><br>
        <div style="background: #f5f5f5; padding: 15px; border-radius: 4px; margin: 10px 0; max-height: 400px; overflow-y: auto;">
            
                <div style="margin-bottom: 10px; padding: 8px; background-color: #f3e5f5; border-radius: 4px;">
                    <span style="color: #805ad5; font-weight: bold;">
                        Customer: 
                    </span>
                    <span>
                        Can you forward me to a human support agent?
                    </span>
                </div>
            

                <div style="margin-bottom: 10px; padding: 8px; background-color: #e3f2fd; border-radius: 4px;">
                    <span style="color: #2c5282; font-weight: bold;">
                        Agent: 
                    </span>
                    <span>
                        Before I assist you, could you please provide your e-mail address so I can follow up if needed?
                    </span>
                </div>
            

                <div style="margin-bottom: 10px; padding: 8px; background-color: #f3e5f5; border-radius: 4px;">
                    <span style="color: #805ad5; font-weight: bold;">
                        Customer: 
                    </span>
                    <span>
                        Sure. My e-mail address is billy at gmail dot com.
                    </span>
                </div>
            

                <div style="margin-bottom: 10px; padding: 8px; background-color: #e3f2fd; border-radius: 4px;">
                    <span style="color: #2c5282; font-weight: bold;">
                        Agent: 
                    </span>
                    <span>
                        Thank you. I've noted your e-mail as billy at gmail dot com. Now, how can I help you with Kayako? If you need assistance from a human support agent, I can help with that process as well.
                    </span>
                </div>
            
        </div>
    </div>
    
</div>""",
    "assigned_agent_id": "309",
    "_is_fully_loaded": False,
    "form_id": "1",
    "assigned_team_id": "1",
    "requester_id": "309",
    "channel_id": "1",
    "priority_id": "1",
    "channel_options": {
        "cc": [],
        "html": True
    }
}

# Encode credentials for Basic Auth
auth_string = f"{KAYAKO_API_USERNAME}:{KAYAKO_API_PASSWORD}"
encoded_auth = base64.b64encode(auth_string.encode()).decode()

# Set headers
headers = {
    "Content-Type": "application/json; charset=UTF-8",
    "Authorization": f"Basic {encoded_auth}"
}

try:
    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # Check if request was successful
    response.raise_for_status()
    
    # Parse and print the JSON response
    result = response.json()
    print(json.dumps(result, indent=2))
    
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
