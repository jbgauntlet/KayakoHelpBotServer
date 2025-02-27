import os
from dotenv import load_dotenv
import requests

from utility_functions import cleanup_files

load_dotenv()
KAYAKO_API_USERNAME = os.getenv('KAYAKO_API_USERNAME')
KAYAKO_API_PASSWORD = os.getenv('KAYAKO_API_PASSWORD')
KAYAKO_API_URL = os.getenv('KAYAKO_API_URL')


def create_article_search_results_request(query):
    """Make the API request to search for articles."""
    # API endpoint with query parameters
    url = f"{KAYAKO_API_URL}/helpcenter/search/articles.json"

    # Prepare the request data
    data = {
        "query": query
    }

    # Set up authentication
    auth = (KAYAKO_API_USERNAME, KAYAKO_API_PASSWORD)

    # Make the API request
    print(f"Searching for articles with query: '{query}'")
    try:
        response = requests.post(url, auth=auth, json=data)

        # Check if the request was successful
        response.raise_for_status()

        # Return the JSON response
        print(f"Response: {response.json()}")
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return None


def create_custom_support_ticket_request(name, email, subject, description, transcript=None):
    """Create a ticket with call transcript and audio attachment."""
    # Format transcript into a string with new line separation if provided
    formatted_transcript = ""
    if transcript:
        print("\n=== Full Conversation Transcript ===")
        for entry in transcript:
            print(f"{entry['role']}: {entry['text']}")
            formatted_transcript += f"{entry['role']}: {entry['text']}\n"
        print("================================\n")

    # API endpoint with query parameters
    url = f"{KAYAKO_API_URL}/cases"

    # Create data payload according to API documentation
    data = {
        "subject": f"[GAUNTLET AI TEST] Customer Support Ticket - {subject}",
        "contents": f"""<div style="font-family: Arial, sans-serif; line-height: 1.6;">
            <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
                <strong>📋 SUBJECT</strong><br>
                <p style="margin-top: 10px; margin-bottom: 0; font-size: 16px;">{subject}</p>
            </div>

            <div style="margin-bottom: 20px; background-color: #f6f8fa; padding: 15px; border-radius: 4px; border-left: 4px solid #0366d6;">
                <strong>📝 SUMMARY</strong><br>
                <p style="margin-top: 10px; margin-bottom: 0;">{description}</p>
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
    ulaw_path = None
    if call_sid:
        try:
            # Check if the raw audio file exists
            ulaw_path = f'call_{call_sid}.ulaw'
            wav_path = f'call_{call_sid}.wav'

            if os.path.exists(ulaw_path):
                print(f"Converting audio for call {call_sid}...")

                try:
                    # Use ffmpeg for better audio conversion with enhancements
                    import subprocess

                    # Command to convert .ulaw to .wav with audio enhancements:
                    # - Proper u-law to PCM conversion
                    # - Bandpass filter for speech frequencies (300-3400 Hz)
                    # - Equalization to enhance speech clarity
                    # - Volume normalization
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output file if it exists
                        '-f', 'mulaw',  # Input format
                        '-ar', '8000',  # Input sample rate
                        '-ac', '1',  # Input channels (mono)
                        '-i', ulaw_path,  # Input file
                        '-af', 'highpass=f=200,lowpass=f=3500,equalizer=f=1000:width_type=h:width=200:g=4,loudnorm',
                        # Audio filters
                        '-ar', '16000',  # Increased output sample rate
                        '-ac', '1',  # Output channels (mono)
                        '-acodec', 'pcm_s16le',  # Output codec (16-bit PCM)
                        wav_path  # Output file
                    ]

                    # Run ffmpeg command
                    process = subprocess.run(
                        ffmpeg_cmd,
                        capture_output=True,
                        text=True
                    )

                    # Check if conversion was successful
                    if process.returncode == 0:
                        print(f"Successfully converted audio to {wav_path} with enhanced quality")
                    else:
                        print(f"FFmpeg audio conversion failed: {process.stderr}")
                        print("Falling back to Python-based conversion")

                        # Fall back to Python-based conversion if ffmpeg fails
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
                            linear_data = sign * (((u_law_data / u) ** (1 / 1.5)) * (2 ** 15 - 1))
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

                        print(f"Successfully converted audio to {wav_path} using fallback method")
                except Exception as e:
                    print(f"Audio conversion failed: {e}")
                    # Fallback message
                    print("Consider checking ffmpeg installation if this error persists")
                    wav_path = None
            else:
                print(f"No audio file found at {ulaw_path}")
                wav_path = None
        except Exception as e:
            print(f"Error processing audio: {e}")
            wav_path = None

    # API endpoint with query parameters
    url = f"{KAYAKO_API_URL}/cases"

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

    result = None
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
                else:
                    print(f"Error creating ticket: {response.status_code} - {response.text}")
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
            else:
                print(f"Error creating ticket: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error creating ticket: {e}")

    finally:
        # Clean up files after ticket creation attempt (regardless of success)
        cleanup_files(ulaw_path, wav_path)

    return result
