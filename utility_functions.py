import os
from dotenv import load_dotenv

load_dotenv()


# Redirect a Twilio call
async def redirect_call(call_sid, twilio_client):
    """Redirect an active Twilio call to a new URL or phone number."""
    try:
        # The URL should return TwiML that Twilio will use for the redirect
        SERVER_URL = os.getenv("SERVER_URL")
        redirect_url = f"{SERVER_URL}/redirect-to-agent"
        twilio_client.calls(call_sid).update(
            url=redirect_url,
            method='POST'
        )

        print(f"Successfully redirected call {call_sid} to {redirect_url}")
        return True
    except Exception as e:
        print(f"Error redirecting call: {e}")
        return False


# Helper function to clean up temporary files
def cleanup_files(ulaw_path, wav_path):
    """Clean up temporary audio files."""
    try:
        # Remove ulaw file if it exists
        if ulaw_path and os.path.exists(ulaw_path):
            os.remove(ulaw_path)
            print(f"Deleted ulaw file: {ulaw_path}")

        # Remove wav file if it exists
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
            print(f"Deleted wav file: {wav_path}")
    except Exception as e:
        print(f"Error cleaning up files: {e}")


# End a Twilio call using the REST API
async def end_call(call_sid, twilio_client):
    """End a Twilio call using the REST API."""
    try:
        twilio_client.calls(call_sid).update(status='completed')
        print(f"Successfully ended call {call_sid}")
    except Exception as e:
        print(f"Error ending call: {e}")
