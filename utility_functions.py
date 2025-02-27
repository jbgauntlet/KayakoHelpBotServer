import os
from dotenv import load_dotenv

# Load environment variables from .env file
# This must be called before accessing any environment variables
load_dotenv()


# Redirect a Twilio call
async def redirect_call(call_sid, twilio_client):
    """
    Redirect an active Twilio call to a new destination.
    
    This asynchronous function initiates a call transfer by instructing Twilio
    to fetch new TwiML instructions from a specified endpoint. The new TwiML
    will control the call flow after redirection.
    
    Args:
        call_sid (str): The unique identifier for the Twilio call to be redirected
        twilio_client (twilio.rest.Client): Initialized Twilio client for API calls
        
    Returns:
        bool: True if redirection was successful, False otherwise
        
    Raises:
        Exception: Any exception that occurs during the redirection process will be caught,
                 logged, and False will be returned
    """
    try:
        # Get the server URL from environment variables
        SERVER_URL = os.getenv("SERVER_URL")
        
        # Construct the URL that will provide TwiML for the redirected call
        redirect_url = f"{SERVER_URL}/redirect-to-agent"
        
        # Update the call with new TwiML instructions
        twilio_client.calls(call_sid).update(
            url=redirect_url,
            method='POST'
        )

        # Log successful redirection
        print(f"Successfully redirected call {call_sid} to {redirect_url}")
        return True
    except Exception as e:
        # Log any errors that occur during redirection
        print(f"Error redirecting call: {e}")
        return False


# Helper function to clean up temporary files
def cleanup_files(ulaw_path, wav_path):
    """
    Clean up temporary audio files created during call processing.
    
    This function safely deletes audio files that were created during
    call recording and processing, to prevent accumulation of temporary files.
    
    Args:
        ulaw_path (str): Path to the µ-law encoded audio file
        wav_path (str): Path to the WAV format audio file
        
    Returns:
        None
        
    Note:
        The function handles missing files gracefully and will not raise
        exceptions if files don't exist or can't be deleted.
    """
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
        # Log any errors but don't propagate them
        print(f"Error cleaning up files: {e}")


# End a Twilio call using the REST API
async def end_call(call_sid, twilio_client):
    """
    End an active Twilio call programmatically.
    
    This asynchronous function terminates a call by setting its status to 'completed'
    through the Twilio REST API, which gracefully ends the connection.
    
    Args:
        call_sid (str): The unique identifier for the Twilio call to be ended
        twilio_client (twilio.rest.Client): Initialized Twilio client for API calls
        
    Returns:
        None
        
    Raises:
        Exception: Any exception that occurs during the call termination process will be caught
                 and logged, but not propagated
    """
    try:
        # Update the call status to 'completed' to end it
        twilio_client.calls(call_sid).update(status='completed')
        print(f"Successfully ended call {call_sid}")
    except Exception as e:
        # Log any errors that occur during call termination
        print(f"Error ending call: {e}")


def convert_call_audio_to_wav(call_sid):
    """
    Convert a µ-law encoded audio file to WAV format.
    
    This function uses the `ffmpeg` command-line tool to convert a µ-law encoded
    audio file to WAV format. It handles errors and missing files gracefully.
    """
    # Prepare paths for audio files if we have a call_sid
    wav_path = None
    ulaw_path = None
    if call_sid:
        try:
            # Define file paths for the audio recordings
            ulaw_path = f'call_{call_sid}.ulaw'
            wav_path = f'call_{call_sid}.wav'

            # Check if the raw µ-law audio file exists
            if os.path.exists(ulaw_path):
                print(f"Converting audio for call {call_sid}...")

                try:
                    # Try using ffmpeg for better audio conversion with enhancements
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

                    # Run ffmpeg command as a subprocess
                    process = subprocess.run(
                        ffmpeg_cmd,
                        capture_output=True,
                        text=True
                    )

                    # Check if the ffmpeg conversion was successful
                    if process.returncode == 0:
                        print(f"Successfully converted audio to {wav_path} with enhanced quality")
                    else:
                        print(f"FFmpeg audio conversion failed: {process.stderr}")
                        print("Falling back to Python-based conversion")

                        # Fall back to Python-based conversion if ffmpeg fails
                        # This is slower but more portable as it doesn't require external dependencies
                        import wave
                        import numpy as np

                        # Function to convert u-law to linear PCM
                        def ulaw2linear(u_law_data):
                            # µ-law decoding table (u=255 for 8-bit µ-law)
                            u = 255
                            u_law_data = np.frombuffer(u_law_data, dtype=np.uint8)
                            # Convert to signed integers based on the sign bit
                            sign = np.where(u_law_data < 128, 1, -1)
                            # Remove sign bit from the data
                            u_law_data = np.where(u_law_data < 128, u_law_data, 255 - u_law_data)
                            # Decode using µ-law formula to get linear PCM values
                            linear_data = sign * (((u_law_data / u) ** (1 / 1.5)) * (2 ** 15 - 1))
                            return linear_data.astype(np.int16)

                        # Read raw µ-law data from file
                        with open(ulaw_path, 'rb') as f:
                            u_law_data = f.read()

                        # Convert µ-law to linear PCM samples
                        linear_data = ulaw2linear(u_law_data)

                        # Create WAV file with proper headers
                        with wave.open(wav_path, 'wb') as wav_file:
                            wav_file.setnchannels(1)  # Mono
                            wav_file.setsampwidth(2)  # 2 bytes (16 bits) per sample
                            wav_file.setframerate(8000)  # 8 kHz sampling rate for µ-law
                            wav_file.writeframes(linear_data.tobytes())

                        print(f"Successfully converted audio to {wav_path} using fallback method")
                except Exception as e:
                    # Log conversion errors but continue with ticket creation
                    print(f"Audio conversion failed: {e}")
                    # Fallback message for troubleshooting
                    print("Consider checking ffmpeg installation if this error persists")
                    wav_path = None
            else:
                print(f"No audio file found at {ulaw_path}")
                wav_path = None
        except Exception as e:
            print(f"Error processing audio: {e}")
            wav_path = None
    return [ulaw_path, wav_path]