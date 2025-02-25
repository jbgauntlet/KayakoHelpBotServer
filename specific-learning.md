input_audio_buffer.committed
{
    "event_id": "event_1121",
    "type": "input_audio_buffer.committed",
    "previous_item_id": "msg_001",
    "item_id": "msg_002"
}

Returned when an input audio buffer is committed, either by the client or automatically in server VAD mode. The item_id property is the ID of the user message item that will be created, thus a conversation.item.created event will also be sent to the client.

input_audio_buffer:
This is a temporary storage area where incoming audio data from the user (via your client application) is collected. Think of it as a holding area for the raw audio before it's processed.

committed:
When the audio in the buffer is "committed," it means that the system has determined that the user has finished speaking (either because the client signaled an end or the server’s voice activity detection (VAD) determined a pause) and that the audio data is now finalized. The committed audio is then ready to be transformed (e.g., transcribed) and added as part of the conversation history.

either by the client or automatically in server VAD mode:

By the client: The client (your application) might explicitly decide that the user is done speaking (for example, via a button press or a timeout) and instructs the system to commit the audio buffer.
Automatically in server VAD mode: The server uses Voice Activity Detection (VAD) to automatically determine when the user has stopped speaking. In this mode, when a sufficient pause is detected, the server will automatically commit the audio without any extra input from the client.
item_id property:
Once the audio buffer is committed, the system creates a new conversation message (often called an "item"). The item_id is the unique identifier assigned to this new user message. It lets you reference this specific message later on if needed.

conversation.item.created event:
After the audio is committed and a new message item is created, the server sends out a conversation.item.created event. This event informs your client that a new user message has been added to the conversation history. Essentially, it’s a notification that the transcription (or the audio message) is now part of the conversation.

In summary, when the system detects that the user’s speech is complete—either manually or automatically—it finalizes the current audio input, assigns it a unique message ID, and then notifies your application that a new message has been created in the conversation history. This process ensures that the conversation context is updated correctly and that any further processing (like generating a response) has access to the complete user input.

