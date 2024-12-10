import openai
import key
import tempfile
import os

conversation = []

openai.api_key = key.openai_key

def get_ai_response(question):
    messages = [
{
    "role": "system",
    "content": """
    You are a salesperson representing Devmate Solutions, a digital marketing and AI automation company. 
    Your goal is to have a concise, interactive conversation to understand the client’s needs, gather their contact information, and assure them that they will be contacted soon for further details. 
    Share service information or prices only if the client asks. Focus on asking relevant questions rather than giving too much detail upfront.

    Services offered:
    - Web Development: $544
    - App Development: $744
    - Social Media Marketing: $344
    - Branding: $244
    - AI Automation: $644 
      (includes automated calls, chatbot creation, and WhatsApp automation)

    Conversation guidelines:
    1. Begin with a friendly introduction, such as: "Hello, I'm Dua, a representative at Devmate Solutions. How can I assist you today?"
    2. Ask questions to understand the client’s business needs and challenges:
       - "Can you tell me a bit about your business?"
       - "What areas are you looking to improve—marketing, automation, or development?"
       - "Are you currently using any tools for digital marketing or automation?"
    3. Gather the client’s contact details:
       - "Could I get your WhatsApp number and email to follow up with you?"
    4. If the client asks about services or prices, provide the requested information briefly and ask follow-up questions like:
       - "Does that sound like something that fits your needs?"
       - "Would you like to discuss a custom plan?"
    5. Offer to schedule a follow-up meeting for customized plans (at least 2 days ahead, excluding Sunday):
       - "I can arrange a meeting with one of our specialists. Would [insert date] work for you?"
    6. Conclude by thanking the client and assuring them they will hear back soon.

    Always keep responses short, polite, and question-focused, waiting for the client's reply before proceeding.
    """
},
        {
            "role": "assistant",
            "content": "Hello, my name is Dua and I am calling from Devmate Solutions. How are you doing today?"
        }
    ]

    for message in conversation:
        if "assistant" in message:
            messages.append({
                "role": "assistant",
                "content": message["assistant"]
            })
        if "user" in message:
            messages.append({
                "role": "user",
                "content": message["user"]
            })
    messages.append({
        "role": "user",
        "content": question
    })

    conversation.append({
        "user": question
    })

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        temperature=0,
        messages=messages,
        stream=True
    )

    def generate():
        ai_response = ""
        for chunk in response:
            if "content" in chunk.choices[0].delta:
                content = chunk.choices[0].delta.content
                ai_response += content
                yield content
        conversation.append({
            "assistant": ai_response
        })
    return generate

def transcribe(request):
    # Create a temporary file with delete=False so we can manually delete it later
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file_name = temp_file.name

    try:
        # Extract the audio content from the request
        data = request.files
        audio_content = data["audio"]
        
        # Write the audio content to the temporary file
        with open(temp_file_name, "wb") as f:
            f.write(audio_content.read())
        
        # Transcribe the audio file
        with open(temp_file_name, "rb") as audio_file:
            transcription = openai.Audio.transcribe("whisper-1", audio_file)

        return transcription["text"]
    
    finally:
        # Always remove the temporary file
        if os.path.exists(temp_file_name):
            os.remove(temp_file_name)