from flask import Flask, request, Response
from flask_cors import CORS
from ai import get_ai_response, transcribe
from elevenlabs import generate, stream, set_api_key, voices
import key

app = Flask(__name__)

CORS(app)

set_api_key(key.ELEVENLABS_API_KEY)

@app.route("/speak", methods=["POST"])
def speak():
    # Transcribe the incoming audio
    question = transcribe(request)
    
    # Get the AI response text
    generate_response = get_ai_response(question)
    response_text = ''.join(generate_response())  # Convert generator to full text

    # Print available voices to help debug voice selection
    available_voices = [voice.name for voice in voices()]
    print("Available voices:", available_voices)

    # Choose an available voice (replace with an actual voice from the list)
    voice_name = "Sarah"  # using an available voice from the list

    # Generate audio
    audio = generate(
        text=response_text,
        voice=voice_name,
        model="eleven_multilingual_v2",
        stream=True
    )
    
    return Response(audio, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(debug=True)