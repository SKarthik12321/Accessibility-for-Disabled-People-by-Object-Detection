# text_to_speech.py
from gtts import gTTS
import os
import playsound

def speech(text):
    print(text)  # For logging purposes
    language = "en"
    output_path = "output.mp3"  # Output file name
    
    try:
        # Create the gTTS object and save the speech to a file
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(output_path)
        
        # Play the sound
        playsound.playsound(output_path)

    except Exception as e:
        print(f"Error occurred during text-to-speech conversion: {e}")

    finally:
        # Remove the temporary file after playing
        if os.path.exists(output_path):
            os.remove(output_path)