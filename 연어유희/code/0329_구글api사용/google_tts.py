import os
import playsound

def synthesize_text(text):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="D:\\__ChatBot\\google_api_set\\chat_api.json"
    """Synthesizes speech from the input string of text."""
    from google.cloud import texttospeech

    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        name="ko-KR-Wavenet-A",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )
    # The response's audio_content is binary.
    with open("D:\\__ChatBot\\google_api_set\\output.mp3", "wb") as out:
        out.write(response.audio_content)
        # print('Audio content written to file "output.mp3"')
    
if __name__ == '__main__':  
    text = "구글 TTS API를 활용하여 문자를 읽어 드립니다"  
    synthesize_text(text)
    playsound.playsound("D:\\__ChatBot\\google_api_set\\output.mp3")
        


