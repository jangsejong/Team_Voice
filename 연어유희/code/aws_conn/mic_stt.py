import speech_recognition as sr

# microphone에서 auido source를 생성합니다
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)
    
def say_anything():    
    # 구글 웹 음성 API로 인식하기 (하루에 제한 50회)
    try:
        answer = r.recognize_google(audio, language='ko')
        print("Google Speech Recognition thinks you said : " + answer)
        return answer
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))