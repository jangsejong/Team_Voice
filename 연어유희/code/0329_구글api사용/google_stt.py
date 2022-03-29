import os
import winsound
import speech_recognition as sr

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="chat_api.json"
    
def say_anything(): 
    try:
        r = sr.Recognizer()
        # microphone에서 auido source를 생성합니다
        with sr.Microphone() as source:
            print("<<< 마이크에 이야기 하세요 >>>")
            winsound.PlaySound("ns_1_01.wav", winsound.SND_FILENAME)
            audio = r.listen(source)
            answer = r.recognize_google(audio, language='ko')                   
            return answer
    except:
        print("마이크 입력 에러 입니다")
        return say_anything()
    
if __name__ == '__main__':    
    print(f"마이크에 입력된 값 : {say_anything()}")


        
