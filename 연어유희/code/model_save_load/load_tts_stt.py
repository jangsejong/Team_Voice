import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

Q_TKN = "<usr>"  # 질문
A_TKN = "<sys>"  # 대답
BOS = '</s>'   # 시작
EOS = '</s>'   # 끝
MASK = '<unused0>'
SENT = '<unused1>'  # 감정
PAD = '<pad>' 

PATH = 'C:\\Users\\남궁지희\\Desktop\\Team\\save\\'
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",  
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

model.load_state_dict(torch.load(PATH + 'model_jihee.pt'))

#####################################
import aws_sql #aws 연결하는 모듈
import mic_stt #mic 연결하는 모듈
from gtts import gTTS 
'''
pip install pipwin
pipwin install pyaudio
pip install gTTS
pip install speechrecognition
pip install pymysql 
'''
sent = '0'
with torch.no_grad():    
    while 1:
        q = mic_stt.say_anything()
        # q = input("나 >> ").strip()   # 인자로 전달된 문자를 String의 왼쪽과 오른쪽에서 제거합니다.
        # print(f'나>> {q}')
        if q == "종료":  
            break        
        a = ""
        while 1:
            input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + sent + A_TKN + a)).unsqueeze(dim=0)
            pred = model(input_ids)
            pred = pred.logits
            gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == EOS:
                break
            a += gen.replace("▁", " ")
            
        print("유희지희 > {}".format(a.strip()))
        print("OK")
        
        # tts 출력 ##
             
        tts = gTTS(text=a, lang='ko')
        tts.save("tts_answer.mp3")
        import playsound
        playsound.playsound("tts_answer.mp3")
        
        ## sql 입력 ##
        aws_sql.insert_table("Question",q)
        aws_sql.insert_table("Answer",a)