import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import time

start = time.time()
print(".......... 유희 지희 를 호출 중 입니다 .............")

Q_TKN = "<usr>"  # 질문
A_TKN = "<sys>"  # 대답
BOS = '</s>'   # 시작
EOS = '</s>'   # 끝
MASK = '<unused0>'
SENT = '<unused1>'  # 감정
PAD = '<pad>' 
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, unk_token="<unk>", pad_token=PAD, mask_token=MASK,)

path ='D:\\__ChatBot\\google_api_set\\Data50000_EPOCH10.pt'
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

model.load_state_dict(torch.load(path))
print("모델 로딩 시간 :",round(time.time() - start,2)," 초")  
model.eval()

import google_stt
import google_tts
import aws_sql
import playsound

sent = '0'
with torch.no_grad():
    while True: 
        print("---------------------------")     
        q=""   
        q = google_stt.say_anything().strip()
        if q == "잘 자": 
            a = "즐거운 대화였어용"
            print(f"유희 >  {a}")
            google_tts.synthesize_text(a)
            playsound.playsound("D:\\__ChatBot\\google_api_set\\output.mp3")
            break     
          
        print(f"    나 >> {q}")                        
        a = ""
        s_time = time.time()
        print("| 답변 생성중 ",end="")
        while True:                       
            input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + sent + A_TKN + a)).unsqueeze(dim=0)
            pred = model(input_ids)
            pred = pred.logits
            gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == EOS:
                break
            a += gen.replace("▁", " ")
            
            print("|",end="")
        print("|")                        
        a = a.strip()            
        print(f"    유희 >> {a}")
        google_tts.synthesize_text(a) 
        playsound.playsound("D:\\__ChatBot\\google_api_set\\output.mp3")
        aws_sql.insert_QnA(q,a)    
        print("Answer gen time :",round(time.time() - s_time,2)," sec")       
        continue
