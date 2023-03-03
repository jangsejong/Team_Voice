import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>') 
tokenizer.tokenize("안녕하세요. 한국어 GPT-2 입니다.😤:)l^o")

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import re

Q_TKN = "<usr>"  # 질문
A_TKN = "<sys>"  # 대답
BOS = '</s>'   # 시작
EOS = '</s>'   # 끝
MASK = '<unused0>'
SENT = '<unused1>'  # 감정
PAD = '<pad>' 

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",  
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

path = 'C:\\Users\\bitcamp\\Desktop\\TP\\' 
Chatbot_Data = pd.read_csv(path + "ChatBotData_지희.csv") 

Chatbot_Data = Chatbot_Data[:1100]
Chatbot_Data.head()

BOS = "</s>"     # 문장의 시작을 나타내는 token
EOS = "</s>"     # 문장의 끝을 나타내는 token
PAD = "<pad>"    
MASK = "<unused0>" 

'''
bos_token : 문장의 시작을 나타내는 token
eos_token : 문장의 끝을 나타내는 token
unk_token : 모르는 단어를 나타내는 token
pad_token : 동일한 batch 내에서 입력의 크기를 동일하게 하기 위해서 사용하는 token

PreTrainedTokenizer 에서 제공되는 함수는

tokenize() : tokenizer를 이용해서 string을 token id의 리스트로 변환한다.
get_added_vocab() : token to index에 해당하는 dict를 리턴한다.
batch_decode() : token id로 구성된 입력을 하나의 연결된 string으로 출력한다.
convert_ids_to_tokens() : token id 의 리스트를 token으로 변환한다. skip_special_tokens=True로 하면 decoding할 때 special token들을 제거한다.
convert_tokens_to_ids() : token string의 리스트를 token id 또는 Token id의 리스트로 변환한다.
decode() : tokenizer 와 vocabulary를 이용해서 token id를 string으로 변환한다. skip_special_token=True로 지정하면 speical token들을 제외한다.
encode() : token string을 token id 의 리스트로 변환한다. add_special_tokens=False로 지정하면 token id로 변환할 때 special token들을 제외한다. 
padding을 통해서 padding token을 어떻게 추가할지도 지정할 수 있다.
'''

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, unk_token="<unk>", pad_token=PAD, mask_token=MASK,)

class ChatbotDataset(Dataset): # 자동으로 데이터를 불러오는 클래스
    def __init__(self, chats, max_len=40):  # 데이터셋의 전처리를 해주는 부분
        self._data = chats     # 챗봇 데이터
        self.max_len = max_len # 최대 길이를 저장한다.
        self.q_token = Q_TKN   # 질문
        self.a_token = A_TKN   # 대답
        self.sent_token = SENT # 감정
        self.eos = EOS         # 문장의 끝을 나타내는 token
        self.mask = MASK       # 마스크를 나타내는 token
        self.tokenizer = koGPT2_TOKENIZER    # 데이터를 불러오는 부분

    def __len__(self):  # chatbotdata 의 길이를 리턴한다.
        return len(self._data) # 데이터의 길이를 리턴한다.
    
    def __getitem__(self, idx):  # 로드한 챗봇 데이터를 차례차례 DataLoader로 넘겨주는 메서드
        turn = self._data.iloc[idx] # 챗봇 데이터의 인덱스를 가져온다.
        q = turn["Q"]  # 질문을 가져온다.
        q = re.sub(r"([?.!,])", r" ", q)  # 구둣점들을 제거한다.

        a = turn["A"]  # 답변을 가져온다.
        a = re.sub(r"([?.!,])", r" ", a)  # 구둣점들을 제거한다.

        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)  
        # 질문 + 감정 # 질문을 토크나이징한다.
        q_len = len(q_toked) # 질문의 길이를 구한다.

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos) 
        # 대답 + 끝 # 대답을 토크나이징한다.
        a_len = len(a_toked) # 대답의 길이를 구한다.

        if q_len > self.max_len: # 질문의 길이가 최대길이보다 크면
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로 
                q_len = len(q_toked)                         #질문의 길이를 구한다.
                a_len = self.max_len - q_len                 #답변의 길이를 최대길이 - 질문길이  
            a_toked = a_toked[:a_len]                        #답변의 길이만큼 대답을 가져온다.
            a_len = len(a_toked)                             #답변의 길이를 구한다.

        if q_len + a_len > self.max_len:                #질문길이 + 답변길이가 최대길이보다 크면
            a_len = self.max_len - q_len        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:       #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로 
                q_len = len(q_toked)                        # 질문의 길이를 구한다.
                a_len = self.max_len - q_len              #답변의 길이를 최대길이 - 질문길이
            a_toked = a_toked[:a_len]               #답변의 길이만큼 대답을 가져온다.
            a_len = len(a_toked)                    # 답변의 길이를 구한다.
            
# 답변 labels = [mask, mask, ...., mask, ..., <bos>,..답변.. <eos>, <pad>....] 
# 답변의 길이만큼 마스크를 추가한다.
        labels = [self.mask,] * q_len + a_toked[1:] # 마스크를 질문길이만큼 넣어준다.

        # mask = 질문길이 0 + 답변길이 1 + 나머지 0
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len) # 답변의 길이만큼 1을 넣어준다.
        # 답변 labels을 index 로 만든다.
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels) # 답변을 index로 변환한다.    
        # 최대길이만큼 PADDING
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        # 질문 + 답변을 index 로 만든다.    
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked) # 질문과 답변을 index로 변환한다.
        # 최대길이만큼 PADDING
        while len(token_ids) < self.max_len: # 질문길이 + 답변길이만큼 넣어준다.
            token_ids += [self.tokenizer.pad_token_id] # PADDING
 
        #질문+답변, 마스크, 답변
        return (token_ids, np.array(mask), labels_ids) # 답변을 index로 변환한다.
    
def collate_batch(batch):   #  batch를 받아서 배치를 만든다.
    data = [item[0] for item in batch]   # batch에서 data만 뽑아서 배열로 만든다.
    mask = [item[1] for item in batch]   # batch에서 mask만 뽑아서 배열로 만든다.
    label = [item[2] for item in batch]  # batch에서 label만 뽑아서 배열로 만든다.
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label) # 배열로 만든다.
    
train_set = ChatbotDataset(Chatbot_Data, max_len=40)

#윈도우 환경에서 num_workers 는 무조건 0으로 지정, 리눅스에서는 2
train_dataloader = DataLoader(train_set, batch_size=32, num_workers=0, shuffle=True, collate_fn=collate_batch,)
            
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_set = ChatbotDataset(Chatbot_Data, max_len=40)
#윈도우 환경에서 num_workers 는 무조건 0으로 지정, 리눅스에서는 2
train_dataloader = DataLoader(train_set, batch_size=32, num_workers=0, shuffle=True, collate_fn=collate_batch,)

model.to(device)
model.train()

learning_rate = 3e-5
criterion = torch.nn.CrossEntropyLoss(reduction="none")  # 기준
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoch = 10
Sneg = -1e18                 
            
print ("start")
for epoch in range(epoch):
    for batch_idx, samples in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids, mask, label = samples
        out = model(token_ids)
        out = out.logits      #Returns a new tensor with the logit of the elements of input
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)   # unsqueeze함수는 squeeze함수의 반대로 1인 차원을 생성하는 함수이다. 그래서 어느 차원에 1인 차원을 생성할 지 꼭 지정해주어야한다.
        mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
        loss = criterion(mask_out.transpose(2, 1), label)
        # 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
        avg_loss = loss.sum() / mask.sum()
        avg_loss.backward()
        # 학습 끝
        optimizer.step()
print ("end")

sent = '0'
with torch.no_grad():
    while 1:
        q = input("user > ").strip()   # 인자로 전달된 문자를 String의 왼쪽과 오른쪽에서 제거합니다.
        if q == "quit":  
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
        print("Chatbot > {}".format(a.strip()))



        
