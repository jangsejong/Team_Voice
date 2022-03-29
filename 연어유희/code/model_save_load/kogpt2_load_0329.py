import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel

Q_TKN = "<usr>"  # 질문
A_TKN = "<sys>"  # 대답
BOS = '</s>'   # 시작
EOS = '</s>'   # 끝
MASK = '<unused0>'
SENT = '<unused1>'  # 감정
PAD = '<pad>' 
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, unk_token="<unk>", pad_token=PAD, mask_token=MASK,)


# torch.save(model.state_dict(), PATH + 'model.pt')
PATH1 = 'C:\\Users\\비트캠프\\Desktop\\챗봇개인화데이터\\'

model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
model.load_state_dict(torch.load(PATH1 + 'Data50000_EPOCH5.pt'))
model.eval()

sent = '0'
with torch.no_grad():
    while 1:
        q = input("나 > ").strip()   # 인자로 전달된 문자를 String의 왼쪽과 오른쪽에서 제거합니다.
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
        print("유희 > {}".format(a.strip()))