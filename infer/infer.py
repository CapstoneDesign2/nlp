import os
import sys

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything


# loop = True;

# inferTest 불러오기
root_path = sys.path[0]
model_path = os.path.join(root_path, '../models')
sys.path.append(model_path)
# inferTest 불러오기

from model import Model
from clean import clean

MODEL_PATH = "./checkpoints/*.ckpt"

from glob import glob

latest_ckpt = sorted(glob(MODEL_PATH))[0]
#model = trainning.Model.load_from_checkpoint(latest_ckpt, hparams_file=HPARAMS_PATH)
model = Model.load_from_checkpoint(latest_ckpt)

model.eval()
#map location 해주면 환경 바뀌어도 가능
def main():
    while True:
        sentence = input("문장을 입력하시오! ")
        judge(sentence=sentence)
    
def infer(x):
    return model(**model.tokenizer(x, return_tensors='pt'))
def judge(sentence):
    sentence=str(sentence)
    if sentence == "":
        print("빈 문장")
    else:
        LABEL_COLUMNS=["가성비", "청결", "맛", "분위기", "친절"]
        test_prediction = infer(clean(sentence))
        output = torch.sigmoid(test_prediction.logits)
        output = output.detach().flatten().numpy()
        for i in zip(LABEL_COLUMNS, output):
            print(f"{i[0]} : {i[1]}")
        return 0
    
if __name__ == "__main__":
    main()