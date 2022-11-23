import os
import sys

import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything


# loop = True;

# inferTest 불러오기

from model import Model
from clean import clean

MODEL_PATH = "./checkpoints/*.ckpt"
THRESHOLD = 0.5

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
    return model(**model.tokenizer(x, return_tensors='pt', truncation=True))
def judge(sentence):
    sentence=str(sentence)
    LABEL_COLUMNS= ['effective', 'clean', 'tasty', 'vibe', 'kind']
    if sentence == "":
        #for l, v in zip(LABEL_COLUMNS, [0, 0, 0, 0, 0]):
        #    print(l, v)
        
        return [0, 0, 0, 0, 0]
    else:
        test_prediction = infer(clean(sentence))
        output = torch.sigmoid(test_prediction.logits)
        output = output.detach().flatten().numpy()
        #for l, v in zip(LABEL_COLUMNS, output):
        #    print(l, v)

        real_output = [1 if x > THRESHOLD else 0 for x in output]

        return real_output
        
    
if __name__ == "__main__":
    main()