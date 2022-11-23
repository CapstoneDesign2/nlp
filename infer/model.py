import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts

from pytorch_lightning import LightningModule

from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer, BertModel

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import hamming_loss

import re
import emoji
from soynlp.normalizer import repeat_normalize

"""# Model 만들기 with Pytorch Lightning"""

THRESHOLD = 0.5

class Model(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters() # 이 부분에서 self.hparams에 위 kwargs가 저장된다.
        self.LABEL = None
        self.model = AutoModelForSequenceClassification.from_pretrained(self.hparams.pretrained_model, return_dict=True, num_labels=5)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.pretrained_tokenizer
            if self.hparams.pretrained_tokenizer
            else self.hparams.pretrained_model
        )

    def forward(self, input_ids, labels=None, **kwargs):
        #forward에 인자 넘기고 싶으면 / self 있는 곳 들에서 인자 넘겨주면 된다.
        '''
        TO DO
        여긴 단일 모델
        '''
        return self.model(input_ids=input_ids, labels=labels)

    def step(self, batch, batch_idx):
        data, labels = batch
    
        output = self(input_ids=data, labels=labels)
        
        loss = output.loss
        logits = output.logits

        y_true = labels.detach().cpu().numpy()
        y_pred = logits.detach().cpu().numpy()

        return {
            'loss':  loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def epoch_end(self, outputs, state='train'):
        
        loss = torch.tensor(0, dtype=torch.float)
        
        # loss 더하기
        for i in outputs:
            loss += i['loss'].cpu().detach()
        loss = loss / len(outputs)
        
        y_true = []
        y_pred = []
        # 이유는 모르겠지만 이렇게 하면 된다.
        for i in outputs:
            for true in i['y_true']:
                y_true.append(np.array([int(x) for x in true]))

            for pred in i['y_pred']:
                y_pred.append(np.array([1 if x > THRESHOLD else 0 for x in pred]))
        
        print(f'[Epoch {self.trainer.current_epoch} {state.upper()}] Loss: {loss}')

        
        #print(y_true)
        #print(y_pred)
        #print(self.LABEL)

        total_acc = accuracy_score(y_true, y_pred)
        total_prec= precision_score(y_true, y_pred, average='macro')
        total_rec = recall_score(y_true, y_pred, average='macro')
        total_prec = f1_score(y_true, y_pred, average='macro')
        hamming = hamming_loss(y_true, y_pred)

        # total_prec= precision_score(y_true, y_pred, labels=self.LABEL, average='macro')
        # total_rec = recall_score(y_true, y_pred, labels=self.LABEL, average='macro')
        # total_prec = f1_score(y_true, y_pred, labels=self.LABEL, average='macro')
        # hamming_loss = hamming_loss(y_true, y_pred, labels=self.LABEL, average='macro')

        self.log(state+'_loss', float(loss), on_epoch=True, prog_bar=True)
        self.log(state+'_acc', float(total_acc), on_epoch=True, prog_bar=True)
        self.log(state+'_prec', float(total_prec), on_epoch=True, prog_bar=True)
        self.log(state+'_rec', float(total_rec), on_epoch=True, prog_bar=True)
        self.log(state+'_hamming', float(hamming), on_epoch=True, prog_bar=True)
        
        #file close
        #result.close()
        # hamming loss 돌려줄까?
        return {'loss': loss}
    
    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, state='train')

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, state='val')

    def configure_optimizers(self):
        if self.hparams.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer == 'SWA':
            from torchcontrib.optim import SWA
            optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
            #optimizer = SWA(swa_lr=self.hparams.lr)
        else:
            raise NotImplementedError('Only AdamW , AdamP and SWA is Supported!')
        if self.hparams.lr_scheduler == 'cos':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)
        elif self.hparams.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    def clean(self, x):
        emojis = ''.join(emoji.EMOJI_DATA.keys()) 
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        x = pattern.sub(' ', x)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        return x

    def encode(self, x, **kwargs):
        return self.tokenizer.encode(
                self.clean(str(x)),
                padding='max_length',
                max_length=self.hparams.max_length,
                truncation=True,
                **kwargs,
        )

    def preprocess_dataframe(self, df):
        df['댓글'] = df['댓글'].map(self.encode)
        # 문장은 input_ids 로 return 해주고 
        #print("리턴 타입 텐서 아니라 list다!")
        
        return df

    def dataloader(self, path, shuffle=False):
        df = self.read_data(path)
        df = self.preprocess_dataframe(df)
        LABEL_COLUMNS = df.columns.tolist()[1:]
        self.LABEL = LABEL_COLUMNS

        #일단 df에서 다 0 아니면 1로 만들어준다
        for i in LABEL_COLUMNS:
            df[i] = df[i].map(lambda x : 1 if x > 0 else 0)

        dataset = TensorDataset(
            torch.tensor(df['댓글'].to_list(), dtype=torch.long),
            torch.tensor(df[LABEL_COLUMNS].values.tolist(), dtype=torch.float),
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size * 1 if not self.hparams.tpu_cores else self.hparams.tpu_cores,
            shuffle=shuffle,
            num_workers=self.hparams.cpu_workers,
        )

    def train_dataloader(self):
        return self.dataloader(self.hparams.train_data_path, shuffle=True)

    def val_dataloader(self):
        return self.dataloader(self.hparams.val_data_path, shuffle=False)
