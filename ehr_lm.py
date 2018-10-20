
import copy
import json
import math
import re
from tqdm import tqdm
import collections
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from analysis import rocstories as rocstories_analysis
from datasets import rocstories
from model_pytorch import DoubleHeadModel, load_openai_pretrained_model
from opt import OpenAIAdam
from text_utils import TextEncoder
from utils import (encode_dataset, iter_data,
                           ResultLogger, make_path)

from loss import MultipleChoiceLossCompute
from model_pytorch import *
import pandas as pd
#from sklearn.cross_validation import train_test_split
from torch.utils.data import *
from skorch.net import *
from loss import *
import util2 as u2
from adam16 import *
from apex.fp16_utils import *
#from apex import amp

#amp_handle = amp.init()

class LM_Dataset():

    def __init__(self,text,batch_size=32):
        self.text = text
        self.idx = 0
        self.batch_size = batch_size
        self.n_batches = len(text) // batch_size + 1

    def next(self):
        batch = self.text[self.idx:self.idx +self.batch_size]
        self.idx += self.batch_size
        return batch

def run_epoch2(train , test):

    train = LM_Dataset(train,batch_size =16)
    test = LM_Dataset(test, batch_size =16)

    opt = OpenAIAdam(dh_model.parameters(),
                           lr=6.25e-5,
                           schedule='warmup_linear',
                           warmup=0.002,
                           t_total= train.n_batches * 3,
                           b1=.9,
                           b2=.999,
                           e=1e-8,
                           l2=0.01,
                           vector_l2=True,
                           max_grad_norm=1)



    #opt = torch.optim.Adam(lr=6.25e-5,params=dh_model.parameters())
    opt = Adam16(lr=6.25e-5,params=dh_model.parameters())
    #opt = torch.optim.SGD(lr=6.25e-5,params=dh_model.parameters())

    opt = FP16_Optimizer(opt,
            static_loss_scale = 1,
            dynamic_loss_scale = False)


    criterion = nn.CrossEntropyLoss(reduce=False)

    L = LangModelLoss(criterion, opt = opt)

    avg_loss_train , avg_loss_test  = 0, 0
    acc_train , acc_test  = 0, 0

    for i in tqdm(range(train.n_batches)):

        data = train.next()
        data , mask = transform_data(data)
        data = torch.from_numpy(data).long()
        mask = torch.from_numpy(mask)

        opt.zero_grad()

        if GPU:
            data = data.cuda()
            mask = mask.cuda().half()

        lm_logits , clf_logits = dh_model(data)

        loss = L( data, mask,  lm_logits= lm_logits, only_return_losses=False)
        print(loss)
        avg_loss_train += loss


    print('Training Loss: ' , avg_loss_train / len(train_loader))

    for i in tqdm(range(test.n_batches)):
        
        data = train.next()
        data , mask = transform_data(data)
        data = torch.from_numpy(data).long()
        mask = torch.from_numpy(mask)

        opt.zero_grad()

        if GPU:
            data = data.cuda()
            mask = mask.cuda().half()


        lm_logits , clf_logits = dh_model(data)
        loss = L( data, mask, lm_logits=lm_logits, only_return_losses=True)

        avg_loss_test += loss

    print('Test Loss: ' , avg_loss_test / len(test_loader))


def run_epoch(train_loader , test_loader):

    opt = OpenAIAdam(dh_model.parameters(),
                           lr=6.25e-5,
                           schedule='warmup_linear',
                           warmup=0.002,
                           t_total= len(train_loader) * 3,
                           b1=.9,
                           b2=.999,
                           e=1e-8,
                           l2=0.01,
                           vector_l2=True,
                           max_grad_norm=1)


    opt = torch.optim.Adam(lr=6.25e-5,params=dh_model.parameters())
   
    print(half)
    
    if half:

        opt = Adam16(lr=6.25e-5,params=dh_model.parameters())

    criterion = nn.CrossEntropyLoss(reduce=False)

    L = LangModelLoss(criterion, opt = opt)

    avg_loss_train , avg_loss_test  = 0, 0
    acc_train , acc_test  = 0, 0

    for (data,mask) , target in tqdm(train_loader):
        opt.zero_grad()

        if GPU:
            data = data.cuda()
            target = target.cuda()
            mask = mask.cuda()#.half()

        if half:
            mask = mask.half()

        lm_logits , clf_logits = dh_model(data)

        loss = L( data, mask,  lm_logits= lm_logits, only_return_losses=False)
        print(loss)
        avg_loss_train += loss


    print('Training Loss: ' , avg_loss_train / len(train_loader))

    for (data,mask) , target in tqdm(test_loader):

        opt.zero_grad()

        if GPU:
            data = data.cuda()
            target = target.cuda()
            mask = mask.cuda()#.half()

        if half:
            mask = mask.half()


        lm_logits , clf_logits = dh_model(data)
        loss = L( data, mask, lm_logits=lm_logits, only_return_losses=True)

        avg_loss_test += loss

    print('Test Loss: ' , avg_loss_test / len(test_loader))

def transform_data(X):

    n_batch = len(X)

    max_len = n_ctx - 2
    max_len = max([len(x) for x in X])

    xmb = np.zeros((n_batch, 1, max_len + 2 , 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 1, max_len + 2), dtype=np.float32)


    start = encoder['_start_']
    
    for i, example in enumerate(X):
        
        x = [start] + example[:max_len] + [clf_token]
        l = len(x)
        xmb[i, 0, :l, 0] = x
        mmb[i, 0, :l] = 1

    #xmb[:, :, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
    xmb[:, :, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + max_len +2)

    return xmb , mmb



if __name__ =='__main__':


    GPU = True
    half = True

    DEFAULT_CONFIG = dotdict({
        'n_embd': 768,
        'n_head': 12,
        'n_layer': 12,
        'embd_pdrop': 0.1,
        'attn_pdrop': 0.1,
        'resid_pdrop': 0.1,
        'afn': 'gelu',
        'clf_pdrop': 0.1})


    args = DEFAULT_CONFIG

    encoder = pickle.load(open('vect.p','rb')).vocabulary_

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #n_gpu = torch.cuda.device_count()
    #print("device", device, "n_gpu", n_gpu)

    text_encoder = TextEncoder()
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    x = pd.read_csv('../notes_small.csv').iloc[:200]
    x['NOTE_TEXT'] = x['NOTE_TEXT'].apply(u2.cleanNotes)

    seq = text_encoder.encode(x['NOTE_TEXT'])
    seq = [ s[:64] if len(s) > 64 else s for s in seq]
    seq = sorted(seq,key=lambda x : len(x))

    #Setup Model
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']

    n_special = 3
    n_ctx = np.array([len(t) for t in seq]).max() + 2
    n_ctx = int(n_ctx)
    
    print(n_ctx)

    vocab = int(n_vocab + n_special + n_ctx)
    dh_model = DoubleHeadModel(args, clf_token, ('classification', 1), vocab, n_ctx)
    load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special)

    if GPU:
        dh_model = dh_model.cuda()#.half()

    if half:
        dh_model = dh_model.half()

    #dh_model = dh_model.type(torch.cuda.HalfTensor)

    #for layer in dh_model.modules():
    #    if isinstance(layer, LayerNorm):
    #        layer.float()


    '''
    split_idx = int(.8 * len(seq))
    shuffle = np.random.permutation(len(seq))
    train_idx = shuffle[:split_idx]
    test_idx = shuffle[split_idx:]

    seq_train = [ seq[i] for i in train_idx]
    seq_test = [ seq[i] for i in test_idx]

    for i in range(10):
        run_epoch2( seq_train  , seq_test)
    '''

    
    seq , mask = transform_data(seq)

    split_idx = int(.8 * seq.shape[0])
    
    shuffle = np.random.permutation(seq.shape[0])
    train_idx = shuffle[:split_idx]
    test_idx = shuffle[split_idx:]
    
    mask_train = mask[train_idx]
    mask_test = mask[test_idx]
    
    seq_train = seq[train_idx]
    seq_test = seq[test_idx]
    
    mask_train = torch.from_numpy(mask_train).float()
    mask_test = torch.from_numpy(mask_test).float()
    
    seq_train = torch.from_numpy(seq_train).long()
    seq_test = torch.from_numpy(seq_test).long()

    y_train = torch.ones(seq_train.shape[0]).long()
    y_test = torch.ones(seq_test.shape[0]).long()

    train_loader = Dataset([seq_train, mask_train ]  , y_train)
    test_loader = Dataset([ seq_test, mask_test ]  , y_test)
    
    train_loader = DataLoader(train_loader,batch_size=16,shuffle=True)
    test_loader = DataLoader(test_loader,batch_size=16,shuffle=False)
   

    for i in range(100):
    
        run_epoch(train_loader , test_loader)
    
