
import copy
import json
import math
import re
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
#from opt import OpenAIAdam
from text_utils import TextEncoder
from utils import (encode_dataset, iter_data,
                           ResultLogger, make_path)

from loss import MultipleChoiceLossCompute
from model_pytorch import *
import pandas as pd
from sklearn.cross_validation import train_test_split
from torch.utils.data import *
#from skorch.net import *
from loss import *
import util2 as u2


if __name__ =='__main__':


    GPU = True

    
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

    x = pd.read_csv('../notes_small.csv').iloc[:1000]

    x['NOTE_TEXT'] = x['NOTE_TEXT'].apply(u2.cleanNotes)
    x['NOTE_TEXT'].to_csv('clean_notes.csv',index=False)
