#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 07:47:34 2017

@author: jshliu

Util scripts
"""

import os
import pandas as pd
import numpy as np
import pickle
import pprint
import tarfile
import copy
import re
import torch
from sklearn.feature_extraction.text import CountVectorizer
import pickle


def readTarFile(f_tar, file, toString = True):
    """
    Given key = [NoteID, NoteCSNID, LineID], output string
    Assume key is unique
    f_tar, dictNotes are output from the tarFile2Dict function
    """
    f = f_tar.extractfile(file)
    if f:
        content = f.read()
        if toString:
            content = content.decode("utf-8") 
    else:
        content = ''
    return content

def splitSentence(content):
    """
    Given block of text, split into sentence
    Output: list of sentences
    """
    # Multiple space to single space, remove separators like - and _
    if pd.notnull(content):
        content = re.sub('\s*\t\t\t', ' ', content)
        content = re.sub('--+|==+|__+', ' ', content)
        content = re.sub('\.\s+', '. ',content)
        content = re.sub(':\s+', ': ',content)
        content = re.sub('\s+\[\*', ' [*', content)
        content = re.sub(' \s+', '. ',content)
        lsS = content.split('. ')
    else:
        lsS = []
    return lsS


def update(s):
    """
    - replace number to <num> (keep number right after text, as typically are certain clinical names)
    - replace time to <time>
    - add space before/after non-character
    """
    s = re.sub('\d+:\d+(:\d+)?\s*((a|A)|(p|P))(m|M)(\s*est|EST)?', ' time ', s)
    s = re.sub('( |^|\(|:|\+|-|\?|\.|/)\d+((,\d+)*|(\.\d+)?|(/\d+)?)', ' num ', s) # cases like: 12,23,345; 12.12; .23, 12/12;
    s = re.sub(r'([a-zA-Z->])([<\),!:;\+\?\"])', r'\1 \2 ', s)
    s = re.sub(r'([\(,!:;\+>\?\"])([a-zA-Z<-])', r' \1 \2', s)
    s = re.sub('\s+', ' ', s)
    return s

def replcDeid(s):
    """
    replace de-identified elements in the sentence (date, name, address, hospital, phone)
    """
    s = re.sub('\[\*\*\d{4}-\d{2}-\d{2}\*\*\]', 'date', s)
    s = re.sub('\[\*\*.*?Name.*?\*\*\]', 'name', s)
    s = re.sub('\[\*\*.*?(phone|number).*?\*\*\]', 'phone', s)
    s = re.sub('\[\*\*.*?(Hospital|Location|State|Address|Country|Wardname|PO|Company).*?\*\*\]', 'loc', s)
    s = re.sub('\[\*\*.*?\*\*\]', 'deidother', s)
    return s

def tag_negation( doc ):

    from nltk.sentiment.util import mark_negation
    return ' '.join( mark_negation(doc.split()) )

def cleanString(s, lower = True):
    s = replcDeid(s)
    s = update(s)
    if lower:
        s = s.lower()
    return s


def replaceContractions(s):
    contractions = ["don't","wouldn't","couldn't","shouldn't", "weren't", "hadn't" , "wasn't", "didn't" , "doesn't","haven't" , "isn't","hasn't"]
    for c in contractions:
        s = s.replace( c, c[:-3] +' not')
    return s

def preprocess_string(s):
    s = cleanString(s, True)
    s = replaceContractions(s)
    return s


def cleanNotes(content):
    """
    Process a chunk of text 
    """
    lsOut = []
    content = str(content)
    if len(content) > 0:
        lsS = splitSentence(content)
        for s in lsS:
            if len(s) > 0:
                s = cleanString(s, lower = True)
                lsOut.append(s)
        out = ' '.join(lsOut)
    else:
        out = ''
    return out



def load_star_space(fn):
    #ss = pd.read_csv(fn,sep='\t')
    ss =  pd.read_csv(fn,sep='\t', quoting=3, header= None)

    keys= list(ss.iloc[:,0])
    keys= dict([ (k,i) for i,k in enumerate(keys)])
    params = torch.from_numpy(np.array( ss.iloc[:,1:]))

    return keys, params

def stopwords():
    
    return pickle.load(open('./data/stop_words.p','rb'))


def build_vocab(text, negation = False, max_df = .7 ,  max_features = 20000, vecPath = '/ifs/data/razavianlab/ehr_ssp_embedding/word2CurDiag_ge5_5.tsv'):
    '''
    Fit vocabulary and create PubMed w2v matrix

    :param text: list of documents for creating vocabulary
    :return: embedding matrix and vectorizer

    TODO: expose parameters
    '''

    import torchwordemb

    #load w2v
    #w2v_vocab, vec = torchwordemb.load_word2vec_bin("./data/PubMed-and-PMC-w2v.bin")
    w2v_vocab , vec = load_star_space(vecPath)

    #vect = CountVectorizer(stop_words = 'english',max_df = max_df,  max_features = max_features)
    vect = CountVectorizer(stop_words = stopwords(),max_df = max_df,  max_features = max_features)

    vect.fit(text)

    no_embedding = [ k for k in vect.vocabulary_.keys() if k not in w2v_vocab ]
    print("No Embeddings for: ")
    print(len(no_embedding))


    vocab = dict([ (k, w2v_vocab[k])  for k in vect.vocabulary_.keys() if k in w2v_vocab])
    new_vocab = dict([ (k,i+1) for i,k in enumerate(vocab.keys()) ])

    embedding = torch.zeros(len(new_vocab)+1, vec.size()[1])

    for k,i in new_vocab.items():

        embedding[i] = vec[vocab[k]]

    if negation:
        n_emb = embedding.size()[0]
        neg_emb = -1 * embedding
        embedding = torch.cat( [embedding, neg_emb],0)
        
        for k,v in new_vocab.items():
            new_vocab[k +'_NEG'] = v +n_emb 

    vect.vocabulary_ = new_vocab

    return embedding, vect


def pad_doc(seq, max_len, n):

    padded_seq = torch.zeros(n , max_len)
    
    start = 0 if len(seq) >=  n else n - len(seq)

    for i,s in enumerate(seq):

        if len(s) > max_len:
            padded_seq[start + i] = torch.Tensor(s[:max_len]).long()
        else:
            if len(s) ==0:
                continue

            padded_seq[start + i , -len(s): ] = torch.Tensor(s).long()

    return padded_seq



def prepare( text, vectorizer , max_len ,unique = False ):

    vocab = vectorizer.vocabulary_
    tokenizer = vectorizer.build_tokenizer()

    if unique:
        seq = [ list(set( [  vocab[y] for y in tokenizer(x) if y in vocab])) for x in text ]
    else:
        seq = [ [  vocab[y] for y in tokenizer(x) if y in vocab] for x in text]
    

    lengths = np.array([ len(s) for s in seq])

    print("Average Sequnce Length: " , lengths.mean())
    print("90% Length: " , np.percentile(lengths, 90))

    padded_seq = pad_doc(seq, max_len, len(seq))

    return padded_seq


def sentence_prepare( text, vectorizer , sent_len , doc_len ,unique = False):

    #from nltk.tokenize import sent_tokenize
    from segtok.segmenter import split_multi

    vocab = vectorizer.vocabulary_
    tokenizer = vectorizer.build_tokenizer()

    #text = [sent_tokenize(doc) for doc in text  ]
    text = [list(split_multi(doc)) for doc in text]

    seq = []
    sent_l = []
    doc_l = []
    for doc in text:
        doc_tok = []
        for sent in doc:
            sent_toks = [vocab[y] for y in tokenizer(sent) if y in vocab]             
            doc_tok.append(sent_toks)
            sent_l.append(len(sent_toks))

        seq.append(doc_tok)
        doc_l.append(len(doc_tok))

    sent_l = np.array(sent_l)
    doc_l = np.array(doc_l)

    print("Average Sent Length: " , sent_l.mean())
    print("90% Length: " , np.percentile(sent_l, 90))

    print("Average Doc Length: " , doc_l.mean())
    print("90% Length: " , np.percentile(doc_l, 90))

    #sent_len = np.percentile(sent_l, 90)
    #doc_len = np.percentile(doc_l, 90)

    padded_docs = torch.zeros(len(seq) , doc_len , sent_len)

    for i, _doc in enumerate(seq):

        if len(_doc) > doc_len:
            _doc = _doc[:doc_len]
            padded_seq = pad_doc(_doc, sent_len, len(_doc))
        else:
            if len(_doc) ==0:
                continue

            padded_seq = pad_doc(_doc, sent_len, doc_len)

        padded_docs[i] = padded_seq

    return padded_docs


#w2v_vocab , vec = load_star_space('/ifs/data/razavianlab/ehr_ssp_embedding/word2CurDiag_ge3.tsv')

#================= Math functions ====================
def softmax(x):

    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x,axis=1),1))
    return e_x / np.expand_dims(e_x.sum(axis=1),1)
