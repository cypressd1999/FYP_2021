"""
Created on Wed Feb 27 2022
@author: david
"""
import os
import re
import spacy
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import logging
# import preprocessing_funcs from src folder
from src.preprocessing_funcs import process_textlines
from src.preprocessing_funcs import create_pretraining_corpus

def read_input(path, max_length=5000):
    '''
    Input: Path to the input file
    Output: annotated sentences: [E1][/E1], [E2][/E2]
    '''
    with open(path, "r", encoding="utf8") as f:
            text = f.readlines()

#     text = process_textlines(text)
    print("Number of paragraph: %d" % len(text))
    
    # to store the sentences split by "."
    paragraphs = []
    for line in text:
        paragraphs += line.split(".")
        
#     num_chunks = math.ceil(len(text)/max_length)
#     print("Splitting into %d max length chunks of size %d" % (num_chunks, max_length))
#     text_chunks = (text[i*max_length:(i*max_length + max_length)] for i in range(num_chunks))

    text_chunks = (text[i] for i in range(len(text)))
    
    D = []
    print("Loading Spacy NLP...")
    nlp = spacy.load("en_core_web_lg")

    for text_chunk in tqdm(text_chunks, total=len(text)):
        D.extend(create_pretraining_corpus(text_chunk, nlp, window_size=40))
    
    sentences = []
    for result in D:
        sentence = result[0][0]
        p1 = result[0][1]
        p2 = result[0][2]
        sentence1 = sentence[:p1[0]] + ['[E1] '] + sentence[p1[0]:p1[1]] + [' [/E1]'] + sentence[p1[1]:p2[0]] + ['[E2] '] + sentence[p2[0]:p2[1]] + [' [/E2]'] + sentence[p2[1]:] 
        sentence1 = " ".join(sentence1)
        if sentence1.find("[E1]") !=-1: # assert every sentence are annotated
            sentences.append(sentence1)
        
        sentence2 = sentence[:p1[0]] + ['[E2] '] + sentence[p1[0]:p1[1]] + [' [/E2]'] + sentence[p1[1]:p2[0]] + ['[E1] '] + sentence[p2[0]:p2[1]] + [' [/E1]'] + sentence[p2[1]:] 
        sentence2 = " ".join(sentence2)
        if sentence2.find("[E1]") !=-1: # assert every sentence are annotated
            sentences.append(sentence2)
        
        e1 = result[1]
        e2 = result[2]
        for sent in paragraphs:
            if e1 in sent and e2 in sent:
                s1 = sent
                s1 = s1.replace(e1, '[E1] ' + e1 + ' [/E1]', 1).replace(e2, '[E2] '+e2+' [/E2]', 1)
                if s1.find("E1")!=-1 and s1.find("E2")!=-1:
                    sentences.append(s1)
                s2 = sent
                s2 = s2.replace(e1, '[E2] ' + e1 + ' [/E2]', 1).replace(e2, '[E1] '+e2+' [/E1]', 1)
                if s2.find("E1")!=-1 and s2.find("E2")!=-1:
                    sentences.append(s2)
                
    print("\nFinished annotating!")        
    print("Number of sentences with annotation: %d" % len(sentences))
    print("")
    return sentences
  
if __name__ == '__main__':
    path = 'input.txt'
    read_input(path)
