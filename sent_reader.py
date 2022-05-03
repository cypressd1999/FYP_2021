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
from argparse import ArgumentParser

def read_input(path, max_length=5000):
    '''
    Input: Path to the input file
    Output: annotated sentences: [E1][/E1], [E2][/E2]
    '''
    with open(path, "r", encoding="utf8") as f:
            text = f.readlines()

#     text = process_textlines(text)
    print("Number of paragraph: %d" % len(text))
    
    
    paragraphs = text
    entities = []
        
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
        
        e1 = result[1].strip()
        e2 = result[2].strip()
        if e1 not in entities:
            entities.append(e1)
        if e2 not in entities:
            entities.append(e2)
        
    for e1 in entities:
        for e2 in entities:
            if e1 != e2:
                for sent in paragraphs:
                    if e1 in sent and e2 in sent:
                        s = sent
                        s = s.replace(e1, '[E1] ' + e1 + ' [/E1]', 1).replace(e2, '[E2] '+e2+' [/E2]', 1)
                        if (s.find("[E1]") - s.find("[/E2]")) * (s.find("[/E1]") - s.find("[E2]")) > 0:
                            if s.find("E1")!=-1 and s.find("E2")!=-1:
                                sentences.append(s)
                
    print("\nFinished annotating!")        
    print("Number of sentences with annotation: %d" % len(sentences))
    print("")
    return sentences
  
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, default="./input.txt", help="input file path for plain text input")
    args = parser.parse_args()
    read_input(args.data)
