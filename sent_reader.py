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

def read_input(path):
    '''
    Input: Path to the input file
    Output: annotated sentences: [E1][/E1], [E2][/E2]
    '''
    with open(args.pretrain_data, "r", encoding="utf8") as f:
            text = f.readlines()
        
    #text = text[:1500] # restrict size for testing
    text = process_textlines(text)

    logger.info("Length of text (characters): %d" % len(text))
    num_chunks = math.ceil(len(text)/max_length)
    logger.info("Splitting into %d max length chunks of size %d" % (num_chunks, max_length))
    text_chunks = (text[i*max_length:(i*max_length + max_length)] for i in range(num_chunks))

    D = []
    logger.info("Loading Spacy NLP...")
    nlp = spacy.load("en_core_web_lg")

    for text_chunk in tqdm(text_chunks, total=num_chunks):
        D.extend(create_pretraining_corpus(text_chunk, nlp, window_size=40))
    print(D)
  
 
