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


