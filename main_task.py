#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:40:16 2019

@author: weetee
"""
from src.tasks.trainer import train_and_fit
from src.tasks.infer import infer_from_trained, FewRel
import logging
from argparse import ArgumentParser
# for reading input
from sent_reader import read_input

'''
This fine-tunes the BERT model on SemEval, FewRel tasks
'''

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default='semeval', help='semeval, fewrel')
    # parser.add_argument("--train_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT', \
    #                     help="training data .txt file path")
    parser.add_argument("--train_data", type=str, default='./original_data/TRAIN_FILE.TXT', \
                        help="training data .txt file path")
    # parser.add_argument("--test_data", type=str, default='./data/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT', \
    #                     help="test data .txt file path")
    parser.add_argument("--test_data", type=str, default='./original_data/TEST_FILE_FULL.TXT', \
                        help="test data .txt file path")
    parser.add_argument("--use_pretrained_blanks", type=int, default=0, help="0: Don't use pre-trained blanks model, 1: use pre-trained blanks model")
    parser.add_argument("--num_classes", type=int, default=19, help='number of relation classes')
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--gradient_acc_steps", type=int, default=2, help="No. of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipped gradient norm")
    parser.add_argument("--fp16", type=int, default=0, help="1: use mixed precision ; 0: use floating point 32") # mixed precision doesn't seem to train well
    parser.add_argument("--num_epochs", type=int, default=11, help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.00007, help="learning rate")
    parser.add_argument("--model_no", type=int, default=0, help='''Model ID: 0 - BERT\n
                                                                            1 - ALBERT\n
                                                                            2 - BioBERT''')
    parser.add_argument("--model_size", type=str, default='bert-base-uncased', help="For BERT: 'bert-base-uncased', \
                                                                                                'bert-large-uncased',\
                                                                                    For ALBERT: 'albert-base-v2',\
                                                                                                'albert-large-v2'\
                                                                                    For BioBERT: 'bert-base-uncased' (biobert_v1.1_pubmed)")
    parser.add_argument("--train", type=int, default=1, help="0: Don't train, 1: train")
    parser.add_argument("--infer", type=int, default=1, help="0: Don't infer, 1: Infer")
    parser.add_argument("--from_file", type=int, default=0, help="0: Read input from command line, 1: Read input from file")
    parser.add_argument("--detect", type=int, default=0, help="Detect entities from input sentences.") # this function doesn't work with long sentences. Use sent_reader.py instead 
    
    args = parser.parse_args()
    
    if (args.train == 1) and (args.task != 'fewrel'):
        net = train_and_fit(args)
        
    if (args.infer == 1) and (args.task != 'fewrel'):
        inferer = infer_from_trained(args, detect_entities=True)
        test = "The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor."
        inferer.infer_sentence(test, detect_entities=False)
        test2 = "After eating the chicken, he developed a sore throat the next morning."
        inferer.infer_sentence(test2, detect_entities=True)
        
        if not args.from_file:
          detect = bool(args.detect)
          while True:
              sent = input("Type input sentence ('quit' or 'exit' to terminate):\n")
              if sent.lower() in ['quit', 'exit']:
                  break
              inferer.infer_sentence(sent, detect_entities=detect)
        else:
          results = []
          path = 'input.txt'
          sentences = read_input(path)
          for sentence in sentences:
            result = inferer.infer_sentence(sentence, detect_entities=False) # return a tuple (sentence, predicted relation)
            if result[1] == 'Other':
              continue
            sent = result[0]
            E1 = sent[sent.find('[E1] ')+5:sent.find(' [/E1]')]
            E2 = sent[sent.find('[E2] ')+5:sent.find(' [/E2]')]
            results.append(tuple((E1, E2, result[1])))
          print("Relations extracted:")
          print(results)
          print("")
            
    if args.task == 'fewrel':
        fewrel = FewRel(args)
        meta_input, e1_e2_start, meta_labels, outputs = fewrel.evaluate()
