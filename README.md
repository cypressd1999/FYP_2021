# Financial Fraud Detection using Text Mining

## Acknowledgement
This project is developed based on the work completed by Wee Tee Soh. The original work can be found:
https://github.com/plkmo/BERT-Relation-Extraction

## Dataset
Unzip original_data.zip to get the dataset for pre-training ```cnn.txt``` and the original training dataset.

## Training
### Pre-training:
To pre-train BERT with plain text data. The default dataset is text data from cnn news.
```python
python3 main_pretraining.py
```

### Fine-tuning:
To fine-tune BERT with annotated data. Followed by --test_data=path and --train_data=path. The default training and testing dataset is the original dataset.

### Inference (--infer=1)
To infer a sentence, you can annotate entity1 & entity2 of interest within the sentence with their respective entities tags [E1], [E2]. 
Example:
```bash
Type input sentence ('quit' or 'exit' to terminate):
[E1] Ajay [/E1] gained entrepreneurial experience from a young age as part of a multi-generational family business named [E2] Ashlin BPG Marketing [/E2]. Ashlin supplies high-end leather accessories to the e-commerce and promotional product industries.

Predicted:  Person-Experience(e1,e2) 
```

To infer a full paragraph without annotating, set --from-file=1 and the sent_reader.py will read the default data file input.txt and output a graph of relations. Spacy NLP is used to extract entities for auto-annotation.
Example:
```python
python3 main_task.py --train=0 --infer=1 --from_file=1
```

Data in input.txt:
```bash
Ajay gained entrepreneurial experience from a young age as part of a multi-generational family business named Ashlin BPG Marketing. Ashlin supplies high-end leather accessories to the e-commerce and promotional product industries.
```

Output:
```bash
Relations extracted:
[(' Ajay ', ' Ashlin BPG Marketing ', 'Person-Experience(e1,e2)'), (' Ashlin BPG Marketing ', ' Ajay ', 'parent-subsidiary(e1,e2)'), (' Ajay ', ' Ashlin ', 'Person-Experience(e1,e2)'), (' Ashlin ', ' Ajay ', 'parent-subsidiary(e1,e2)'), (' Ashlin BPG Marketing ', ' Ashlin ', 'parent-subsidiary(e1,e2)'), (' Ashlin ', ' Ashlin BPG Marketing ', 'parent-subsidiary(e1,e2)'), ('Ajay', 'Ashlin BPG Marketing', 'Person-Experience(e1,e2)'), ('Ajay', 'Ashlin', 'Person-Experience(e1,e2)'), ('Ashlin BPG Marketing', 'Ajay', 'parent-subsidiary(e1,e2)'), ('Ashlin', 'Ajay', 'parent-subsidiary(e1,e2)')]
```
