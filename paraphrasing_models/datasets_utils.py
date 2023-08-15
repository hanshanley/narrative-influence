import csv

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from torch import optim
from datasets import load_dataset

def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


class ParaphraseDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer =AutoTokenizer.from_pretrained(args.tokenizer,use_fast=False,cache_dir = 'cache')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        text1 = [x[0] for x in data]
        text2 = [x[1] for x in data]
        labels = [int(x[2]) for x in data]
        queryEncdoing = self.tokenizer(text1, return_tensors='pt', padding=True, truncation=True,max_length=256)
        contextEncoding = self.tokenizer(text2, return_tensors='pt', padding=True, truncation=True,max_length=256)

        token_ids = torch.LongTensor(queryEncdoing['input_ids'])
        attention_mask = torch.LongTensor(queryEncdoing['attention_mask'])
        #token_type_ids = torch.LongTensor(queryEncdoing['token_type_ids'])

        token_ids2 = torch.LongTensor(contextEncoding['input_ids'])
        attention_mask2 = torch.LongTensor(contextEncoding['attention_mask'])
        #token_type_ids2 = torch.LongTensor(contextEncoding['token_type_ids'])
        labels = torch.LongTensor(labels)

        return (token_ids, attention_mask,
                token_ids2, attention_mask2,labels)

    def collate_fn(self, all_data):
        (token_ids,  attention_mask,
         token_ids2, attention_mask2,labels) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'attention_mask_2': attention_mask2,
                'labels': labels
            }
        return batched_data
    
    
class ParaphraseTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,use_fast=False,cache_dir = 'cache')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        text1 = [x[0] for x in data]
        text2 = [x[1] for x in data]
        labels = [x[2] for x in data]

        queryEncdoing = self.tokenizer(text1, return_tensors='pt', padding=True, truncation=True,max_length=256)
        contextEncoding = self.tokenizer(text2, return_tensors='pt', padding=True, truncation=True,max_length=256)

        token_ids = torch.LongTensor(queryEncdoing['input_ids'])
        attention_mask = torch.LongTensor(queryEncdoing['attention_mask'])

        token_ids2 = torch.LongTensor(contextEncoding['input_ids'])
        attention_mask2 = torch.LongTensor(contextEncoding['attention_mask'])
        labels = torch.FloatTensor(labels)


        return (token_ids, attention_mask,
                token_ids2, attention_mask2,labels)

    def collate_fn(self, all_data):
        (token_ids,  attention_mask,
         token_ids2, attention_mask2,labels) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'attention_mask_2': attention_mask2,
                'labels':labels,
            }
        return batched_data

class ParaphraseInferenceDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,use_fast=False,cache_dir = 'cache')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        labels = [str(x[0]) for x in data]
        labels2 = [str(x[1]) for x in data]
        labels3 = [str(x[2]) for x in data]
        text1 = [x[3] for x in data]
        text2 = [x[4] for x in data]

        queryEncdoing = self.tokenizer(text1, return_tensors='pt', padding=True, truncation=True,max_length=256)
        contextEncoding = self.tokenizer(text2, return_tensors='pt', padding=True, truncation=True,max_length=256)

        token_ids = torch.LongTensor(queryEncdoing['input_ids'])
        attention_mask = torch.LongTensor(queryEncdoing['attention_mask'])

        token_ids2 = torch.LongTensor(contextEncoding['input_ids'])
        attention_mask2 = torch.LongTensor(contextEncoding['attention_mask'])


        return (token_ids, attention_mask,
                token_ids2, attention_mask2,labels,labels2,labels3)

    def collate_fn(self, all_data):
        (token_ids,  attention_mask,
         token_ids2, attention_mask2,labels,labels2,labels3) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'attention_mask_2': attention_mask2,
                'labels':labels,
                'labels2':labels2,
                'labels3':labels3
            }
        return batched_data


import csv
def load_paraphrase_data(file_name, split='train'):
    num_labels = set()
    data = []
    with open(file_name, 'r') as fp:
        csv_reader = csv.reader(fp,delimiter= '\t')
        for row in csv_reader:
            data.append((row[0],row[1],int(row[2])))
    return data

def load_paraphrase_inference_data(file_name, split='train'):
    num_labels = set()
    data = []
    with open(file_name, 'r') as fp:
        csv_reader = csv.reader(fp,delimiter= '\t')
        for row in csv_reader:
            data.append((row[0],row[2],row[3],row[1],row[4]))
    return data
