import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score, precision_score, recall_score,mean_squared_error
from transformers import AutoTokenizer, AutoModel



# change it with respect to the original model
from tqdm import tqdm
from datasets_utils import load_paraphrase_data,ParaphraseDataset,load_paraphrase_inference_data,ParaphraseInferenceDataset
#from InstructorEmbedding import INSTRUCTOR


TQDM_DISABLE=False
# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


MPNET_HIDDEN_SIZE = 768
N_CLASSES = 2

TYPE = 'microsoft/mdeberta-v3-base'
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class ParaphraseClassifier(torch.nn.Module):
    '''
    This module performs paraphrasing of mulitingual passages between Chinese and other languages 
    using a mDeBERTa base model.
    '''
    def __init__(self, config):
        super(FactCheckClassifier, self).__init__()
        self.model =  AutoModel.from_pretrained(TYPE,cache_dir = 'cache')

        for  param in self.model.parameters():
            param.requires_grad = True
        
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.average_factcheck_head = torch.nn.Linear(MPNET_HIDDEN_SIZE*2,MPNET_HIDDEN_SIZE)
        self.average_factcheck_head.requires_grad = True
        self.linear = torch.nn.Linear(MPNET_HIDDEN_SIZE, N_CLASSES)
        self.linear.requires_grad = True
        self.relu = torch.nn.ReLU()
        self.batchnorm = torch.nn.BatchNorm1d(MPNET_HIDDEN_SIZE)



    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                          input_ids_2, attention_mask_2):

        output1 = self.model(input_ids_1, attention_mask_1)['last_hidden_state'][:,0,:]
        output2 = self.model(input_ids_2, attention_mask_2)['last_hidden_state'][:,0,:]
        
        cat_out = torch.cat((output1, output2), -1)
        new_out = self.average_factcheck_head(cat_out)
        new_out= self.relu(new_out)
        new_out = self.batchnorm(new_out)
        new_out = self.dropout(new_out)
        new_out = self.linear(new_out)
        return new_out


def save_model(model, optimizer, args, config, filepath,epoch):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath+str(epoch))
    print(f"save the model to {filepath+str(epoch)}")



def model_eval(dataloader, model, device):
    model = model.eval()  # switch to eval model, will turn off randomness like dropout

    y_true = []
    y_pred = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            b_ids_1, b_mask_1,b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                       batch['attention_mask_1'],batch['token_ids_2'],
                                       batch['attention_mask_2'], batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)

            b_labels = b_labels.to(device)

            logits = model.predict_factcheck(b_ids_1, b_mask_1,b_ids_2, b_mask_2)
            preds = logits.argmax(dim=-1).flatten().cpu().numpy()
            #preds = np.argmax(y_hat, axis=1).flatten()
            b_labels = b_labels.flatten().cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(b_labels)

    accuracy = np.mean(np.array(y_pred) == np.array(y_true))
    f1 = f1_score(y_true, y_pred, average='binary')
    prec = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, prec,recall, y_pred, y_true


def train(args):
    device = torch.device('cuda') 
    # Load data
    # Create the data and its corresponding datasets and dataloader
    print('TRAIN')
    train_gpt_data = load_paraphrase_data( args.train_data, split ='train')
    print('DEV')
    dev_gpt_data = load_paraphrase_data(args.dev_data, split ='dev')

    train_data = ParaphraseDataset(train_gpt_data, args)
    dev_data = ParaphraseDataset(dev_gpt_data, args)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=train_data.collate_fn)
    print('STOP')
    dev_dataloader = DataLoader(dev_data, shuffle=True, batch_size=args.batch_size,
                                    collate_fn=dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = ParaphraseClassifier(config)
    model = model.to(device)

    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0
    best_dev_f1 = 0 

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model = model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
           # model = model.train()
            b_ids_1, b_mask_1,b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                       batch['attention_mask_1'],batch['token_ids_2'],
                                       batch['attention_mask_2'], batch['labels'])

            b_ids_1 = b_ids_1.to(device)
            b_mask_1 = b_mask_1.to(device)
            
            b_ids_2 = b_ids_2.to(device)
            b_mask_2 = b_mask_2.to(device)


            b_labels = b_labels.to(device)


            optimizer.zero_grad()
            logits = model.predict_paraphrase(b_ids_1, b_mask_1,b_ids_2, b_mask_2)
            loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
            num_batches += 1
        train_loss = train_loss / (num_batches)
        print(train_loss)
        dev_acc, dev_f1, dev_prec, dev_recall, *_ = model_eval(dev_dataloader, model, device)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath,epoch)
        print(f"Epoch {epoch}, dev acc :: {dev_acc :.3f}, dev f1 :: {dev_f1 :.3f},  dev prec :: {dev_prec :.3f},  dev recall :: {dev_recall :.3f}")
        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str )
    parser.add_argument("--dev_data", type=str )
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--tokenizer", type=str, default=TYPE)
    # hyper parameters
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=1e-5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    args.filepath = 'NAME OF WEIGHT FILE' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    #train(args)
