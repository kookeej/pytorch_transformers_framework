import argparse
import pickle
from tqdm import tqdm
import gc

import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import DefaultConfig

config = DefaultConfig()



# Tokenizer
def tokenizing(dataset, args):
    """
    This is the preprocessing code for transformers. You can customize it to fit your own dataset.
    Depending on your dataset size and hardware specifications, you can choose one of two versions.
    version 1. A dataset of a size that your hardware can handle.
    version 2. A dataset that your hardware cannot handle.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    sentences= dataset['sentence'].tolist()
    labels = dataset['labels'].tolist()
    length = len(sentences)
    
    """
    version 1. 
    If your dataset is enough for your hardware to handle, use this code.
    """
    tokenized = tokenizer(
        sentences,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=args.max_len
    )  
    
    
    """
    version 2.
    If your dataset is too big to handle, use this code.
    """
    input_ids = []
    attention_mask = []
    
    for i in tqdm(range(0, len(sentences), 256)):
        chunck = sentences[i:i+256]
        tokenized = tokenizer(
            chunck,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=args.max_len
        )
        input_ids.extend(tokenized['input_ids'])
        attention_mask.extend(tokenized['attention_mask'])

    tokenized = {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    
    return tokenized, labels, length




# Dataset
class CustomDataset(Dataset):
    """
    This is the code that creates the dataset format to put in the dataloader.
    """
    def __init__(self, tokenized_dataset, labels, length):
        self.tokenized_dataset = tokenized_dataset
        self.length = length
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.tokenized_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return self.length
    
    
def pro_dataset(dataset, batch_size, args, mode="train"):
    """
    This is the code for tokenizing, creating a custom dataset, and creating a dataloader.
    """
    tokenized, labels, length = tokenizing(dataset, args)
    custom_dataset = CustomDataset(tokenized, labels, length)
    if mode == "train":
        OPT = True
    else:
        OPT = False
    dataloader = DataLoader(
        custom_dataset, 
        batch_size=batch_size,
        shuffle=OPT,
        drop_last=OPT
    )
    return dataloader





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/sample_train.csv', help="train dataset path")
    parser.add_argument('--max_length', type=int, default=256, help="max token length for tokenizing")

    args = parser.parse_args()
        
    dataset = pd.read_csv(args.path)
    train_dataset, valid_test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    valid_dataset, test_dataset = train_test_split(valid_test_dataset, test_size=0.5, random_state=42)
    del valid_test_dataset
    
    print("="*27)
    print("train dataset size: {0:>7}\nvalid dataset size: {1:>7}\ntest dataset size : {2:>7}".format(len(train_dataset), len(valid_dataset), len(test_dataset)))
    print("="*27)
    
    print("Preprocessing dataset...")
    train_dataloader = pro_dataset(train_dataset, config.TRAIN_BATCH, args, mode='train')
    valid_dataloader = pro_dataset(valid_dataset, config.VALID_BATCH, args, mode='train')
    test_dataloader = pro_dataset(test_dataset, config.TEST_BATCH, args, mode='test')
    
    # Save DataLoader with pickle file.
    print("Save DataLoader...")
    gc.collect()
    pickle.dump(train_dataloader, open('data/train_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    gc.collect()
    pickle.dump(valid_dataloader, open('data/valid_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)   
    gc.collect()
    pickle.dump(test_dataloader, open('data/test_dataloader.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    print("Complete!")     
