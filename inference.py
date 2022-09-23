import pandas as pd
import pickle
from tqdm import tqdm
import argparse
from time import sleep
from tqdm import tqdm

import torch

from config import DefaultConfig
from model import CustomModel
from preprocessing import CustomDataset

def inference(model, test_dataloader):
    """
    You can customize this code to fit your task.
    """
    preds_lst = []
    with torch.no_grad():
        print("Inference....")
        model.eval()
        bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        for idx, items in bar:
            sleep(0.1)
            item = {key: val.to(device) for key,val in items.items()}
            outputs = model(**item)
            loss = criterion(outputs, item['labels'].view(-1, 1).float())
            
            preds_lst.append(loss.item())
            
        print(f"Loss: {sum(preds_lst)/len(preds_lst)}")
        logger.info(f"Loss: {sum(preds_lst)/len(preds_lst)}")
    
        

if __name__ == '__main__':
    config = DefaultConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/test_dataloader.pkl', help="test dataloader path")
    parser.add_argument('--model_path', type=str, default='data/model.bin', help="saved model path")

    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Test DataLoader...")
    test_dataloader = pickle.load(open(args.path, 'rb'))
    
    print("Loading saved model...")
    model = CustomModel(config.MODEL_CONFIG)
    model.parameters
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Inference
    inference(model, test_dataloader)
    print("Inference Finish!")
    