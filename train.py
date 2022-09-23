import argparse
import pickle
from tqdm import tqdm
import gc
import numpy as np
import copy
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from config import DefaultConfig
from model import CustomModel
from preprocessing import CustomDataset
from utils import get_criterion, get_optimizer, get_scheduler, seed_everything

from colorama import Fore, Style
b_ = Fore.CYAN
y_ = Fore.YELLOW
g_ = Fore.GREEN
r_ = Fore.RED
sr_ = Style.RESET_ALL

# Settings
config = DefaultConfig()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed_everything(config.SEED)

def request_logger(logger_name):
    logger = logging.getLogger(logger_name)
    if len(logger.handlers) > 0:
        return logger
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler = logging.FileHandler(f'data/test.log')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def train_regressor(train_dataloader, valid_dataloader, args):
    logger.info("="*94)
    logger.info("{0:^23}|{1:^7}|{2:^20}|{3:^20}|{4:^20}".format('datetime',
                                                                'epoch',
                                                                'training loss',
                                                                'validation loss',
                                                                'best loss'))
    logger.info("="*94)
    
    # Load model...
    model = CustomModel(conf=config.MODEL_CONFIG)
    model.parameters
    model.to(device)
    
    # Set criterion, optimizer, scheduler...
    criterion = get_criterion()
    optimizer = get_optimizer(model=model, lr=args.lr)
    scheduler = get_scheduler(optimizer, train_dataloader, args)
    
    train_total_loss = []
    valid_total_loss = []

    best_val_loss = np.inf

    for epoch in range(args.epochs):
        model.train()
        print(f"{y_}[EPOCH {epoch+1}]{sr_}")
        

        # Training loss/accuracy
        train_loss_value = 0
        train_epoch_loss = []

        # Validation loss/accuracy
        valid_loss_value = 0
        valid_epoch_loss = []

        train_bar = tqdm(train_dataloader, total=len(train_dataloader))
        for idx, items in enumerate(train_bar):
            if idx == 2: break
            item = {key: val.to(device) for key, val in items.items()}
            
            optimizer.zero_grad()
            
            outputs = model(**item)
            
            preds = torch.argmax(outputs, dim=-1)
            
            loss = criterion(outputs, item['labels'].view(-1))

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_value += loss.item()
            if (idx + 1) % config.TRAIN_LOG_INTERVAL == 0:
                train_epoch_acc.append(train_batch_acc/config.TRAIN_LOG_INTERVAL)
                train_epoch_loss.append(train_loss_value/config.TRAIN_LOG_INTERVAL)
                
                train_bar.set_description("Loss: {:3f} / {:3f}".\
                    format(train_loss_value/config.TRAIN_LOG_INTERVAL,
                           sum(train_epoch_loss)/len(train_epoch_loss)))

                train_loss_value = 0
                train_total_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))

        with torch.no_grad():
            print(f"{b_}---- Validation ----{sr_}")
            model.eval()
            valid_bar = tqdm(valid_dataloader, total=len(valid_dataloader))
            for idx, items in enumerate(valid_bar):
                if idx == 2: break
                item = {key: val.to(device) for key,val in items.items()}
                outputs = model(**item)

                preds = torch.argmax(outputs, dim=-1)
                loss = criterion(outputs, item['labels'].view(-1))

                valid_loss_value += loss.item()
                if (idx + 1) % config.VALID_LOG_INTERVAL == 0:
                    valid_epoch_loss.append(valid_loss_value/config.VALID_LOG_INTERVAL)
                    
                    valid_bar.set_description("Loss: {:3f} / {:3f}".\
                        format(valid_loss_value/config.VALID_LOG_INTERVAL,
                               sum(valid_epoch_loss)/len(valid_epoch_loss)))

                    valid_loss_value = 0
                    
            
            print("{}Best Loss: {:3f}    |    This epoch Loss: {:3f}{}".format(g_, best_val_loss, (sum(valid_epoch_loss)/len(valid_epoch_loss)), sr_))
            if best_val_loss > (sum(valid_epoch_loss)/len(valid_epoch_loss)):
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f'data/model.bin')
                print(f"{r_}Best Loss Model was Saved!{sr_}")
                best_val_loss = (sum(valid_epoch_loss)/len(valid_epoch_loss))

            valid_total_loss.append(sum(valid_epoch_loss)/len(valid_epoch_loss))
            valid_total_acc.append(sum(valid_epoch_acc)/len(valid_epoch_loss))
            logger.info("{0:^23}|{1:^7}|{2:^20.6f}|{3:^20.6f}|{4:^20.6f}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                                epoch+1,
                                                                                sum(train_epoch_loss)/len(train_epoch_loss),
                                                                                sum(valid_epoch_loss)/len(valid_epoch_loss),
                                                                                best_val_loss))
            logger.info("-"*94)
        print()
    del train_total_loss, valid_total_loss, train_loss_value, train_epoch_loss, valid_loss_value, valid_epoch_loss, train_bar, valid_bar

    
                                              
def train_classifier(train_dataloader, valid_dataloader, args):
    logger.info("="*114)
    logger.info("{0:^23}|{1:^7}|{2:^20}|{3:^20}|{4:^20}|{5:^20}".format('datetime',
                                                                        'epoch',
                                                                        'training loss',
                                                                        'training accuracy',
                                                                        'validation loss',
                                                                        'best loss'))
    logger.info("="*114)
    
    # Load model...
    model = CustomModel(conf=config.MODEL_CONFIG)
    model.parameters
    model.to(device)
    
    # Set criterion, optimizer, scheduler...
    criterion = get_criterion()
    optimizer = get_optimizer(model=model, lr=args.lr)
    scheduler = get_scheduler(optimizer, train_dataloader, args)
    
    train_total_loss = []
    train_total_acc = []
    valid_total_loss = []
    valid_total_acc = []

    best_val_loss = np.inf
    best_val_acc = -1

    for epoch in range(args.epochs):
        model.train()
        print(f"{y_}[EPOCH {epoch+1}]{sr_}")
        

        # Training loss/accuracy
        train_loss_value = 0
        train_epoch_loss = []
        train_batch_acc = 0
        train_epoch_acc = []

        # Validation loss/accuracy
        valid_loss_value = 0
        valid_epoch_loss = []
        valid_batch_acc = 0
        valid_epoch_acc = []

        train_bar = tqdm(train_dataloader, total=len(train_dataloader))
        for idx, items in enumerate(train_bar):
            if idx == 2: break
            item = {key: val.to(device) for key, val in items.items()}
            
            optimizer.zero_grad()
            
            outputs = model(**item)
            
            preds = torch.argmax(outputs, dim=-1)
            
            loss = criterion(outputs, item['labels'].view(-1))

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_value += loss.item()
            train_batch_acc += (sum(preds == item['labels']) / config.TRAIN_BATCH)
            if (idx + 1) % config.TRAIN_LOG_INTERVAL == 0:
                train_epoch_acc.append(train_batch_acc/config.TRAIN_LOG_INTERVAL)
                train_epoch_loss.append(train_loss_value/config.TRAIN_LOG_INTERVAL)
                
                train_bar.set_description("Loss: {:3f} / {:3f}  | Accuracy: {:3f}".\
                    format(train_loss_value/config.TRAIN_LOG_INTERVAL,
                           sum(train_epoch_loss)/len(train_epoch_loss),
                           train_batch_acc/config.TRAIN_LOG_INTERVAL))

                train_loss_value = 0
                train_batch_acc = 0

                train_total_loss.append(sum(train_epoch_loss)/len(train_epoch_loss))
                train_total_acc.append(sum(train_epoch_acc)/len(train_epoch_acc))

        with torch.no_grad():
            print(f"{b_}---- Validation ----{sr_}")
            model.eval()
            valid_bar = tqdm(valid_dataloader, total=len(valid_dataloader))
            for idx, items in enumerate(valid_bar):
                if idx == 2: break
                item = {key: val.to(device) for key,val in items.items()}
                outputs = model(**item)

                preds = torch.argmax(outputs, dim=-1)
                loss = criterion(outputs, item['labels'].view(-1))

                valid_loss_value += loss.item()
                valid_batch_acc += (sum(preds == item['labels']) / config.VALID_BATCH)
                if (idx + 1) % config.VALID_LOG_INTERVAL == 0:
                    valid_epoch_acc.append(valid_batch_acc/config.VALID_LOG_INTERVAL)
                    valid_epoch_loss.append(valid_loss_value/config.VALID_LOG_INTERVAL)
                    
                    valid_bar.set_description("Loss: {:3f} / {:3f}  | Accuracy: {:3f}".\
                        format(valid_loss_value/config.VALID_LOG_INTERVAL,
                               sum(valid_epoch_loss)/len(valid_epoch_loss),
                               valid_batch_acc/config.VALID_LOG_INTERVAL))

                    valid_loss_value = 0
                    valid_batch_acc = 0
                    
            
            print("{}Best Loss: {:3f}    |    This epoch Loss: {:3f}{}".format(g_, best_val_loss, (sum(valid_epoch_loss)/len(valid_epoch_loss)), sr_))
            if best_val_loss > (sum(valid_epoch_loss)/len(valid_epoch_loss)):
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f'data/model.bin')
                print(f"{r_}Best Loss Model was Saved!{sr_}")
                best_val_loss = (sum(valid_epoch_loss)/len(valid_epoch_loss))

            valid_total_loss.append(sum(valid_epoch_loss)/len(valid_epoch_loss))
            valid_total_acc.append(sum(valid_epoch_acc)/len(valid_epoch_loss))
            logger.info("{0:^23}|{1:^7}|{2:^20.6f}|{3:^20.6f}|{4:^20.6f}|{5:^20.6f}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                                epoch+1,
                                                                                sum(train_epoch_loss)/len(train_epoch_loss),
                                                                                sum(train_epoch_acc)/len(train_epoch_acc),
                                                                                sum(valid_epoch_loss)/len(valid_epoch_loss),
                                                                                best_val_loss))
            logger.info("-"*114)
        print()
    del train_total_loss, train_total_acc, valid_total_loss, valid_total_acc, train_loss_value, train_epoch_loss, valid_loss_value, valid_epoch_loss, train_bar, valid_bar




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--mode', type=str, default='classification', help='classification/regression')
    parser.add_argument('--logger', type=str, default='test', help='logger name')
    
    args = parser.parse_args()
    logger = request_logger(f'{args.logger}')
    
    train_dataloader = pickle.load(open('data/train_dataloader.pkl', 'rb'))
    valid_dataloader = pickle.load(open('data/valid_dataloader.pkl', 'rb'))
    
    
    if args.mode == 'classification':
        train_classifier(train_dataloader, valid_dataloader, args)
        
    else:
        train_regressor(train_dataloader, valid_dataloader, args)
    
    
        
