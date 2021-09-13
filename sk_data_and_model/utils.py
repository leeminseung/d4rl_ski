from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import logging

class EarlyStopper():
    def __init__(self, patience=20):
        self.patience = patience
        self.anger = 0
        self.best_loss = np.Inf
        self.stop = False
        self.save_model = False

    def check_early_stopping(self, validation_loss):
        if self.best_loss == np.Inf:
            self.best_loss = validation_loss

        elif self.best_loss < validation_loss:
            self.anger += 1
            self.save_model = False

            if self.anger >= self.patience:
                self.stop = True

        elif self.best_loss >= validation_loss:
            self.save_model = True
            self.anger = 0
            self.best_loss = validation_loss

class SKIDataset(nn.Module):
    def __init__(self, dataset):
        super(SKIDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[idx, :-1, :], self.dataset[idx, -1, 4]

    def __len__(self):
        return self.dataset.shape[0]

def get_dataloader(dataset, device, args):
    # pin_memory = False if device=='cpu' else True
    pin_memory = False
    batch_size = args.batch_size
    num_workers = args.num_workers

    dataset = dataset.astype(np.float32)

    train_data, validation_data = train_test_split(dataset, test_size=0.1, random_state=42)
    train_data, test_data = train_test_split(train_data, test_size=0.1, random_state=42)

    d4rl_train_dataset = SKIDataset(train_data)
    d4rl_train_dataloader = DataLoader(d4rl_train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)

    d4rl_val_dataset = SKIDataset(validation_data)
    d4rl_val_dataloader = DataLoader(d4rl_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)

    d4rl_test_dataset = SKIDataset(test_data)
    d4rl_test_dataloader = DataLoader(d4rl_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)
    return d4rl_train_dataloader, d4rl_val_dataloader, d4rl_test_dataloader

def get_logger(name: str, file_path: str, stream=False)-> logging.RootLogger:

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # prevent loggint to stdout
    logger.propagate = False
    return logger