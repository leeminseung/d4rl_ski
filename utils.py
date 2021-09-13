import math
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import logging

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class StdScaler():
    def __init__(self, dataset, device):
        self.stds = dict()
        self.means = dict() 
        for key in dataset.keys():
            if key == 'terminals':
                continue
            self.stds[key] = torch.tensor(dataset[key].std(0), device=device)
            self.means[key] = torch.tensor(dataset[key].mean(0), device=device)

    def scale(self, x, key):
        return (x - self.means[key]) / self.stds[key]

    def inverse_scale(self, x, key):
        return x * self.stds[key] + self.means[key]

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

class D4rlDataset(nn.Module):
    def __init__(self, dataset):
        super(D4rlDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, idx):
        # state_batch, action_batch, reward_batch, next_state_batch, mask_batch
        return self.dataset['observations'][idx], self.dataset['actions'][idx], self.dataset['rewards'][idx], self.dataset['next_observations'][idx], float(not self.dataset['terminals'][idx])

    def __len__(self):
        return self.dataset['observations'].shape[0]

class SKIDataset(nn.Module):
    def __init__(self, dataset):
        super(SKIDataset, self).__init__()
        self.dataset = dataset

    def __getitem__(self, idx):
        # state, action, reward, next_state, mask
        state = self.dataset[idx, 0, :-1]
        action = self.dataset[idx, 0, -1]
        next_state = self.dataset[idx, 1, :-1]
        
        # Current std: 0.391660, mean: 29.983252
        reward = - abs(next_state[-1]*0.392 + 29.98 - 30)
        mask = 1.
        return state, np.expand_dims(action, axis=0), reward, next_state, mask

    def __len__(self):
        return self.dataset.shape[0]

def get_dataloader(dataset, device, data_mode):
    pin_memory = False if device=='cpu' else True
    batch_size = 256
    num_workers = 1

    # train_data, validation_data = train_test_split(dataset, test_size=0.1, random_state=42)
    # train_data, test_data = train_test_split(train_data, test_size=0.1, random_state=42)

    # d4rl_train_dataset = D4rlDataset(train_data)
    # d4rl_train_dataloader = DataLoader(d4rl_train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)

    # d4rl_val_dataset = D4rlDataset(validation_data)
    # d4rl_val_dataloader = DataLoader(d4rl_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)

    # d4rl_test_dataset = D4rlDataset(test_data)
    # d4rl_test_dataloader = DataLoader(d4rl_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)
    
    if data_mode == 'd4rl':
        d4rl_dataset = D4rlDataset(dataset)
        returned_dataloader = DataLoader(d4rl_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)

    elif data_mode == 'ski':
        ski_dataset = SKIDataset(dataset)
        returned_dataloader = DataLoader(ski_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)

    return returned_dataloader

class ActionSpaceBox():
    def __init__(self, shape, high, low):
        self.shape = shape
        self.high = np.array([high])
        self.low = np.array([low])

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