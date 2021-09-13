import pandas as pd
import matplotlib.pyplot as plt
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import datetime
from tqdm import tqdm
import os
from utils import SKIDataset, get_logger, EarlyStopper
from sklearn.model_selection  import train_test_split
import argparse

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder1 = nn.Linear(input_dim * args.seq_length, hidden_dim*4)
        self.encoder2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.encoder3 = nn.Linear(hidden_dim*2, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        
        nn.init.kaiming_uniform_(self.encoder1.weight)
        nn.init.kaiming_uniform_(self.encoder2.weight)
        nn.init.kaiming_uniform_(self.encoder3.weight)

    def forward(self, x):
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        x = self.encoder3(x)
        x = self.ln(x)
        return x

    def save_model(self, path):
        # torch.save(self.state_dict(), os.path.join("model_loss", path, "big_encoder.pt"))
        torch.save(self.state_dict(), os.path.join(path, "big_encoder.pt"))

    def load_model(self, path):
        # self.load_state_dict(torch.load(os.path.join("model_loss", path, "big_encoder.pt")))
        self.load_state_dict(torch.load(os.path.join(path, "big_encoder.pt")))

class ProbabilisticTransitionModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_sigma=10, min_sigma=1e-4): # past : max_sigma=10, min_sigma=1e-4
        super(ProbabilisticTransitionModel, self).__init__()
        
        self.fc = nn.Linear(hidden_dim, hidden_dim//2)
        self.ln = nn.LayerNorm(hidden_dim//2)
        self.fc_mu = nn.Linear(hidden_dim//2, output_dim)
        self.fc_sigma = nn.Linear(hidden_dim//2, output_dim)
        
        nn.init.kaiming_uniform_(self.fc.weight)
        nn.init.kaiming_uniform_(self.fc_mu.weight)
        nn.init.kaiming_uniform_(self.fc_sigma.weight)
        
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma

    def forward(self, x):
        x = F.relu(self.ln(self.fc(x)))
        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

class EnsembleProbabilisticTransitionModel(nn.Module):
    def __init__(self, hidden_dim, output_dim, ensemble_size=5):
        super().__init__()

        self.stochastic_models = [ProbabilisticTransitionModel(hidden_dim, output_dim) for _ in range(ensemble_size)]
        self.ensemble_size = ensemble_size
    
    def forward(self, x):
        mu_sigma_list = [model.forward(x) for model in self.stochastic_models]
        mus, sigmas = zip(*mu_sigma_list)
        mus, sigmas = torch.stack(mus), torch.stack(sigmas)
        model = random.choice(self.stochastic_models)
        return mus, sigmas, model.sample_prediction(x)

    def parameters(self):
        parameters = list()
        for stochastic_model in self.stochastic_models:
            parameters += list(stochastic_model.parameters())
        return parameters

    def to(self, device):
        for stochastic_model in self.stochastic_models:
            stochastic_model = stochastic_model.to(device)
        return self

    def save_model(self, path):
        # for idx in range(self.ensemble_size):
        #     torch.save(self.stochastic_models[idx].state_dict(), os.path.join("model_loss", path, "ensemble_{}.pt".format(idx)))
        
        for idx in range(self.ensemble_size):
            torch.save(self.stochastic_models[idx].state_dict(), os.path.join("model_loss", path, "ensemble_{}.pt".format(idx)))

    def load_model(self, path):
        # for idx in range(self.ensemble_size):        
            # self.stochastic_models[idx].load_state_dict(torch.load(os.path.join("model_loss", path, "ensemble_{}.pt".format(idx))))

        for idx in range(self.ensemble_size):        
            self.stochastic_models[idx].load_state_dict(torch.load(os.path.join(path, "ensemble_{}.pt".format(idx))))

def train_and_evaluation(model, dataloader, optimizer, device, args, logger, early_stopper, start_time):
    train_dataloader, validation_dataloader = dataloader
    train_losses = list()
    evaluation_losses = list()

    for epoch in tqdm(range(args.epochs)):
        train_loss_mean = train(model, dataloader=train_dataloader, optimizer=optimizer, device=device)

        msg = '{} Epoch, Train Mean Loss: {}'.format(epoch, train_loss_mean)
        logger.info(msg)
        train_losses.append(train_loss_mean)

        evaluation_loss_mean = evaluation(model=model, dataloader=validation_dataloader, device=device)

        msg = '{} Epoch, Evaluation Mean Loss: {}'.format(epoch, evaluation_loss_mean)
        logger.info(msg)
        evaluation_losses.append(evaluation_loss_mean)

        early_stopper.check_early_stopping(evaluation_loss_mean)

        if early_stopper.save_model:
            model['encoder'].save_model(start_time)
            model['ensemble'].save_model(start_time)
            msg = '\n\n\t Best Model Saved!!! \n'
            logger.info(msg)

        if early_stopper.stop:
            msg = '\n\n\t Early Stop by Patience Exploded!!! \n'
            logger.info(msg)
            break

    return model, train_losses, evaluation_losses

def train(model, dataloader, optimizer, device):
    epoch_loss = 0

    model['encoder'].train()
    for i in range(model['ensemble'].ensemble_size):
        model['ensemble'].stochastic_models[i].train()
    
    loss_fn = nn.MSELoss()
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        encoder_output = model['encoder'](x.reshape(x.shape[0], -1))
        mu, sigma, predict_output = model['ensemble'](encoder_output)
    
        loss = loss_fn(predict_output.squeeze(), y)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss /= len(dataloader)

    return epoch_loss

def evaluation(model, dataloader, device):
    epoch_loss = 0

    model['encoder'].eval()
    for i in range(model['ensemble'].ensemble_size):
        model['ensemble'].stochastic_models[i].eval()
    
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            encoder_output = model['encoder'](x.reshape(x.shape[0], -1))
            mu, sigma, predict_output = model['ensemble'](encoder_output)
        
            loss = loss_fn(predict_output.squeeze(), y)
            epoch_loss += loss.item()

    epoch_loss /= len(dataloader)

    return epoch_loss


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='SK')

    parser.add_argument('--seq_length', type=int, default=4, help='sequence size')
    parser.add_argument('--batch_size', type=int, default=2000, help='batch size')
    parser.add_argument('--hidden_dim', type=int, default=32, help='hidden dim')
    parser.add_argument('--input_dim', type=int, default=6, help='input_dim')
    parser.add_argument('--output_dim', type=int, default=1, help='output_dim')
    parser.add_argument('--epochs', type=int, default=100, help='epoch')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--ensemble_size', type=int, default=5, help='ensemble size')
    parser.add_argument('--noise_std', type=float, default=0.01, help='nosie std')
    parser.add_argument('--patience', type=int, default=30)
    
    args = parser.parse_args()
    device = torch.device("cuda:3")

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    # get data and dataloader
    entire_data = np.load("/home/lms/d4rl_ski/sk_data_and_model/step_size_1,sequential_length_5.npy")
    entire_data = entire_data.astype(np.float32)

    train_data, validation_data = train_test_split(entire_data, test_size=0.1, random_state=42)
    train_dataset = SKIDataset(train_data)
    validation_dataset = SKIDataset(validation_data)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # make model
    model = {}
    model['encoder'] = Encoder(input_dim=args.input_dim, hidden_dim=args.hidden_dim, args=args).to(device)
    model['ensemble'] = EnsembleProbabilisticTransitionModel(hidden_dim=args.hidden_dim, output_dim=args.output_dim, ensemble_size=args.ensemble_size).to(device)

    # optimizer
    parameters = list(model['encoder'].parameters()) + list(model['ensemble'].parameters())
    optimizer = optim.Adam(parameters, lr=0.01)
    
    # Early Stopper
    early_stopper = EarlyStopper(patience=args.patience)

    # make directory
    if not os.path.exists("model_loss"):
        os.mkdir("model_loss")

    if os.path.isdir(os.path.join("model_loss", start_time)) and os.path.exists(os.path.join("model_loss", start_time)):
        print('Already Existing Directory. Please wait for 1 minute.')
        exit()

    os.mkdir(os.path.join("model_loss", start_time))

    # logger
    system_logger = get_logger(name='Autoencoder model', file_path=os.path.join('model_loss', start_time, 'train_log.log'))

    system_logger.info('===== Arguments information =====')
    system_logger.info(vars(args))

    system_logger.info('===== Model Structure =====')
    system_logger.info(model)

    # train & evaluation
    model, train_losses, evaluation_losses = train_and_evaluation(model=model, dataloader=(train_dataloader, validation_dataloader), 
                                                                optimizer=optimizer, device=device, args=args, logger=system_logger, 
                                                                early_stopper=early_stopper, start_time=start_time)

    np.save(os.path.join('model_loss', start_time, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join('model_loss', start_time, 'evaluation_losses.npy'), np.array(evaluation_losses))