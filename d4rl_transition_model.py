import gym
import d4rl
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch
import os
import tqdm
import matplotlib.pyplot as plt
import datetime

import time

class BigEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BigEncoder, self).__init__()
         
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.encoder(x)

class StochasticTransitionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device='cpu', minimum_std = 0.001):
        super(StochasticTransitionModel, self).__init__()

        self.make_prob_parameters = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * output_size)
        )

        self.minimum_std = minimum_std
        self.output_size = output_size
        self.device = device

    def forward(self, x):
        flatted_mu_std = self.make_prob_parameters(x)
        reshaped_mu_std = flatted_mu_std.reshape(-1, self.output_size, 2)
        mu = reshaped_mu_std[:, :, 0]
        std = reshaped_mu_std[:, :, 1]

        std += self.minimum_std
        epsilon = torch.randn((x.shape[0], self.output_size)).to(self.device)
        next_state_prediction = epsilon * std + mu
        return next_state_prediction


class EntireEnsembleModel(nn.Module):
    def __init__(self, input_size, encoder_hidden_size, encoder_output_size, transition_model_hidden_size, transition_model_output_size, ensemble_size, learning_rate, device='cpu'):
        super(EntireEnsembleModel, self).__init__()
        
        t1 = time.time()
        self.ensemble_size = ensemble_size
        self.big_encoder = BigEncoder(input_size, encoder_hidden_size, encoder_output_size)
        self.stochastic_models = list()
        for _ in range(ensemble_size):
            self.stochastic_models.append(StochasticTransitionModel(encoder_hidden_size, transition_model_hidden_size, transition_model_output_size, device))
        
        t1 = time.time()
        self.all_parameters = list(self.big_encoder.parameters())
        for idx in range(ensemble_size):
            self.all_parameters += list(self.stochastic_models[idx].parameters())

        self.optimizer = torch.optim.Adam(self.all_parameters, lr=learning_rate)
        self.ensemble_size = ensemble_size

    def forward(self, state, action):
        state_action = torch.cat((state, action), dim=1)
        latent = self.big_encoder(state_action)
        selected_model_idx = random.choice(range(len(self.stochastic_models)))
        selected_model =self.stochastic_models[selected_model_idx]
        next_state_reward_prediction = selected_model(latent)
        return next_state_reward_prediction

    def save_model(self, path):
        torch.save(self.big_encoder.state_dict(), os.path.join("model_pt", path + "_big_encoder.pt"))
        for idx in range(self.ensemble_size):
            torch.save(self.stochastic_models[idx].state_dict(), os.path.join("model_pt", path + "_ensemble_{}.pt".format(idx)))

    def load_model(self, path):
        self.big_encoder.load_state_dict(torch.load(os.path.join("model_pt", path + "_big_encoder.pt")))
        for idx in range(self.ensemble_size):        
            self.stochastic_models[idx].load_state_dict(torch.load(os.path.join("model_pt", path + "_ensemble_{}.pt".format(idx))))
    
    def to(self, device):
        self.big_encoder = self.big_encoder.to(device)
        for idx in range(self.ensemble_size):
            self.stochastic_models[idx] = self.stochastic_models[idx].to(device)

class D4rlDataset(nn.Module):
    def __init__(self, d4rl_dataset):
        super(D4rlDataset, self).__init__()
        self.d4rl_dataset = d4rl_dataset
        self.state_array = d4rl_dataset['observations']
        self.next_state_array = d4rl_dataset['next_observations']
        self.action_array = d4rl_dataset['actions']
        self.reward_array = d4rl_dataset['rewards']
        # self.terminal_array = d4rl_dataset['terminals']     

    def __getitem__(self, idx):
        return {'state': self.state_array[idx], 'action': self.action_array[idx], 'next_state': self.next_state_array[idx], 'reward': self.reward_array[idx]}

    def __len__(self):
        return self.state_array.shape[0]

def get_dataloader(dataset, device):
    pin_memory = False if device=='cpu' else True   

    whole_ind = np.arange(dataset['observations'].shape[0])
    train_ind, val_ind = train_test_split(whole_ind, test_size=0.1, random_state = 42)
    train_ind, test_ind = train_test_split(train_ind, test_size=0.1, random_state = 42)

    train_dataset = {}
    val_dataset = {}
    test_dataset = {}

    for key in dataset.keys():
        train_dataset[key] = dataset[key][train_ind]
        val_dataset[key] = dataset[key][val_ind]
        test_dataset[key] = dataset[key][test_ind]

    d4rl_train_dataset = D4rlDataset(dataset)
    d4rl_train_dataloader = DataLoader(d4rl_train_dataset, batch_size=2000, shuffle=True, drop_last=False, pin_memory=pin_memory, num_workers=32)

    d4rl_val_dataset = D4rlDataset(dataset)
    d4rl_val_dataloader = DataLoader(d4rl_val_dataset, batch_size=2000, shuffle=False, drop_last=False, pin_memory=pin_memory, num_workers=32)

    d4rl_test_dataset = D4rlDataset(dataset)
    d4rl_test_dataloader = DataLoader(d4rl_test_dataset, batch_size=2000, shuffle=False, drop_last=False, pin_memory=pin_memory, num_workers=32)
    return d4rl_train_dataloader, d4rl_val_dataloader, d4rl_test_dataloader

def train_transition_model(model, start_time, device):
    loss_fn = nn.MSELoss()
    epochs = 150
    train_losses = []
    val_losses = []
    best_val_loss = 1e8
    
    for epoch in tqdm.tqdm(range(epochs)):
        loss_sum = 0
        for row in d4rl_train_dataloader:
            state = row['state'].to(device)
            reward = row['reward'].to(device)
            action = row['action'].to(device)
            next_state = row['next_state'].to(device)

            next_state_reward_prediction = model(state, action)
            next_state_reward = torch.cat((next_state, reward.unsqueeze(1)), dim=1)
            loss = loss_fn(next_state_reward, next_state_reward_prediction)

            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()

            loss_sum += loss.item()
        
        train_losses.append(loss_sum/len(d4rl_train_dataloader))

        print('Train Mean Loss at {} epoch: {}'.format(loss_sum/len(d4rl_train_dataloader), epoch))
        loss_sum = 0
        with torch.no_grad():
            for row in d4rl_val_dataloader:
                state = row['state'].to(device)
                reward = row['reward'].to(device)
                action = row['action'].to(device)
                next_state = row['next_state'].to(device)

                next_state_reward_prediction = model(state, action)
                next_state_reward = torch.cat((next_state, reward.unsqueeze(1)), dim=1)
                loss = loss_fn(next_state_reward, next_state_reward_prediction)

                loss_sum += loss.item()
        
        val_losses.append(loss_sum/len(d4rl_train_dataloader))
        print('Validation Mean Loss at {} epoch: {}'.format(loss_sum/len(d4rl_train_dataloader), epoch))
        
        plt.plot(train_losses, label='train loss', color='r')
        plt.plot(val_losses, label='validation loss', color='b')
        if epoch == 0:
            plt.legend()
        plt.savefig('result/{}.png'.format(start_time))
        if loss_sum < best_val_loss:
            model.save_model(start_time)
            best_val_loss = loss_sum

    return model

if __name__ == '__main__':

    start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.synchronize()

    env = gym.make('halfcheetah-medium-expert-v2')
    env.reset()
    env.step(env.action_space.sample())

    dataset = env.get_dataset()
    dataset = d4rl.qlearning_dataset(env)
    
    d4rl_train_dataloader, d4rl_val_dataloader, d4rl_test_dataloader = get_dataloader(dataset, device)

    model = EntireEnsembleModel(input_size=17+6, encoder_hidden_size=64, encoder_output_size=64, transition_model_hidden_size=64, transition_model_output_size=18, ensemble_size=5, learning_rate=0.0005, device=device)
    model.to(device)

    model = train_transition_model(model, start_time, device)