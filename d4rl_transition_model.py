import gym
import d4rl
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch
import os

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
    def __init__(self, input_size, hidden_size, output_size, minimum_std = 0.001):
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

    def forward(self, x):
        flatted_mu_std = self.make_prob_parameters(x)
        reshaped_mu_std = flatted_mu_std.reshape(-1, self.output_size, 2)
        mu = reshaped_mu_std[:, :, 0]
        std = reshaped_mu_std[:, :, 1]

        std += self.minimum_std
        epsilon = torch.randn((x.shape[0], self.output_size))
        next_state_prediction = epsilon * std + mu
        return next_state_prediction


class EntireEnsembleModel(nn.Module):
    def __init__(self, input_size, encoder_hidden_size, encoder_output_size, transition_model_hidden_size, transition_model_output_size, ensemble_size, learning_rate):
        super(EntireEnsembleModel, self).__init__()
        
        self.ensemble_size = ensemble_size
        self.big_encoder = BigEncoder(input_size, encoder_hidden_size, encoder_output_size)
        self.stochastic_models = list()
        for _ in range(ensemble_size):
            self.stochastic_models.append(StochasticTransitionModel(encoder_hidden_size, transition_model_hidden_size, transition_model_output_size))
        
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
        next_state_prediction = selected_model(latent)
        return next_state_prediction

    def save_model(self, path):
        torch.save(self.big_encoder.state_dict(), os.path.join("model_pt", path + "_big_encoder.pt"))
        for idx in range(self.ensemble_size):
            torch.save(self.stochastic_models[idx].state_dict(), os.path.join("model_pt", path + "_ensemble_{}.pt".format(idx)))

    def load_model(self, path):
        self.big_encoder.load_state_dict(torch.load(os.path.join("model_pt", path + "_big_encoder.pt")))
        for idx in range(self.ensemble_size):        
            self.stochastic_models[idx].load_state_dict(torch.load(os.path.join("model_pt", path + "_ensemble_{}.pt".format(idx))))

class D4rlDataset(nn.Module):
    def __init__(self, d4rl_dataset):
        super(D4rlDataset, self).__init__()
        self.d4rl_dataset = d4rl_dataset
        self.state_array = d4rl_dataset['observations']
        self.next_state_array = d4rl_dataset['next_observations']
        self.action_array = d4rl_dataset['actions']
        self.reward_array = d4rl_dataset['rewards']
        self.terminal_array = d4rl_dataset['terminals']        

    def __getitem__(self, idx):
        return {'state': self.state_array[idx], 'action': self.action_array[idx], 'next_state': self.next_state_array[idx]}

    def __len__(self):
        return self.state_array.shape[0]

def get_dataloader(dataset):
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
    d4rl_train_dataloader = DataLoader(d4rl_train_dataset, batch_size=256, shuffle=True, drop_last=False)

    d4rl_val_dataset = D4rlDataset(dataset)
    d4rl_val_dataloader = DataLoader(d4rl_val_dataset, batch_size=256, shuffle=True, drop_last=False)

    d4rl_test_dataset = D4rlDataset(dataset)
    d4rl_test_dataloader = DataLoader(d4rl_test_dataset, batch_size=256, shuffle=True, drop_last=False)
    return d4rl_train_dataloader, d4rl_val_dataloader, d4rl_test_dataloader

def train_transition_model(model):
    loss_fn = nn.MSELoss()
    epochs = 100
    for epoch in range(epochs):
        loss_sum = 0
        for row in d4rl_train_dataloader:
            state = row['state']
            action = row['action']
            next_state = row['next_state']

            next_state_prediction = model(state, action)
            loss = loss_fn(next_state, next_state_prediction)

            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()

            loss_sum += loss.item()

        print('Train Mean Loss at {} epoch: {}'.format(loss_sum/len(d4rl_train_dataloader), epoch))

        loss_sum = 0
        with torch.no_grad():
            for row in d4rl_val_dataloader:
                state = row['state']
                action = row['action']
                next_state = row['next_state']

                next_state_prediction = model(state, action)
                loss = loss_fn(next_state, next_state_prediction)

                loss_sum += loss.item()

        print('Validation Mean Loss at {} epoch: {}'.format(loss_sum/len(d4rl_train_dataloader), epoch))
    return model

if __name__ == '__main__':
    env = gym.make('halfcheetah-medium-expert-v2')
    env.reset()
    env.step(env.action_space.sample())

    dataset = env.get_dataset()
    dataset = d4rl.qlearning_dataset(env)

    d4rl_train_dataloader, d4rl_val_dataloader, d4rl_test_dataloader = get_dataloader(dataset)
    model = EntireEnsembleModel(input_size=17+6, encoder_hidden_size=64, encoder_output_size=64, transition_model_hidden_size=64, transition_model_output_size=17, ensemble_size=5, learning_rate=0.0005)
    model = train_transition_model(model)
    model.save_model('0804')