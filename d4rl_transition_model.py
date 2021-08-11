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
import argparse
import time
from get_logger import get_logger

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
    def __init__(self, input_size, hidden_size, output_size, device='cpu', minimum_std = 0.001, gaussian_noise_std=0):
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
        self.gaussian_noise_std = gaussian_noise_std
        self.device = device

    def forward(self, x):
        flatted_mu_logvar = self.make_prob_parameters(x)
        reshaped_mu_logvar = flatted_mu_logvar.reshape(-1, self.output_size, 2)
        mu = reshaped_mu_logvar[:, :, 0]
        logvar = reshaped_mu_logvar[:, :, 1]
        std = torch.exp(logvar).clone() # for network stability

        std += self.minimum_std
        epsilon = torch.randn((x.shape[0], self.output_size)).to(self.device)
        next_state_prediction = epsilon * std + mu
        return next_state_prediction


class EntireEnsembleModel(nn.Module):
    def __init__(self, input_size, encoder_hidden_size, encoder_output_size, transition_model_hidden_size, transition_model_output_size, ensemble_size, learning_rate, device='cpu'):
        super(EntireEnsembleModel, self).__init__()
        self.ensemble_size = ensemble_size
        self.big_encoder = BigEncoder(input_size, encoder_hidden_size, encoder_output_size)
        self.stochastic_models = list()
        for _ in range(ensemble_size):
            self.stochastic_models.append(StochasticTransitionModel(encoder_hidden_size, transition_model_hidden_size, transition_model_output_size, device))
        
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

    def step(self, state, action): 
        '''
        used in evaluation process. different compared to 'forward' function in that this function uses mean value of ensemble model output.
        '''
        state_action = torch.cat((state, action), dim=1)
        latent = self.big_encoder(state_action)
        selected_model_idx = random.choice(range(len(self.stochastic_models)))
        selected_model =self.stochastic_models[selected_model_idx]
        next_state_reward_prediction = selected_model(latent)
        return next_state_reward_prediction

    def save_model(self, path):
        torch.save(self.big_encoder.state_dict(), os.path.join("model_pt", path, "big_encoder.pt"))
        for idx in range(self.ensemble_size):
            torch.save(self.stochastic_models[idx].state_dict(), os.path.join("model_pt", path, "ensemble_{}.pt".format(idx)))

    def load_model(self, path):
        self.big_encoder.load_state_dict(torch.load(os.path.join("model_pt", path, "big_encoder.pt")))
        for idx in range(self.ensemble_size):        
            self.stochastic_models[idx].load_state_dict(torch.load(os.path.join("model_pt", path, "ensemble_{}.pt".format(idx))))
    
    def to(self, device):
        self.big_encoder = self.big_encoder.to(device)
        for idx in range(self.ensemble_size):
            self.stochastic_models[idx] = self.stochastic_models[idx].to(device)

    def print_model(self, logger):
        logger.info('\n\n')
        logger.info()
        logger.info(self.big_encoder)
        logger.info(self.stochastic_models[0])
        logger.info('\n')

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

def get_dataloader(dataset, device, args):
    pin_memory = False if device=='cpu' else True
    batch_size = args.batch_size
    num_workers = args.num_workers   

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
    d4rl_train_dataloader = DataLoader(d4rl_train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)

    d4rl_val_dataset = D4rlDataset(dataset)
    d4rl_val_dataloader = DataLoader(d4rl_val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)

    d4rl_test_dataset = D4rlDataset(dataset)
    d4rl_test_dataloader = DataLoader(d4rl_test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory, num_workers=num_workers)
    return d4rl_train_dataloader, d4rl_val_dataloader, d4rl_test_dataloader

def train_transition_model(model, start_time, device, d4rl_std, args, logger):
    loss_fn = nn.MSELoss()
    epochs = args.epochs
    train_losses = []
    val_losses = []
    best_val_loss = 1e8
    state_std, action_std = d4rl_std

    # relative_state_std shape : (17,) in case 'HalfCheetah-v2'
    # relative_action_std shape : (6,) in case 'HalfCheetah-v2'
    relative_state_std = state_std * args.relative_gaussian_noise 
    relative_action_std = action_std * args.relative_gaussian_noise

    for epoch in tqdm.tqdm(range(epochs)):
        loss_sum = 0
        for row in d4rl_train_dataloader:
            # Make noise term
            state_noise = torch.randn(args.batch_size, len(state_std), device=device) * torch.tensor(relative_state_std, dtype=torch.float, device=device)
            action_noise = torch.randn(args.batch_size, len(action_std), device=device) * torch.tensor(relative_action_std, dtype=torch.float, device=device)

            state = row['state'].to(device)
            reward = row['reward'].to(device)
            action = row['action'].to(device)
            next_state = row['next_state'].to(device)

            next_state_reward_prediction = model(state+state_noise, action+action_noise)
            next_state_reward = torch.cat((next_state, reward.unsqueeze(1)), dim=1)
            loss = loss_fn(next_state_reward, next_state_reward_prediction)

            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()

            loss_sum += loss.item()
        
        train_losses.append(loss_sum/len(d4rl_train_dataloader))
        msg = '{} Epoch, Train Mean Loss: {}'.format(epoch, loss_sum/len(d4rl_train_dataloader))
        logger.info(msg)

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
        msg = '{} Epoch, Validation Mean Loss: {}'.format(epoch, loss_sum/len(d4rl_train_dataloader))
        logger.info(msg)
        
        plt.plot(train_losses, label='train loss', color='r')
        plt.plot(val_losses, label='validation loss', color='b')

        if epoch == 0:
            plt.legend()
        plt.savefig(os.path.join('transition_model_loss', start_time, start_time + '.png'))
        np.save(os.path.join('transition_model_loss', start_time, 'train_losses.npy'), np.array(train_losses))
        np.save(os.path.join('transition_model_loss', start_time, 'val_losses.npy'), np.array(val_losses))

        if loss_sum < best_val_loss:
            model.save_model(start_time)
            best_val_loss = loss_sum
            msg = '\n\n\t Best Model Saved!!! \n'
            logger.info(msg)


    return model

if __name__ == '__main__':
    # get Arguments
    parser = argparse.ArgumentParser(description='SKI: Traning Transition Model Args')
    parser.add_argument('--epochs', default=150, type=int, help='Set epochs to train Transition Model')
    parser.add_argument('--relative_gaussian_noise', default=0, type=float, help='Relative gaussian noise to std in d4rl dataset')
    parser.add_argument('--batch_size', default=2000, type=int, help='Batch size used in d4rl dataloader')
    parser.add_argument('--num_workers', default=32, type=int, help='Num workers used in d4rl dataloader')
    parser.add_argument('--hidden_node', default=64, type=int, help='Number of hidden nodes')

    args = parser.parse_args()

    start_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    
    # make directory
    if os.path.isdir(os.path.join("transition_model_loss", start_time)) and os.path.exists(os.path.join("transition_model_loss", start_time)):
        print('Already Existing Directory. Please wait for 1 minute.')
        exit()

    os.mkdir(os.path.join("transition_model_loss", start_time))
    os.mkdir(os.path.join("model_pt", start_time))    

    env = gym.make('halfcheetah-medium-expert-v2')

    # dataset = env.get_dataset()
    dataset = d4rl.qlearning_dataset(env)
    
    d4rl_train_dataloader, d4rl_val_dataloader, d4rl_test_dataloader = get_dataloader(dataset, device, args)

    model = EntireEnsembleModel(input_size=17+6, encoder_hidden_size=args.hidden_node, encoder_output_size=args.hidden_node, transition_model_hidden_size=args.hidden_node, transition_model_output_size=18, ensemble_size=5, learning_rate=0.0005, device=device)
    model.to(device)

    # record model structure
    system_logger = get_logger(name='d4rl_transition_model', file_path=os.path.join('transition_model_loss', start_time, start_time + '_train_log.log'))

    system_logger.info('===== Arguments information =====')
    system_logger.info(vars(args))

    system_logger.info('===== Model Structure =====')
    model.print_model(system_logger)

    system_logger.info('===== Loss History of Transition Model =====')
    model = train_transition_model(model, start_time, device, (dataset['observations'].std(0), dataset['actions'].std(0)), args, system_logger)