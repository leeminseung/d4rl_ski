import torch.nn as nn
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

    def forward(self, state_and_action):
        if len(state_and_action.shape) > 2:
            state_and_action = state_and_action.reshape(state_and_action.shape[0], -1)
        latent = self.big_encoder(state_and_action)
        selected_model_idx = random.choice(range(len(self.stochastic_models)))
        selected_model =self.stochastic_models[selected_model_idx]
        next_state_reward_prediction = selected_model(latent)
        return next_state_reward_prediction

    def step(self, state_and_action): 
        '''
        used in evaluation process. different compared to 'forward' function in that this function uses mean value of ensemble model output.
        '''
        if len(state_and_action.shape) > 2:
            state_and_action = state_and_action.reshape(state_and_action.shape[0], -1)
        latent = self.big_encoder(state_and_action)
        selected_model_idx = random.choice(range(len(self.stochastic_models)))
        selected_model =self.stochastic_models[selected_model_idx]
        next_state_reward_prediction = selected_model(latent)
        return next_state_reward_prediction

    def save_model(self, path):
        torch.save(self.big_encoder.state_dict(), os.path.join("model_loss", path, "big_encoder.pt"))
        for idx in range(self.ensemble_size):
            torch.save(self.stochastic_models[idx].state_dict(), os.path.join("model_loss", path, "ensemble_{}.pt".format(idx)))

    def load_model(self, path):
        self.big_encoder.load_state_dict(torch.load(os.path.join("model_loss", path, "big_encoder.pt")))
        for idx in range(self.ensemble_size):        
            self.stochastic_models[idx].load_state_dict(torch.load(os.path.join("model_loss", path, "ensemble_{}.pt".format(idx))))
    
    def to(self, device):
        self.big_encoder = self.big_encoder.to(device)
        for idx in range(self.ensemble_size):
            self.stochastic_models[idx] = self.stochastic_models[idx].to(device)

    def print_model(self, logger):
        logger.info('\n\n')
        logger.info(self.big_encoder)
        logger.info(self.stochastic_models[0])
        logger.info('\n')
