import torch
import torch.nn as nn
import os

import torch.nn as nn
import torch

class Simple1DCNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequential_length, device, lr=0.0005):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, hidden_size*2, kernel_size=3, stride = 1, padding =1),
            nn.ReLU(),
            nn.Conv1d(hidden_size*2, hidden_size*4, kernel_size= 3, stride = 1, padding =1),
            nn.ReLU(),
            nn.Conv1d(hidden_size*4, hidden_size*8, kernel_size= 3, stride = 1, padding =1),
            nn.ReLU(),
            nn.Conv1d(hidden_size*8 , hidden_size*8, kernel_size= 3, stride = 1, padding =1),
        )
        
        self.fc_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(8*hidden_size*sequential_length, output_size),
        )

        self.device = device
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        # wanring!. Hard Coded dimension permutation
        x = x.permute(0, 2, 1)
        output = self.encoder(x)
        output = self.fc_layer(output.reshape(x.shape[0], -1))
        return output

    def get_latent(self, x):
        x = x.permute(0, 2, 1)
        return self.encoder(x)

    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join("sk_data_and_model/model_loss", path, "model.pt"))

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join("sk_data_and_model/model_loss", path, "model.pt")))