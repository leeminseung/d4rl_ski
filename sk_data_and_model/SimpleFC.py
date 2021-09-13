import torch
import torch.nn as nn
import os

class SimpleFC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, lr=0.0005):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        self.device = device
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        output = self.encoder(x)
        return output

    def get_latent(self, x):
        return self.encoder(x)

    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join("model_loss", path, "model.pt"))

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join("model_loss", path, "model.pt")))