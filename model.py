import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256 + state_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, state_dim)
        )

    def forward(self, h_t, x_t):
        x_onehot = F.one_hot(x_t, num_classes=256).float()
        inp = torch.cat([h_t, x_onehot], dim=-1)
        delta = self.net(inp)
        h_tp1 = h_t + delta
        h_tp1 = F.normalize(h_tp1, dim=-1)  # ensure unit norm
        return h_tp1


class Decoder(nn.Module):
    def __init__(self, input_dim, state_dim):
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, state_dim)
        )
        self.x_net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 256)
        )

    def forward(self, h_tp1):
        delta = self.delta_net(h_tp1)
        h_reconstructed = F.normalize(h_tp1 - delta, dim=-1)
        logits = self.x_net(h_tp1)
        return h_reconstructed, logits

