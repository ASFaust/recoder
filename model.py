import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, 512)
        self.g_l1 = nn.Linear(input_dim, 512)
        self.l2 = nn.Linear(512, output_dim)
        self.g_l2 = nn.Linear(512, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h1 = self.l1(x)
        g1 = self.sigmoid(self.g_l1(x))
        h1_g = h1 * g1
        h2 = self.l2(h1_g)
        g2 = self.sigmoid(self.g_l2(h1_g))
        return h2 * g2

class Encoder(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = MLP(state_dim + 256, state_dim)

    def forward(self, h_t, x_t):
        x_onehot = F.one_hot(x_t, num_classes=256).float()
        inp = torch.cat([h_t, x_onehot], dim=-1)
        delta = self.net(inp)
        h_tp1 = h_t + delta
        return h_tp1


class Decoder(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.delta_net = MLP(state_dim, state_dim)
        self.x_net = MLP(state_dim, 256)

    def forward(self, next_state):
        delta = self.delta_net(next_state)
        logits = self.x_net(next_state)
        return next_state + delta, logits

