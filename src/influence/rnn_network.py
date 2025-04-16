from typing import List, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn

class RNNPolicy(nn.Module):
    def __init__(self, input_size=12, hidden_size=15, output_size=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_weights = len(self.get_weights())
        self.hidden = None
        self.observations = [0]*5

    def forward(self, x):
        self.observations.append(x)
        x = torch.tensor(np.array([self.observations[-5:]]), dtype=torch.float32)
        out, self.hidden = self.rnn(x, self.hidden)
        out = self.fc(out)
        return torch.tanh(out).detach().numpy().flatten().astype(np.float64)

    def get_weights(self):
        # Flatten all parameters to a numpy array
        return list(torch.cat([p.data.view(-1) for p in self.parameters()]).numpy())

    def setWeights(self, flat_weights):
        # Assign from flat vector back into model parameters
        pointer = 0
        for p in self.parameters():
            numel = p.numel()
            new_weights = torch.tensor(flat_weights[pointer:pointer + numel]).view(p.shape)
            p.data = new_weights.to(p.dtype)
            pointer += numel