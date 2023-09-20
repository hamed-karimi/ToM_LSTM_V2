import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionNet(nn.Module):
    def __init__(self, states_size):
        super(ActionNet, self).__init__()
        self.fc1 = nn.Linear(states_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 9)

    def forward(self, goals, environment_only, agent_only):
        x = torch.concat([goals, environment_only, agent_only], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions
