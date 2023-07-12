import torch.nn as nn
import torch.nn.functional as F


class AgentNet(nn.Module):
    def __init__(self, states_num):
        super(AgentNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 8)
        self.fc1 = nn.Linear(64, states_num)

    def forward(self, x):
        x = self.conv1(x).squeeze()
        if x.dim() == 1:  # 1 batch
            x = x.unsqueeze(dim=0)
        agent_repr = self.fc1(F.relu(x))
        return agent_repr
