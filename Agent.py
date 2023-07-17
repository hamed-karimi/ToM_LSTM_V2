import torch.nn as nn
import torch.nn.functional as F


class AgentNet(nn.Module):
    def __init__(self, height, states_num):
        super(AgentNet, self).__init__()
        kernel_size = 4
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=kernel_size)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=height - kernel_size + 1)
        self.fc1 = nn.Linear(32, states_num)

    def forward(self, x):
        x = self.conv1(x).squeeze()
        x = self.conv2(F.relu(x))
        if x.dim() == 1:  # 1 batch
            x = x.unsqueeze(dim=0)
        x = x.squeeze()
        agent_repr = self.fc1(F.relu(x))

        return agent_repr
