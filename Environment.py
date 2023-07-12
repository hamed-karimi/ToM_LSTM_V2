import torch.nn as nn
import torch.nn.functional as F
import Utilities
import numpy as np
import torch


def num_flat_features(x):  # This is a function we added for convenience to find out the number of features in a layer.
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class EnvironmentNet(nn.Module):

    # let's assume that the environment is a 8x8 gridworld
    def __init__(self, height, goal_num, env_conv_channel_size, states_num, layers_num):
        super(EnvironmentNet, self).__init__()
        self.h_0 = None
        self.c_0 = None
        kernel_size = 4
        self.conv1 = nn.Conv2d(in_channels=goal_num,
                               out_channels=env_conv_channel_size,
                               kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(in_channels=env_conv_channel_size,
                               out_channels=env_conv_channel_size,
                               kernel_size=height-kernel_size+1)
        self.lstm = nn.LSTM(input_size=env_conv_channel_size,
                            hidden_size=states_num,
                            num_layers=layers_num,
                            batch_first=True)
        self.hidden_size = states_num
        self.layers_num = layers_num
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):  # x: environment without the agent channel
        batch_size = x.shape[0]
        episode_len = x.shape[1]
        env_over_episode = []
        for step in range(episode_len):
            y = self.conv1(x[:, step, :, :, :])  # .squeeze()
            if y.dim() == 1:  # 1 batch
                y = y.unsqueeze(dim=0)
            y = F.relu(y)
            y = self.conv2(y)
            env_over_episode.append(F.relu(y))
        env_over_episode = torch.stack(env_over_episode, dim=1).flatten(start_dim=2)

        self.h_0 = torch.zeros((self.layers_num, batch_size, self.hidden_size),
                               requires_grad=True, device=self.device)
        self.c_0 = torch.zeros((self.layers_num, batch_size, self.hidden_size),
                               requires_grad=True, device=self.device)

        env_belief, (h_n, c_n) = self.lstm(env_over_episode, (self.h_0, self.c_0))

        hidden = (h_n, c_n)
        return env_belief, hidden
