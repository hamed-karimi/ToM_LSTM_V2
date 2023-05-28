import torch.nn as nn
import torch.nn.functional as F


def num_flat_features(x):  # This is a function we added for convenience to find out the number of features in a layer.
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class TraitsNet(nn.Module):
    # let's assume that the agent is a 4x4 image
    def __init__(self, agent_size, traits_num):
        super(TraitsNet, self).__init__()
        self.conv1 = nn.Conv2d(1, agent_size, 2)  # 1 input image channel, 4 output channels, 2x2 square convolution
        # kernel
        self.conv2 = nn.Conv2d(4, 8, 2)  # 4 channels from the conv1 layer, 8 output channels, 2x2 square convolution
        # kernel
        self.fc1 = nn.Linear(8 * 2 * 2, traits_num)  # 2*2 from image dimension

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
