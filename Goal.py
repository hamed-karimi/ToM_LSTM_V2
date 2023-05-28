import torch.nn as nn
import torch
import torch.nn.functional as F


class GoalNet(nn.Module):
    def __init__(self, states_size, goal_num):
        super(GoalNet, self).__init__()
        self.fc1 = nn.Linear(states_size, 8)  # +1 for staying
        self.fc2 = nn.Linear(8, goal_num + 1)

    def forward(self, mental_states):
        x = self.fc1(mental_states)
        goals = self.fc2(F.relu(x))
        return goals
