import torch
import torch.nn as nn
import torch.nn.functional as F
import ObjectFactory
import Utilities
from Traits import TraitsNet
from Environment import EnvironmentNet
from Mental import MentalNet
from Goal import GoalNet


class ToMNet(nn.Module):
    # this is meant to operate as the meta-controller
    def __init__(self):
        super(ToMNet, self).__init__()
        # self.mental_states = None
        self.utility = Utilities.Utilities()
        self.params = self.utility.params
        self.factory = ObjectFactory.ObjectFactory(utility=self.utility)
        self.environment_net = self.factory.get_environment_net()
        self.agent_net = self.factory.get_agent_net()
        self.mental_net = self.factory.get_mental_net()
        self.goal_net = self.factory.get_goal_net()
        self.action_net = self.factory.get_action_net()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, environment: torch.Tensor,
                inject_true_goals=False, goals=torch.empty(0)):

        episode_len = environment.shape[1]
        environment_only = environment[:, :, 1:, :, :]
        agent_only = environment[:, :, :1, :, :]
        env_repr, _ = self.environment_net(environment_only)
        agent_repr = []
        for step in range(episode_len):
            agent_repr.append(self.agent_net(agent_only[:, step, :, :]))
        agent_repr = torch.stack(agent_repr, dim=1)

        goal_seq, goal_prob_seq = [], []
        action_seq, action_prob_seq = [], []

        mental_states, _ = self.mental_net(F.relu(env_repr),
                                           F.relu(agent_repr))

        for step in range(episode_len):
            step_goal = self.goal_net(F.relu(mental_states[:, step, :]))
            goal_seq.append(step_goal)
            goal_prob_seq.append(self.softmax(step_goal))

            step_action = self.action_net(F.relu(step_goal),
                                          F.relu(env_repr[:, step, :]),
                                          F.relu(agent_repr[:, step, :]))
            action_seq.append(step_action)
            action_prob_seq.append(self.softmax(step_action))

        goals_prob = torch.stack(goal_prob_seq, dim=1)
        actions_prob = torch.stack(action_prob_seq, dim=1)

        actions_prob_of_true_goals = None
        if inject_true_goals:
            action_seq_of_true_goals, actions_prob_seq_of_true_goals = [], []
            for step in range(episode_len):
                step_goal = torch.zeros(goals.shape[0], self.params.GOAL_TYPE_NUM+1,
                                        dtype=torch.float,
                                        device=goals.device)

                step_goal.index_put_((torch.arange(goals.shape[0], device=goals.device),
                                      goals[:, step].long()),
                                     torch.ones(goals[:, step].shape, device=goals.device))

                step_action = self.action_net(F.relu(step_goal),
                                              F.relu(env_repr[:, step, :]),
                                              F.relu(agent_repr[:, step, :]))
                actions_prob_seq_of_true_goals.append(self.softmax(step_action))

            actions_prob_of_true_goals = torch.stack(actions_prob_seq_of_true_goals, dim=1)

        return goals_prob, actions_prob, actions_prob_of_true_goals
