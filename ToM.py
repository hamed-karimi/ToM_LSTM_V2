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
        factory = ObjectFactory.ObjectFactory(utility=self.utility)
        self.environment_net = factory.get_environment_net()
        self.agent_net = factory.get_agent_net()
        self.mental_net = factory.get_mental_net()
        self.goal_net = factory.get_goal_net()
        self.action_net = factory.get_action_net()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, environment: torch.Tensor, reinitialize_mental, recalculate_mental=True,
                predefined_goals=torch.empty(0), goal_reached=torch.empty(0)):

        # environment is a tensor containing the steps of the whole episode
        if recalculate_mental:
            return self.forward_from_new_mental(environment, reinitialize_mental)
        else:
            return self.forward_action_layer(predefined_goals, goal_reached)

    def forward_from_new_mental(self, environment, seq_start=True):
        episode_len = environment.shape[1]
        environment_only = environment[:, :, 1:, :, :]
        agent_only = environment[:, :, :1, :, :]
        env_repr, _ = self.environment_net(environment_only, seq_start)
        agent_repr = []
        for step in range(episode_len):
            agent_repr.append(self.agent_net(agent_only[:, step, :, :]))
        agent_repr = torch.stack(agent_repr, dim=1)
        mental_states, _ = self.mental_net(env_repr, agent_repr, seq_start)

        goal_seq, goal_prob_seq = [], []
        action_seq, action_prob_seq = [], []
        for step in range(episode_len):
            step_goal = self.goal_net(F.relu(mental_states[:, step, :]))
            goal_seq.append(step_goal)
            goal_prob_seq.append(self.softmax(step_goal))

            step_action = self.action_net(F.relu(step_goal), F.relu(mental_states[:, step, :]))
            action_seq.append(step_action)
            action_prob_seq.append(self.softmax(step_action))

        goals = torch.stack(goal_seq, dim=1)
        goals_prob = torch.stack(goal_prob_seq, dim=1)

        actions = torch.stack(action_seq, dim=1)
        actions_prob = torch.stack(action_prob_seq, dim=1)

        return goals, goals_prob, actions, actions_prob

    def forward_action_layer(self, predefined_goals, goal_reached):
        reached_goals = predefined_goals[goal_reached]
        binary_goals = torch.zeros(reached_goals.shape[0], self.params.GOAL_NUM + 1)
        binary_goals.index_put_((torch.arange(reached_goals.shape[0]), reached_goals), torch.ones(reached_goals.shape[0]))
        actions = self.fc_action(torch.cat([
            F.relu(binary_goals),
            F.relu(self.mental_states[goal_reached, :])
        ], dim=1))
        actions = self.softmax(actions)
        return actions
