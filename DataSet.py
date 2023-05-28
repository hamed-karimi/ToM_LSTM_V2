import gc
import os.path
import pickle

from torch import nn
import numpy as np
import torch
from torch.utils.data import Dataset
from os.path import join,exists
import Utilities


class AgentActionDataSet(Dataset):
    def __init__(self):
        self.utility = Utilities.Utilities()
        self.params = self.utility.params
        self.dir_path = self.params.DATA_DIRECTORY

        self.environments = torch.load(join(self.dir_path, 'environments.pt'))
        self.target_goals = torch.load(join(self.dir_path, 'selected_goals.pt'))
        self.target_actions = torch.load(join(self.dir_path, 'actions.pt'))
        self.target_needs = torch.load(join(self.dir_path, 'needs.pt'))
        self.reached_goal = torch.load(join(self.dir_path, 'goal_reached.pt'))
        if not exists(join(self.dir_path, 'discounted_selected_goals.pt')):
            self.discounted_target_goals, self.has_target_distribution = self.initialize_discounted_target_goal_label()
            torch.save(self.discounted_target_goals, join(self.dir_path, 'discounted_selected_goals.pt'))
            torch.save(self.has_target_distribution, join(self.dir_path, 'has_target_distribution.pt'))
        else:
            self.discounted_target_goals = torch.load(join(self.dir_path, 'discounted_selected_goals.pt'))
            self.has_target_distribution = torch.load(join(self.dir_path, 'has_target_distribution.pt'))

        print('dataset size: ', self.environments.shape[:2])

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.environments)

    def __getitem__(self, episode: int):
        return self.environments[episode, :, :, :, :], \
            self.target_goals[episode, :], \
            self.target_actions[episode, :], \
            self.target_needs[episode, :, :], \
            self.reached_goal[episode, :], \
            self.discounted_target_goals[episode, :, :], \
            self.has_target_distribution[episode, :]

    def initialize_discounted_target_goal_label(self):
        target_goals_prob = torch.zeros(self.target_goals.shape[0], self.target_goals.shape[1], self.params.GOAL_NUM+1)
        has_target_dist = torch.zeros(self.target_goals.shape, dtype=torch.bool)
        staying_prob = torch.zeros(self.params.GOAL_NUM+1, )
        staying_prob[self.params.GOAL_NUM] = 1.
        softmax = nn.Softmax(dim=0)
        for episode in range(self.reached_goal.shape[0]):
            stayed_indices = torch.argwhere(torch.eq(self.target_goals[episode, :], self.params.GOAL_NUM))
            target_goals_prob[episode, stayed_indices, :] = staying_prob
            has_target_dist[episode, stayed_indices] = True
            goal_reached_indices = torch.argwhere(self.reached_goal[episode, :])
            for step in goal_reached_indices.__reversed__():
                seen_target_goal = self.target_goals[episode, step.item()]
                backward = max(step.item()-1, 0)
                target_goals_prob[episode, step.item(), seen_target_goal] = 1
                has_target_dist[episode, step.item()] = True
                prob = torch.ones(self.params.GOAL_NUM+1, )
                prob[-1] = 0
                while self.target_goals[episode, backward] != self.params.GOAL_NUM and not self.reached_goal[episode, backward]:
                    # all_maps = self.environments[episode, backward+1, :, :, :]
                    # agent_map = self.environments[episode, backward+1, 0, :, :]
                    # agent_object_distances = get_collection_distances(agent_map, all_maps)
                    # agent_object_distances = torch.roll(agent_object_distances, 2, dims=1)
                    if step.item() - backward == 1:
                        prob[seen_target_goal] *= self.params.DISCOUNT_FACTOR
                        prob[torch.arange(self.params.GOAL_NUM+1) != seen_target_goal] *= (1-self.params.DISCOUNT_FACTOR)

                    target_goals_prob[episode, backward, :-1] = softmax(prob[:-1])  # * 1/agent_object_distances[:, :-1])
                    has_target_dist[episode, backward] = True
                    backward -= 1
        return target_goals_prob, has_target_dist


def get_collection_distances(map1, map2):
    p1 = torch.argwhere(map1)
    p2 = torch.argwhere(map2)
    return torch.cdist(p1.float(), p2[:, 1:].float())


def get_agent_appearance():
    utility = Utilities.Utilities()
    params = utility.get_params()
    dir_path = params.DATA_DIRECTORY
    agent_face_file_object = open(join(dir_path, 'agent_face.pkl'), 'rb')
    agent_face = pickle.load(agent_face_file_object)
    return np.expand_dims(agent_face, axis=0)
