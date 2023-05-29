import os.path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SequentialSampler, BatchSampler, DataLoader, SubsetRandomSampler, Sampler
import Utilities
from DataSet import AgentActionDataSet, get_agent_appearance
from ObjectFactory import ObjectFactory


def get_data_loader(utility):
    def get_generator_from_sampler(sampler: Sampler, batch_size):
        batch_sampler = BatchSampler(sampler=sampler,
                                     batch_size=batch_size,
                                     drop_last=False)
        params = {'batch_sampler': batch_sampler,
                  'pin_memory': True}

        generator = DataLoader(dataset, **params)
        return generator

    dataset = AgentActionDataSet()
    train_batch_size = utility.params.BATCH_SIZE
    train_sampler = SubsetRandomSampler(np.arange(int(utility.params.TRAIN_PROPORTION * len(dataset))))
    test_sampler = SequentialSampler(np.arange(int(utility.params.TRAIN_PROPORTION * len(dataset)), len(dataset)))

    train_generator = get_generator_from_sampler(train_sampler, batch_size=train_batch_size)
    test_generator = get_generator_from_sampler(test_sampler, batch_size=1)

    return train_generator, test_generator


def change_require_grads(model, goal_grad, action_grad):
    for params in model.goal_net.parameters():
        params.requires_grad = goal_grad
    for params in model.action_net.parameters():
        params.requires_grad = action_grad


def train(train_data_generator):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utility = Utilities.Utilities()
    params = utility.params
    writer = SummaryWriter()
    factory = ObjectFactory(utility=utility)
    tom_net = factory.get_tom_net()
    optimizer = torch.optim.Adam(tom_net.parameters(),
                                 lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    global_index = 0

    for epoch in range(params.NUM_EPOCHS):
        epoch_goal_loss = 0
        epoch_all_actions_loss = 0
        epoch_action_prediction_performance = 0
        n_batch = 0
        seq_start = True
        # goal_criterion = nn.NLLLoss(reduction='mean', weight=torch.tensor([4.5, 4.5, 1]).to(device))
        goal_criterion = nn.CrossEntropyLoss(reduction='mean', weight=torch.tensor([4.5, 4.5, 1]).to(device))
        action_criterion = nn.NLLLoss(reduction='mean')
        for train_idx, data in enumerate(train_data_generator):
            # environment_batch.shape: [batch_size, step_num, objects+agent(s), height, width]
            # target_goal: 2 is staying
            environments_batch, \
                goals_batch, \
                actions_batch, \
                needs_batch, \
                goal_reached_batch, \
                targets_prob_batch, \
                has_target_dist_batch = data

            environments_batch = environments_batch.to(device)
            goals_batch = goals_batch.to(device)
            actions_batch = actions_batch.to(device)
            needs_batch = needs_batch.to(device)
            goal_reached_batch = goal_reached_batch.to(device)
            # targets_prob_batch = targets_prob_batch.to(device)
            # has_target_dist_batch = has_target_dist_batch.to(device)
            # agent_appearance_batch = agent_appearance.repeat(environments_batch.shape[0], 1, 1, 1)

            change_require_grads(tom_net, goal_grad=True, action_grad=True)
            optimizer.zero_grad()

            goals, goals_prob, actions, actions_prob = tom_net(environments_batch,
                                                               reinitialize_mental=seq_start)

            # 2 losses:
            # 1. goal loss
            # 2. action loss

            # goal loss
            change_require_grads(tom_net, goal_grad=True, action_grad=False)
            stayed_batch = torch.zeros(goals_batch.shape).to(device)
            stayed_batch[goals_batch == params.GOAL_NUM] = 1
            reached_or_stayed_batch = torch.logical_or(goal_reached_batch, stayed_batch)

            goal_loss = goal_criterion(goals_prob[reached_or_stayed_batch, :],
                                       goals_batch[reached_or_stayed_batch].long())
            goal_loss.backward(retain_graph=True)

            # all actions loss
            change_require_grads(tom_net, goal_grad=True, action_grad=True)
            action_loss = action_criterion(actions_prob.reshape(actions_prob.shape[0], 9, -1),
                                           actions_batch.long())
            action_loss.backward()

            # Inject true goals to the action net, and compute derivative
            # change_require_grads(tom_net, goal_grad=False, action_grad=True)

            # reached_or_stayed = goals_batch[torch.logical_or(goal_reached_batch, stayed_batch)]
            # binary_goals = torch.zeros(reached_goals.shape[0], self.params.GOAL_NUM + 1)
            # binary_goals.index_put_((torch.arange(reached_goals.shape[0]), reached_goals),
            #                         torch.ones(reached_goals.shape[0]))
            ###
            optimizer.step()
            #
            epoch_all_actions_loss += action_loss.item()
            epoch_goal_loss += goal_loss.item()
            epoch_action_prediction_performance += torch.eq(actions_batch, torch.argmax(actions_prob, dim=2)).sum().item() / \
                                                   (actions_batch.shape[0]*actions_batch.shape[1])

            n_batch += 1
            print('epoch: ', epoch, ', batch: ', train_idx)
            global_index += 1

        writer.add_scalar("Loss/goal", epoch_goal_loss / n_batch, epoch)
        writer.add_scalar("Loss/all_action", epoch_all_actions_loss / n_batch, epoch)
        writer.add_scalar("Accuracy/action", epoch_action_prediction_performance / n_batch, epoch)

    writer.flush()
    if not os.path.exists('./Model'):
        os.mkdir('./Model')
    torch.save(tom_net.cpu().state_dict(), './Model/ToM_RNN.pt')
