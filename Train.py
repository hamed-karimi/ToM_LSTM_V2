import os.path
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import Utilities
from ObjectFactory import ObjectFactory


def change_require_grads(model, goal_grad, action_grad, mental_grad=True, agent_grad=True):
    for params in model.goal_net.parameters():
        params.requires_grad = goal_grad
    for params in model.action_net.parameters():
        params.requires_grad = action_grad
    for params in model.mental_net.parameters():
        params.requires_grad = mental_grad
    for params in model.agent_net.parameters():
        params.requires_grad = agent_grad


def train(train_data_generator, validation_data_generator):
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
        n_train_batch = 0
        n_validation_batch = 0
        seq_start = True
        # goal_criterion = nn.NLLLoss(reduction='mean', weight=torch.tensor([4.5, 4.5, 1]).to(device))
        goal_criterion = nn.CrossEntropyLoss(reduction='mean', weight=torch.tensor([4.5, 4.5, 1]).to(device))
        action_criterion = nn.NLLLoss(reduction='mean')
        for train_idx, data in enumerate(train_data_generator):
            continue
            # environment_batch.shape: [batch_size, step_num, objects+agent(s), height, width]
            # target_goal: 2 is staying
            environments_batch, \
                goals_batch, \
                actions_batch, \
                needs_batch, \
                goal_reached_batch, \
                retrospective_goals_batch = data

            change_require_grads(tom_net,
                                 goal_grad=True,
                                 action_grad=True,
                                 mental_grad=True,
                                 agent_grad=True)

            optimizer.zero_grad()

            goals_prob, actions_prob, action_prob_of_true_goals = tom_net(environments_batch,
                                                                          inject_true_goals=True,
                                                                          goals=retrospective_goals_batch)
            # 2 losses:
            # 1. goal loss
            # 2. all actions losses
            # 3. retrospective goal losses

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
            action_loss.backward(retain_graph=True)

            # Inject true goals to the action net, and compute derivative
            # retrospective goal losses
            change_require_grads(tom_net,
                                 goal_grad=False,
                                 action_grad=True,
                                 mental_grad=False,
                                 agent_grad=False)

            action_true_goal_loss = action_criterion(
                action_prob_of_true_goals.reshape(action_prob_of_true_goals.shape[0], 9, -1),
                actions_batch.long())
            action_true_goal_loss.backward()

            optimizer.step()

            epoch_all_actions_loss += action_loss.item()
            epoch_goal_loss += goal_loss.item()
            # epoch_action_prediction_performance += torch.eq(actions_batch,
            #                                                 torch.argmax(actions_prob, dim=2)).sum().item() / \
            #                                        (actions_batch.shape[0] * actions_batch.shape[1])

            n_train_batch += 1
            print('epoch: ', epoch, ', batch: ', train_idx)
            global_index += 1

        validation_goal_prediction_accuracy = 0
        validation_action_prediction_accuracy = 0
        for valid_idx, data in enumerate(validation_data_generator):
            environments_batch, \
                goals_batch, \
                actions_batch, \
                needs_batch, \
                goal_reached_batch, \
                retrospective_goals_batch = data

            with torch.no_grad():
                goals_prob, actions_prob, _ = tom_net(environments_batch,
                                                      inject_true_goals=False,
                                                      goals=None)
                goals_pred = torch.argmax(goals_prob, dim=2)
                action_pred = torch.argmax(actions_prob, dim=2)

                validation_goal_prediction_accuracy += torch.eq(goals_batch, goals_pred).sum().item() / \
                                                       (goals_batch.shape[0] * goals_batch.shape[1])

                validation_action_prediction_accuracy += torch.eq(actions_batch, action_pred).sum().item() / \
                                                         (actions_batch.shape[0] * actions_batch.shape[1])
                n_validation_batch += 1

        writer.add_scalar("Train Loss/goal", epoch_goal_loss / n_train_batch, epoch)
        writer.add_scalar("Train Loss/all_action", epoch_all_actions_loss / n_train_batch, epoch)
        writer.add_scalar("Validation Accuracy/goal", validation_goal_prediction_accuracy / n_validation_batch, epoch)
        writer.add_scalar("Validation Accuracy/action", validation_action_prediction_accuracy / n_validation_batch, epoch)

    writer.flush()
    if not os.path.exists('./Model'):
        os.mkdir('./Model')
    torch.save(tom_net.cpu().state_dict(), './Model/ToM_RNN_V2.pt')
