import os.path
import matplotlib.pyplot as plt
import numpy as np
import torch
import Utilities
from torch.utils.tensorboard import SummaryWriter
from Visualizer import visualizer
from ObjectFactory import ObjectFactory


def load_tom_net(factory, utility):
    tom_net = factory.get_tom_net()
    weights = torch.load('./Model/ToM_RNN_V2.pt')
    tom_net.load_state_dict(weights)
    return tom_net


def get_collection_distances(map1, map2):
    p1 = torch.argwhere(map1)
    p2 = torch.argwhere(map2)
    return torch.cdist(p1.float(), p2[:, 1:].float())


def test(test_data_generator):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utility = Utilities.Utilities()
    params = utility.params
    writer = SummaryWriter()
    test_figures_dir = './Test/'
    if not os.path.exists(test_figures_dir):
        os.mkdir(test_figures_dir)

    # agent_appearance = torch.tensor(get_agent_appearance(),
    #                                 dtype=torch.float32)  # add one dimension for batch
    factory = ObjectFactory(utility=utility)
    tom_net = load_tom_net(factory, utility).to(device)
    global_index = 0
    width, height = 5, 5
    grids_in_fig = width*height
    env_input = []
    non_equal_distance_to_objects = []
    goal_is_object = []
    goal_is_nearest_object = []

    pred_goal_is_object = []
    pred_goal_is_nearest_object = []
    goal_reached_in_last_step = True

    pred_goals, pred_actions = [], []
    true_goals, true_needs, true_actions = [], [], []
    for test_idx, data in enumerate(test_data_generator):
        # environment_batch.shape: [batch_size, step_num, objects+agent(s), height, width]
        # target_goal: 2 is staying
        environments_batch, goals_batch, actions_batch, needs_batch, reached_goal_batch, _, _ = data

        environments_batch = environments_batch.to(device)
        goals_batch = goals_batch.to(device)
        actions_batch = actions_batch.to(device)
        needs_batch = needs_batch.to(device)
        reached_goal_batch = reached_goal_batch.to(device)

        step_num = environments_batch.shape[1]

        seq_start = True

        for step in range(step_num):
            goals, goals_prob, actions, actions_prob = tom_net(environments_batch[:, step, :, :, :].unsqueeze(dim=1),
                                                               seq_start)
            seq_start = False

            true_goals.append(goals_batch[:, step])
            true_actions.append(actions_batch[:, step])
            true_needs.append(needs_batch[:, step, :])
            env_input.append(environments_batch[:, step, :, :, :])

            pred_goals.append(goals_prob)
            pred_actions.append(actions_prob)

            if (global_index + 1) % grids_in_fig == 0:
                fig, ax = visualizer(height, width, env_input, true_goals,
                                     true_actions, true_needs, pred_goals, pred_actions)
                env_input = []
                pred_goals, pred_actions = [], []
                true_goals, true_needs, true_actions = [], [], []
                fig.savefig('{0}/{1}_{2}.png'.format(test_figures_dir, global_index-width*height+1, global_index+1))
                plt.close()

            # Check if the goal object is the nearest one
            if goal_reached_in_last_step:
                distance_to_objects = get_collection_distances(environments_batch[:, step, 0, :, :].squeeze(),
                                                               environments_batch[:, step, 1:, :, :].squeeze())
                pred_goal_index = torch.argmax(goals_prob.squeeze()).item()
                true_goal_index = goals_batch[:, step]
                pred_goal_is_object.append(pred_goal_index < params.GOAL_NUM)  # selected goal is an object and not staying
                goal_is_object.append(true_goal_index.item() < params.GOAL_NUM)

                if not torch.all(distance_to_objects == distance_to_objects[0, 0].item(), dim=1):
                    non_equal_distance_to_objects.append(True)
                else:
                    non_equal_distance_to_objects.append(False)

                if torch.argmin(distance_to_objects) == pred_goal_index:  # This implies that the goal is an object
                    pred_goal_is_nearest_object.append(True)
                else:
                    pred_goal_is_nearest_object.append(False)

                if torch.argmin(distance_to_objects) == true_goal_index:
                    goal_is_nearest_object.append(True)
                else:
                    goal_is_nearest_object.append(False)
            ###
            global_index += 1
            if goal_reached_in_last_step and goals_batch[0, step].item() == params.GOAL_NUM:
                goal_reached_in_last_step = True
            else:
                goal_reached_in_last_step = reached_goal_batch[0, step].item()

        #     if global_index == 50:
        #         break
        # if global_index == 50:
        #     break

    goal_is_object = np.array(goal_is_object)
    goal_is_nearest_object = np.array(goal_is_nearest_object)
    pred_goal_is_object = np.array(pred_goal_is_object)
    pred_goal_is_nearest_object = np.array(pred_goal_is_nearest_object)
    non_equal_distance_to_objects = np.array(non_equal_distance_to_objects)

    def visualize_stat(ax, xs, ys, x_axis, y_axis, title):
        font_size = 10
        ax[x_axis, y_axis].set_title(title, {'fontsize': font_size})
        bar = ax[x_axis, y_axis].bar(xs, ys)
        ax[x_axis, y_axis].bar_label(bar, padding=1.5, fmt='%.2f')
        ax[x_axis, y_axis].set_ylim(top=1.35)
        ax[x_axis, y_axis].set_yticks([])

    fig, ax = plt.subplots(3, 3, figsize=(10, 8))
    ax[0, 0].axis('off')
    ax[0, 2].axis('off')

    visualize_stat(ax, ['Equal', 'Different'],
                   [1 - non_equal_distance_to_objects.sum() / non_equal_distance_to_objects.shape[0],
                    non_equal_distance_to_objects.sum() / non_equal_distance_to_objects.shape[0]], 0, 1,
                   'Distance to objects')

    visualize_stat(ax, ['Object', 'Staying'],
                   [
                       goal_is_object[~non_equal_distance_to_objects].sum() /
                       goal_is_object[~non_equal_distance_to_objects].shape[0],
                       1 - goal_is_object[~non_equal_distance_to_objects].sum() /
                       goal_is_object[~non_equal_distance_to_objects].shape[0]], 1, 0,
                   'True goal in equal \ndistance situation')

    visualize_stat(ax, ['Object', 'Staying'],
                   [
                       pred_goal_is_object[~non_equal_distance_to_objects].sum() / pred_goal_is_object[~non_equal_distance_to_objects].shape[0],
                       1 - pred_goal_is_object[~non_equal_distance_to_objects].sum() / pred_goal_is_object[~non_equal_distance_to_objects].shape[0]],
                   2, 0, 'Predicted goal in equal \ndistance situation')

    visualize_stat(ax, ['Object', 'Staying'],
                   [
                       goal_is_object[non_equal_distance_to_objects].sum() / goal_is_object[non_equal_distance_to_objects].shape[0],
                       1 - goal_is_object[non_equal_distance_to_objects].sum() / goal_is_object[non_equal_distance_to_objects].shape[0]],
                   1, 2, 'True goal in non-equal \ndistance situation')

    visualize_stat(ax, ['Object', 'Staying'],
                   [
                       pred_goal_is_object[non_equal_distance_to_objects].sum() / pred_goal_is_object[non_equal_distance_to_objects].shape[0],
                       1 - pred_goal_is_object[non_equal_distance_to_objects].sum() / pred_goal_is_object[non_equal_distance_to_objects].shape[0]],
                   2, 2, 'Predicted goal in \nnon-equal distance situation')

    visualize_stat(ax, ['Nearest', 'Furthest '],
                   [
                       goal_is_nearest_object[non_equal_distance_to_objects & goal_is_object].sum() /
                       goal_is_nearest_object[non_equal_distance_to_objects & goal_is_object].shape[0],
                       1 - goal_is_nearest_object[non_equal_distance_to_objects & goal_is_object].sum() /
                       goal_is_nearest_object[non_equal_distance_to_objects & goal_is_object].shape[0]], 1, 1,
                   'True goal object in \nnon-equal distance situation')

    visualize_stat(ax, ['Nearest', 'Furthest '],
                   [
                       pred_goal_is_nearest_object[non_equal_distance_to_objects & pred_goal_is_object].sum() /
                       pred_goal_is_nearest_object[non_equal_distance_to_objects & pred_goal_is_object].shape[0],
                       1 - pred_goal_is_nearest_object[non_equal_distance_to_objects & pred_goal_is_object].sum() /
                       pred_goal_is_nearest_object[non_equal_distance_to_objects & pred_goal_is_object].shape[0]],
                   2, 1, 'Predicted goal object in \nnon-equal distance situation')

    plt.tight_layout(h_pad=2., w_pad=4.)
    fig.savefig('{0}/stats.png'.format(test_figures_dir))