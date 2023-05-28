import matplotlib.pyplot as plt
import torch
import numpy as np


def get_figure_title(objects_color_name, need):
    title = '$n_{0}: {1:.2f}'.format('{' + objects_color_name[0] + '}', need[0, 0])
    for i in range(1, len(objects_color_name)):
        title += ", n_{0}: {1:.2f}$".format('{' + objects_color_name[i] + '}', need[0, i])
    return title


def visualizer(fig_height, fig_width, env_input, true_goals, true_actions, true_needs, pred_goals, pred_actions):
    color_options = [[1, 0, .2], [0, .8, .2], [1, 1, 1]]
    objects_color_name = ['red', 'green']
    all_actions = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
                            [1, 1], [-1, -1], [-1, 1], [1, -1]])

    fig, ax = plt.subplots(fig_height, fig_width, figsize=(15, 12))

    for step in range(fig_height * fig_width):
        fig_col = step % fig_width
        fig_row = step // fig_width

        env_height = env_input[step].shape[2]
        env_width = env_input[step].shape[3]
        agent_location = torch.argwhere(env_input[step][0, 0, :, :]).squeeze()
        object_locations = torch.argwhere(env_input[step][0, 1:, :, :]).squeeze()

        step_pred_action = torch.argmax(pred_actions[step].squeeze()).item()
        step_pred_goal = torch.argmax(pred_goals[step].squeeze()).item()
        # agent:
        ax[fig_row, fig_col].scatter(agent_location[1].item(), agent_location[0].item(),
                                     marker='$\U0001F601$', s=40, facecolor='#8A2BE2')

        for obj in range(object_locations.shape[0]):
            ax[fig_row, fig_col].scatter(object_locations[obj, 2].item(), object_locations[obj, 1].item(),
                                         marker='*', s=40, facecolor=color_options[obj])

        def get_scale(action):
            if action in [1, 2]:  # up or down
                scale = [0.2, 0]
            else:
                scale = [0, 0.2]
            return scale
        pred_scale = get_scale(step_pred_action)
        true_scale = get_scale(true_actions[step])

        ax[fig_row, fig_col].arrow(agent_location[1].item() + pred_scale[0], agent_location[0].item() + pred_scale[1],
                                   all_actions[step_pred_action][1] / 2, all_actions[step_pred_action][0] / 2,
                                   color=color_options[step_pred_goal], head_width=.1, linestyle='-',
                                   width=0.01, alpha=1)

        ax[fig_row, fig_col].arrow(agent_location[1].item() - true_scale[0], agent_location[0].item() - true_scale[1],
                                   all_actions[true_actions[step]][1] / 2, all_actions[true_actions[step]][0] / 2,
                                   color=color_options[true_goals[step]], head_width=.1, linestyle=':',
                                   width=0.005, alpha=.3)

        ax[fig_row, fig_col].set_title(get_figure_title(objects_color_name, true_needs[step]), fontsize=10)

        ax[fig_row, fig_col].set_xticks(list(np.arange(-1, env_width + 1, dtype=np.float32)))
        # ax[fig_row, fig_col].xaxis.set_major_locator(plt.NullLocator())
        # ax[fig_row, fig_col].xaxis.set_major_locator(plt.NullLocator())

        ax[fig_row, fig_col].set_yticks(list(np.arange(-1, env_height + 1, dtype=np.float32)))
        # ax[fig_row, fig_col].yaxis.set_major_locator(plt.NullLocator())
        # ax[fig_row, fig_col].yaxis.set_major_locator(plt.NullLocator())

        ax[fig_row, fig_col].tick_params(length=0)
        ax[fig_row, fig_col].invert_yaxis()

        ax[fig_row, fig_col].set(adjustable='box')

    fig.suptitle('Predicted: ->  Truth: -->\n', y=.99)
    plt.tight_layout(pad=0.4, w_pad=6, h_pad=.8)

    return fig, ax
