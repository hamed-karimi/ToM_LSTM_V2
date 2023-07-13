import torch
from Environment import EnvironmentNet
from Agent import AgentNet
from Traits import TraitsNet
from Mental import MentalNet
from Goal import GoalNet
from Action import ActionNet
from ToM import ToMNet
import torch.nn.init as init


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


class ObjectFactory:
    def __init__(self, utility):
        self.action_net = None
        self.goal_net = None
        self.agent = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.environment_net = None
        self.mental_net = None
        self.trait_nest = None
        self.tom_net = None
        self.params = utility.params

    def get_environment_net(self):
        self.environment_net = EnvironmentNet(height=self.params.HEIGHT,
                                              goal_num=self.params.GOAL_TYPE_NUM,
                                              env_conv_channel_size=self.params.ENVIRONMENT_CONV1_OUT_CHANNEL,
                                              states_num=self.params.ENVIRONMENT_LSTM_STATES_NUM,
                                              layers_num=self.params.ENVIRONMENT_LSTM_LAYERS_NUM).to(self.device)
        self.environment_net.apply(weights_init_orthogonal)
        return self.environment_net

    def get_agent_net(self):
        self.agent = AgentNet(states_num=self.params.AGENT_STATES_NUM).to(self.device)
        self.agent.apply(weights_init_orthogonal)
        return self.agent

    def get_traits_net(self):
        trait_net = TraitsNet(agent_size=self.params.AGENT_SIZE,
                              traits_num=self.params.TRAITS_NUM).to(self.device)
        trait_net.apply(weights_init_orthogonal)
        return trait_net

    def get_mental_net(self):
        self.mental_net = MentalNet(agent_states_num=self.params.AGENT_STATES_NUM,
                                    env_states_num=self.params.ENVIRONMENT_LSTM_STATES_NUM,
                                    mental_states_num=self.params.MENTAL_LSTM_STATES_NUM,
                                    layers_num=self.params.MENTAL_LSTM_LAYERS_NUM).to(self.device)
        self.mental_net.apply(weights_init_orthogonal)
        return self.mental_net

    def get_goal_net(self):
        self.goal_net = GoalNet(states_size=self.params.MENTAL_LSTM_STATES_NUM,
                                goal_num=self.params.GOAL_TYPE_NUM).to(self.device)
        self.goal_net.apply(weights_init_orthogonal)
        return self.goal_net

    def get_action_net(self):
        self.action_net = ActionNet(states_size=self.params.ENVIRONMENT_LSTM_STATES_NUM
                                                + self.params.GOAL_TYPE_NUM + 1
                                                + self.params.AGENT_STATES_NUM).to(self.device)
        self.action_net.apply(weights_init_orthogonal)
        return self.action_net

    def get_tom_net(self):
        self.tom_net = ToMNet().to(self.device)
        self.tom_net.apply(weights_init_orthogonal)
        return self.tom_net

    def zeros(self, shape, dtype=None):
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def ones(self, shape, dtype=None):
        return torch.ones(shape, dtype=dtype, device=self.device)