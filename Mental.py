import torch.nn as nn
import torch
import Utilities


class MentalNet(nn.Module):
    def __init__(self, agent_states_num, env_states_num, mental_states_num, layers_num):
        super(MentalNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.h_0 = None
        self.c_0 = None
        self.num_layers = layers_num
        self.lstm = nn.LSTM(input_size=agent_states_num + env_states_num,
                            hidden_size=mental_states_num,
                            num_layers=layers_num,
                            batch_first=True)
        self.hidden_size = mental_states_num

    def forward(self, environment, agent, reinitialize):
        # maybe we should save h_0 and c_0 for next predictions, as the sequence is basically infinite
        batch_size = environment.shape[0]
        if reinitialize:
            self.h_0 = torch.zeros((self.num_layers, batch_size, self.hidden_size),
                                   requires_grad=True, device=self.device)
            self.c_0 = torch.zeros((self.num_layers, batch_size, self.hidden_size),
                                   requires_grad=True, device=self.device)

        environment_combined = torch.concat([environment, agent], dim=2)
        mental_states, (h_n, c_n) = self.lstm(environment_combined, (self.h_0, self.c_0))

        hidden = (h_n, c_n)
        return mental_states, hidden
