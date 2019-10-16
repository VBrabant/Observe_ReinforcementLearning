import random

import torch
import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self, h, w, hidden_lstm_size, outputs):
        super(MyModel, self).__init__()
        self.hidden_lstm_size = hidden_lstm_size
        self.num_outputs = outputs
        
        self.conv = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(16, 32, kernel_size=5, stride=2),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=5, stride=2),
                nn.BatchNorm2d(32),
                nn.ELU()
                )
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        
        linear_input_size = convw * convh * 32
        self.linear_input_size = linear_input_size
        
        self.lstm = nn.LSTM(self.linear_input_size, self.hidden_lstm_size, batch_first=True)
        
        self.advantage = nn.Sequential(
            nn.Linear(self.hidden_lstm_size, outputs)
        )
        
        self.value = nn.Sequential(
            nn.Linear(self.hidden_lstm_size, 1)
        )
        
    def forward(self, x, hidden):
        x = self.conv(x)
        x = x.reshape(1, -1, self.linear_input_size)
        x, hidden = self.lstm(x, hidden)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean(), hidden

    def act(self, state, hidden, epsilon):
        state = torch.FloatTensor(state).unsqueeze(0)
        q_value, hidden_act = self.forward(state, hidden)
        if random.random() > epsilon: # apply epsilon policy
            action  = int(q_value[0].max(1)[1].data[0])
        else:
            action = int(random.randrange(self.num_outputs))
        return action, hidden_act