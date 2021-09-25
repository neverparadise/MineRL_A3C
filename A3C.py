import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class A3C_LSTM(nn.Module):
    def __init__(self, num_actions):
        super(A3C_LSTM, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(3, 8, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv_size = conv2d_size_out(64, 8, 4)
        conv_size = conv2d_size_out(conv_size, 4, 2)
        conv_size = conv2d_size_out(conv_size, 3, 1)
        linear_input_size = conv_size * conv_size * 32 # 4 x 4 x 32 = 512

        self.lstm_i_dim = 64  # input dimension of LSTM
        self.lstm_h_dim = 64  # output dimension of LSTM
        self.lstm_N_layer = 1  # number of layers of LSTM
        self.Conv2LSTM = nn.Linear(linear_input_size, self.lstm_i_dim)
        self.lstm = nn.LSTM(input_size=self.lstm_i_dim, hidden_size=self.lstm_h_dim, num_layers=self.lstm_N_layer)

        self.fc_pi = nn.Linear(self.lstm_h_dim, self.num_actions)
        self.fc_v = nn.Linear(self.lstm_h_dim, 1)

    def pi(self, x, softmax_dim=1):
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        prob = prob.squeeze(1)
        return prob

    def v(self, x):
        v = self.fc_v(x)
        v = v.squeeze(1)
        return v

    def forward(self, x, hidden):
        if(len(x.shape) < 4):
            x = x.unsqueeze(0).to(device=device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.Conv2LSTM(x))
        x = x.unsqueeze(1)  #
        x, new_hidden = self.lstm(x, hidden)
        return x, new_hidden