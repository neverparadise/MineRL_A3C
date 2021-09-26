import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class A3C_GRU(nn.Module):
    def __init__(self, num_actions):
        super(A3C_GRU, self).__init__()
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

        self.gru_i_dim = 64  # input dimension of LSTM
        self.gru_h_dim = 64  # output dimension of LSTM
        self.gru_N_layer = 1  # number of layers of LSTM
        self.Conv2GRU = nn.Linear(linear_input_size, self.gru_i_dim)
        self.gru = nn.GRU(input_size=self.gru_i_dim, hidden_size=self.gru_h_dim, num_layers=self.gru_N_layer)
        self.fc_pi = nn.Linear(self.gru_h_dim, self.num_actions)
        self.fc_v = nn.Linear(self.gru_h_dim, 1)

    def pi(self, x, softmax_dim=1):
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        v = self.fc_v(x)
        return v

    def forward(self, x, hidden):
        if(len(x.shape) < 4):
            x = x.unsqueeze(0).to(device=device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = F.relu(self.Conv2GRU(x))
        #print(f"After Conv2GRU : {x.shape}")
        x = x.unsqueeze(0)  #
        x, new_hidden = self.gru(x, hidden)
        #print(f"After GRU : {x.shape}") batch : (1, 10, 64)
        return x, new_hidden

    def init_hidden_state(self, batch_size, training=None):

        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return torch.zeros([1, batch_size, self.gru_h_dim], device=device)
        else:
            return torch.zeros([1, 1, self.gru_h_dim], device=device)