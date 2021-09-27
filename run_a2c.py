import minerl
import gym
import argparse
import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import ray
import yaml
import os
from A3C_GRU import A3C_GRU
from copy import deepcopy
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--yaml', type=str, default="treechop", help='name of yaml file')
parse = parser.parse_args()
parse.yaml = 'navigate.yaml'

with open(parse.yaml) as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

class ActorCrictic:
    def __init__(self, model, save_path, training, writer, **args):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize learner model and actor model
        self.model = model
        self.save_path = save_path
        self.training = training

        # Hyperparams
        self.GAMMA = args['gamma']
        self.LR = args['lr']

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), self.LR)
        self.writer = writer

        # Env
        self.env_name = args['env_name']
        self.env = gym.make(self.env_name)
        self.max_epi = args['max_epi']
        self.agent_num = args['agent_num']
        self.score = 0

        # datas for learning
        self.transitions = []
        self.hiddens = []

        # time_step counter
        self.ts_max = 10
        self.seq_len = 1

    def put_transition(self, item):
        self.transitions.append(item)

    def put_hidden(self, item):
        self.hiddens.append(item)

    def make_batch(self):
        device = 'cuda'
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for i, transition in enumerate(self.transitions):
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append(([r]))
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        s_batch = torch.stack(s_lst).float().to(device) #
        a_batch = torch.tensor(a_lst).to(device) #
        # print(a_batch.shape) torch.Size([10, 1])
        r_batch = torch.tensor(r_lst).float().to(device)
        s_prime_batch = torch.stack(s_prime_lst).float().to(device)
        done_batch = torch.tensor(done_lst).float().to(device)

        batch_length = len(self.transitions)
        del self.transitions
        self.transitions = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch, batch_length

    def make_19action(self, env, action_index):
            # Action들을 정의
            action = env.action_space.noop()
            if (action_index == 0):
                action['camera'] = [0, -5]
                action['attack'] = 0
            elif (action_index == 1):
                action['camera'] = [0, -5]
                action['attack'] = 1
            elif (action_index == 2):
                action['camera'] = [0, 5]
                action['attack'] = 0
            elif (action_index == 3):
                action['camera'] = [0, 5]
                action['attack'] = 1
            elif (action_index == 4):
                action['camera'] = [-5, 0]
                action['attack'] = 0
            elif (action_index == 5):
                action['camera'] = [-5, 0]
                action['attack'] = 1
            elif (action_index == 6):
                action['camera'] = [5, 0]
                action['attack'] = 0
            elif (action_index == 7):
                action['camera'] = [5, 0]
                action['attack'] = 1

            elif (action_index == 8):
                action['forward'] = 0
                action['jump'] = 1
            elif (action_index == 9):
                action['forward'] = 1
                action['jump'] = 1
            elif (action_index == 10):
                action['forward'] = 1
                action['attack'] = 0
            elif (action_index == 11):
                action['forward'] = 1
                action['attack'] = 1
            elif (action_index == 12):
                action['back'] = 1
                action['attack'] = 0
            elif (action_index == 13):
                action['back'] = 1
                action['attack'] = 1
            elif (action_index == 14):
                action['left'] = 1
                action['attack'] = 0
            elif (action_index == 15):
                action['left'] = 1
                action['attack'] = 1
            elif (action_index == 16):
                action['right'] = 1
                action['attack'] = 0
            elif (action_index == 17):
                action['right'] = 1
                action['attack'] = 1
            else:
                action['attack'] = 1

            return action
    def make_sequence(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for i, transition in enumerate(self.transitions):
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(torch.tensor([[a]], device=device))
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        #s_batch = torch.cat(s_lst, axis=1).float().to(device)
        s_batch = s_lst
        #a_batch = torch.tensor(a_lst, axis=1).to(device) # torch.Size([10, 1, 1])
        a_batch = a_lst
        r_batch = torch.tensor(r_lst).float().to(device)
        #s_prime_batch = torch.cat(s_prime_lst, axis=1).float().to(device)
        s_prime_batch = s_prime_lst
        done_batch = torch.tensor(done_lst).float().to(device)

        seq_length = len(self.transitions)
        del self.transitions
        self.transitions = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch, seq_length

    def calcul_loss(self):
        x, a, r, x_prime, done, seq_length = self.make_sequence()
        with torch.autograd.set_detect_anomaly(True):
            total_loss = torch.zeros(1, device=device)
            for i in range(self.ts_max):
                # Calculate TD Target
                print(f"x shape : {x[i].shape}")

                v_prime = self.model.v(x_prime[i])
                td_target = r[i] + self.GAMMA * v_prime * done[i]

                # Calculate V
                v = self.model.v(x[i])  # torch.Size([1, 1, 1])
                delta = td_target - v
                pi = self.model.pi(x[i], softmax_dim=2) #  torch.Size([1, 1, 19])
                print(f"pi shape : {pi.shape}")
                 # a : list contain 10 items : torch.Size([1, ,1])
                action = a[i]  # action : torch.Size([1, 1])
                print(f"action shape : {action.shape}")
                pi = pi.squeeze(0)
                print(f"pi shape : {pi.shape}")
                pi_a = pi.gather(1, action)
                loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(v, td_target.detach())
                loss = loss.mean()
                total_loss += loss
            self.optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            self.optimizer.step()

        return total_loss.item()

    def calcul_loss_origin(self):
        with torch.autograd.set_detect_anomaly(True):
            s, a, r, s_prime, done, batch_length = self.make_batch()
            hiddens_prime = self.model.init_hidden_state(batch_size=batch_length, training=True)
            hiddens = self.model.init_hidden_state(batch_size=batch_length, training=True)

            # Calculate TD Target
            x_prime, hiddens_prime = self.model.forward(s_prime, hiddens_prime)
            v_prime = self.model.v(x_prime)
            td_target = r + self.GAMMA * v_prime * done
            #print(f"td target shape : {td_target.shape}")

            # Calculate V
            x, hiddens = self.model.forward(s, hiddens)
            v = self.model.v(x) # torch.Size([1, 10, 1])
            #print(f"v shape : {v.shape}")

            delta = td_target - v
            #print(f"delta shape : {delta.shape}")
            pi = self.model.pi(x, softmax_dim=2) #  torch.Size([1, 10, 19])
            #print(f"pi shape : {pi.shape}")
            a = a.unsqueeze(0)  # a : torch.Size([1, 10, 1])
            pi_a = pi.gather(2, a)
            loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(v, td_target.detach())
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            self.hiddens = []
            return loss.mean().item()

    def converter(self, env_name ,observation):
        if (env_name == 'MineRLNavigateDense-v0' or
            env_name == 'MineRLNavigate-v0'):
            obs = observation['pov']
            obs = obs / 255.0 # [64, 64, 3]
            compass_angle = observation['compassAngle']
            compass_angle_scale = 180
            compass_scaled = compass_angle / compass_angle_scale
            compass_channel = np.ones(shape=list(obs.shape[:-1]) + [1], dtype=obs.dtype) * compass_scaled
            obs = np.concatenate([obs, compass_channel], axis=-1)
            obs = torch.from_numpy(obs)
            obs = obs.permute(2, 0, 1)
            return obs.float().cuda()
        else:
            obs = observation['pov']
            obs = obs / 255.0
            obs = torch.from_numpy(obs)
            obs = obs.permute(2, 0, 1)
            return obs.float().cuda()

    def save_model(self):
        torch.save({'model_state_dict': self.model.state_dict()}, self.save_path + 'A3C_MineRL.pth')
        print("model saved")

    def train(self):
        n_epi = 0
        while n_epi < self.max_epi:
            loss = 0
            device = 'cuda'
            done = False
            state = self.env.reset()
            state = self.converter(self.env_name, state)
            # RNN must have a shape like sequence length, batch size, input size
            hidden = self.model.init_hidden_state(batch_size=1, training=True)
            while not done:
                for t in range(self.ts_max):
                    x, hidden = self.model.forward(state, hidden)
                    # because of rnn input, softmax_dim needs to be 2
                    prob = self.model.pi(x, softmax_dim=2) # torch.Size([1, 1, 19])
                    # print(f"prob shape : {prob.shape}")
                    m = Categorical(prob)
                    action_index = m.sample().item()
                    action = self.make_19action(self.env, action_index)
                    s_prime, reward, done, _ = self.env.step(action)
                    s_prime = self.converter(self.env_name, s_prime)
                    x_prime, hidden = self.model.forward(s_prime, hidden)
                    self.put_transition((x, action_index, reward, x_prime, done ))
                    state = s_prime
                    self.score += reward

                    if done:
                        break
                loss = self.calcul_loss()
                self.writer.add_scalar('Loss/train', loss, n_epi)
                self.writer.add_scalar('Rewards/train', self.score, n_epi)

                print(self.score)
                if done:
                    break
            n_epi += 1

            # Write down loss, rewards

            self.save_model()
            self.score = 0.0
            print(self.score)

        self.env.close()



def main():
    writer = SummaryWriter()
    model = A3C_GRU(channels=4, num_actions=19).cuda()
    save_path = os.curdir + '/trained_models/'
    agent = ActorCrictic(model,save_path,False, writer, **args)
    agent.train()

main()