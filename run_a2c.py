import minerl
import gym
import argparse
import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.nn.functional as F

import ray
import yaml
import os
from A3C_GRU import A3C_GRU
from copy import deepcopy

with open('treechop.yaml') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

class ActorCrictic:
    def __init__(self, model, save_path, tested, **args):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize learner model and actor model
        self.model = model
        self.save_path = save_path
        self.tested = tested

        # Hyperparams
        self.GAMMA = args['gamma']
        self.LR = args['lr']

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), self.LR)

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
        self.batch_size = self.ts_max
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


        del self.transitions
        self.transitions = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

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

    def calcul_loss(self):
        with torch.autograd.set_detect_anomaly(True):
            hiddens = torch.cat(self.hiddens, 1)
            print(f"hidden shape in train : {hiddens.shape}")
            # hiddens must be tuple
            hiddens = tuple(each.data for each in hiddens)
            s, a, r, s_prime, done = self.make_batch()

            # Calculate TD Target
            x_prime, new_hidden_prime = self.model.forward(s_prime, hiddens)
            v_prime = self.model.v(x_prime)
            td_target = r + self.GAMMA * v_prime * done
            print(f"td target shape : {td_target.shape}")

            # Calculate V
            x, new_hidden = self.model.forward(s, hiddens)
            v = self.model.v(x) # torch.Size([1, 10, 1])
            print(f"v shape : {v.shape}")

            delta = td_target - v
            print(f"delta shape : {delta.shape}")
            pi = self.model.pi(x, softmax_dim=2) #  torch.Size([1, 10, 19])
            print(f"pi shape : {pi.shape}")
            a = a.unsqueeze(0)  # a : torch.Size([1, 10, 1])
            pi_a = pi.gather(2, a)
            loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(v, td_target.detach())
            loss = loss.mean()
            loss.backward(retain_graph=True)

            self.hiddens = []
            return loss.mean().item()

    def converter(self, env_name ,observation):
        if env_name == 'MineRLNavigate-v0':
            obs = observation
            obs = obs / 255.0
            obs = torch.from_numpy(obs)
            obs = obs.permute(2, 0, 1)
            return obs.float()
        else:
            obs = observation['pov']
            obs = obs / 255.0
            obs = torch.from_numpy(obs)
            obs = obs.permute(2, 0, 1)
            return obs.float()

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
                    self.put_transition((state, action_index, reward, s_prime, done ))
                    self.put_hidden(hidden)
                    state = s_prime
                    self.score += reward

                    if done:
                        break
                #self.optimizer.zero_grad()
                #loss = self.calcul_loss()
                #self.optimizer.step()

            # Write down loss, rewards

            self.save_model()
            self.score = 0.0

        self.env.close()



def main():
    model = A3C_GRU(19).cuda()
    save_path = os.curdir + '/trained_models/'
    agent = ActorCrictic(model,save_path,False, **args)
    agent.train()

main()