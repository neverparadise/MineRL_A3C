import minerl
import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.optim as optim
import os
from copy import deepcopy
import ray
from torch.utils.tensorboard import SummaryWriter
from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@ray.remote(num_gpus=0.4)
class ActorLearner:
    def __init__(self, model, save_path, training, **args):
        print(f"actor_learner thread : {device}")
        # Initialize learner model and actor model
        self.learner_model = model
        self.actor_model = deepcopy(model)
        self.save_path = save_path
        self.training = training

        # Hyperparams and optimizer
        self.GAMMA = args['gamma']
        self.LR = args['lr']
        self.optimizer = optim.Adam(self.actor_model.parameters(), self.LR)
        self.writer = SummaryWriter('runs/a3c/')

        # Env
        import minerl
        self.env_name = args['env_name']
        self.env = gym.make(self.env_name)
        self.max_epi = args['max_epi']
        self.agent_num = args['agent_num']
        self.score = 0


        # time_step counter
        self.ts_max = 10
        self.seq_len = 1

        # datas for learning
        self.transitions = deque(maxlen=self.ts_max)


    def put_transition(self, item):
        self.transitions.append(item)

    def make_sequence(self):
        device = 'cuda'
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
        x, a, r, x_prime, done, seq_length = self.make_sequence()
        total_loss = torch.zeros(1, device=device)
        for i in range(self.ts_max):
            # Calculate TD Target
            print(f"x shape : {x[i].shape}")

            v_prime = self.actor_model.v(x_prime[i])
            td_target = r[i] + self.GAMMA * v_prime * done[i]

            # Calculate V
            v = self.actor_model.v(x[i])  # torch.Size([1, 1, 1])
            delta = td_target - v
            pi = self.actor_model.pi(x[i], softmax_dim=2) #  torch.Size([1, 1, 19])
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

    def accumulate_gradients(self):
        for actor_net, learner_net in zip(self.actor_model.named_parameters(), self.learner_model.named_parameters()):
            learner_net[1].grad = deepcopy(actor_net[1].grad)

    def converter(self, env_name, observation):
        if (env_name == 'MineRLNavigateDense-v0' or
                env_name == 'MineRLNavigate-v0'):
            obs = observation['pov']
            obs = obs / 255.0  # [64, 64, 3]
            compass_angle = observation['compassAngle']
            compass_angle_scale = 180
            compass_scaled = compass_angle / compass_angle_scale
            compass_channel = np.ones(shape=list(obs.shape[:-1]) + [1], dtype=obs.dtype) * compass_scaled
            obs = np.concatenate([obs, compass_channel], axis=-1)
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
        torch.save({'model_state_dict': self.learner_model.state_dict()}, self.save_path + 'A3C_MineRL.pth')
        print("model saved")

    def train(self, T, T_max):
        n_epi = 0
        while T < T_max and n_epi < self.max_epi:
            loss = 0
            device = 'cuda'
            done = False
            state = self.env.reset()
            state = self.converter(self.env_name, state)
            # RNN must have a shape like sequence length, batch size, input size
            hidden = self.actor_model.init_hidden_state(training=True)

            while not done:
                for t in range(self.ts_max):
                    x, hidden = self.actor_model.forward(state, hidden)
                    # because of rnn input, softmax_dim needs to be 2
                    prob = self.actor_model.pi(x, softmax_dim=2) # torch.Size([1, 1, 19])
                    # print(f"prob shape : {prob.shape}")
                    m = Categorical(prob)
                    action_index = m.sample().item()
                    action = self.make_19action(self.env, action_index)
                    s_prime, reward, done, _ = self.env.step(action)
                    s_prime = self.converter(self.env_name, s_prime)
                    x_prime, hidden = self.actor_model.forward(s_prime, hidden)
                    self.put_transition((x, action_index, reward, x_prime, done))
                    state = s_prime
                    self.score += reward
                    T += 1
                    if done:
                        break
                # compute last hidden for new_hiddens l ist
                loss = self.calcul_loss()
                self.writer.add_scalar('Loss/train', loss)
                self.accumulate_gradients()
                if done:
                    break
            self.writer.add_scalar('Rewards/train', self.score, n_epi)
            print(f"Episode {n_epi} : {self.score}")
            n_epi += 1

            # Write down loss, rewards
            self.save_model()
            self.score = 0.0

        self.env.close()



