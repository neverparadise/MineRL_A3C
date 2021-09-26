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


@ray.remote(num_gpus=0.4)
class ActorLearner:
    def __init__(self, model, save_path, training, **args):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        s_batch = torch.stack(s_lst).float().to(device)
        a_batch = torch.tensor(a_lst).to(device)
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

    def calcul_loss(self):
        s, a, r, s_prime, done, batch_length = self.make_batch()
        hiddens_prime = self.actor_model.init_hidden_state(batch_size=batch_length, training=True)
        hiddens = self.actor_model.init_hidden_state(batch_size=batch_length, training=True)
        # print(f"hidden shape in train : {hiddens.shape}")

        # Calculate TD Target
        x_prime, hiddens_prime = self.actor_model.forward(s_prime, hiddens_prime)
        v_prime = self.actor_model.v(x_prime)
        td_target = r + self.GAMMA * v_prime * done

        # Calculate V
        x, hiddens = self.actor_model.forward(s, hiddens)
        v = self.actor_model.v(x) # torch.Size([1, 10, 1])
        # print(f"v shape : {v.shape}")

        delta = td_target - v
        pi = self.actor_model.pi(x, softmax_dim=2) #  torch.Size([1, 10, 19])
        # print(f"pi shape : {pi.shape}")
        a = a.unsqueeze(0)  # a : torch.Size([1, 10, 1])
        pi_a = pi.gather(2, a)             # a : torch.Size([10, 1])
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(v, td_target.detach())
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.hiddens = []
        return loss.mean().item()

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
            hidden = self.actor_model.init_hidden_state(batch_size=1, training=True)

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
                    self.put_transition((state, action_index, reward, s_prime, done ))
                    self.put_hidden(hidden)
                    state = s_prime
                    self.score += reward
                    T += 1
                    if done:
                        break

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



