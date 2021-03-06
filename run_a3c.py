import minerl
import gym
import argparse
import torch
import torch.optim as optim
import ray
import yaml
import os
from A3C_GRU import A3C_GRU
from ActorLearner import ActorLearner
from torch.utils.tensorboard import SummaryWriter

with open('navigate.yaml') as f:
    args = yaml.load(f, Loader=yaml.FullLoader)

# Hyperparmeters
AGENT_NUM = args['agent_num']
num_channels = args['num_channels']

T = 0
T_MAX = 1000000

def main():

    ray.init()
    global T
    global T_MAX
    save_path = os.curdir + '/trained_models/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"main thread : {device}")

    model = A3C_GRU(channels=4, num_actions=19).cuda()
    learner_model = ray.get(ray.put(model))

    actor_learners = [ActorLearner.remote(learner_model, save_path, True, **args)
     for i in range(AGENT_NUM)]

    result = [actor_learner.train.remote(T, T_MAX)
              for actor_learner in actor_learners]

    ray.get(result)

main()

