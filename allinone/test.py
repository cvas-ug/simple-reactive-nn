import gym
import random
import numpy as np
import argparse
from actorcitic4 import Actor, act
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import os
import random
import torch.nn as nn


SAVEPATH1 = os.getcwd() + '/train/actor_params.pth'

env = gym.make("FetchPickAndPlace-v1")
env2 = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--use-cuda',default=True,
                    help='run on gpu.')
parser.add_argument('--save-path1',default=SAVEPATH1,
                    help='model save interval (default: {})'.format(SAVEPATH1))
parser.add_argument('--max-grad-norm', type=float, default=250,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='discount factor for rewards (default: 0.9)')
parser.add_argument('--tau', type=float, default=1.00,
                    help='parameter for GAE (default: 1.00)')
args = parser.parse_args() 

model = Actor()

if args.use_cuda:
    model.cuda()

torch.cuda.manual_seed_all(25)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

if os.path.isfile(args.save_path1):
    print('Loading A3C parametets ...')
    model.load_state_dict(torch.load(args.save_path1))

for p in model.fc1.parameters():
    p.requires_grad = False
for p in model.fc2.parameters():
    p.requires_grad = False

FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor

model.eval()

max_eps = 200000
max_steps = 50
ep_numb = 0
done = True
success = 0         
  
while ep_numb < max_eps:
    ep_numb +=1
    lastObs = env.reset()
    goal = lastObs['desired_goal']
    objectPos = lastObs['observation'][3:6]
    object_rel_pos = lastObs['observation'][6:9]
    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object    
    timeStep = 0 #count the total number of timesteps
    state_inp = torch.from_numpy(env2.observation(lastObs)).type(FloatTensor)
    Ratio=[]
    while np.linalg.norm(object_oriented_goal) >= 0.015 and timeStep <= env._max_episode_steps:
        env.render()
        action = [0, 0, 0, 0, 0, 0]
        act_tensor, ratio = act(state_inp, model, True, False)       
        #print(act_tensor)     

        Ratio.append(ratio.cpu().detach().numpy())
        for i in range(len(object_oriented_goal)):
            action[i] = act_tensor[i].cpu().detach().numpy()

        object_oriented_goal = object_rel_pos.copy()            
        object_oriented_goal[2] += 0.03
        
        action[3] = 0.05
        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1
        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]
        state_inp = torch.from_numpy(env2.observation(obsDataNew)).type(FloatTensor)
        if timeStep >= env._max_episode_steps: break
    
    while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
        env.render()
        action = [0, 0, 0, 0, 0, 0]
        act_tensor, ratio = act(state_inp, model, False, False)    
        Ratio.append(ratio.cpu().detach().numpy())
        for i in range(len(object_oriented_goal)):
            action[i] = act_tensor[i].cpu().detach().numpy()
        
        action[3]= -0.01 
        action[5] = act_tensor[3].cpu().detach().numpy()
        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]
        state_inp = torch.from_numpy(env2.observation(obsDataNew)).type(FloatTensor)
        if timeStep >= env._max_episode_steps: break

    while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
            
        env.render()
        action = [0, 0, 0, 0, 0, 0]
        act_tensor, ratio = act(state_inp, model, False, True)    
        Ratio.append(ratio.cpu().detach().numpy())

        for i in range(len(goal - objectPos)):
            action[i] = act_tensor[i].cpu().detach().numpy()
        
        action[3] = -0.01
        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1
        state_inp = torch.from_numpy(env2.observation(obsDataNew)).type(FloatTensor)
        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]
        if timeStep >= env._max_episode_steps: break
    
    while True: #limit the number of timesteps in the episode to a fixed duration
        env.render()
        action = [0, 0, 0, 0, 0, 0]
        action[3] = -0.01 # keep the gripper closed

        obsDataNew, reward, done, info = env.step(action)
        timeStep += 1

        objectPos = obsDataNew['observation'][3:6]
        object_rel_pos = obsDataNew['observation'][6:9]

        if timeStep >= env._max_episode_steps: break

    if info['is_success'] == 1.0:
        success +=1
    if done:
        if ep_numb % 100==0:            
            print("num episodes {}, success {}".format(ep_numb, success))