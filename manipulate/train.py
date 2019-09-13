import gym
import random
import numpy as np
import argparse
from actorcitic4 import Actor, act
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda
import matplotlib.pyplot as plt
from torch.distributions import Normal
import os
import random
import torch.nn as nn
from itertools import count
import time
import csv

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, counter, lock, optimizer=None):
    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    
    env = gym.make("FetchPickAndPlace-v1")
    env2 = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

    model = Actor()
       
    if args.use_cuda:
        model.cuda()
    torch.cuda.manual_seed_all(12)
    
    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    
    for p in model.fc1.parameters():
        p.requires_grad_(False)
    for p in model.fc2.parameters():
        p.requires_grad_(False)
    
    model.train()

    done = True       
    for num_iter in count():
        with lock:
            counter.value += 1
        #print(num_iter, counter.value)
        lastObs = env.reset()
        goal = lastObs['desired_goal']
        objectPos = lastObs['observation'][3:6]
        object_rel_pos = lastObs['observation'][6:9]
        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object    
        timeStep = 0 #count the total number of timesteps
        if rank == 0:

            if num_iter % args.save_interval == 0 and num_iter > 0:
                #print ("Saving model at :" + args.save_path)            
                torch.save(shared_model.state_dict(), args.save_path1)

        if num_iter % (args.save_interval * 2.5) == 0 and num_iter > 0 and rank == 1:    # Second saver in-case first processes crashes 
            #print ("Saving model for process 1 at :" + args.save_path)            
            torch.save(shared_model.state_dict(), args.save_path1)
        
        model.load_state_dict(shared_model.state_dict())
        criterion = nn.MSELoss()
        
        while np.linalg.norm(object_oriented_goal) >= 0.01 and timeStep <= env._max_episode_steps:
            action = [0, 0, 0, 0, 0, 0]
            object_oriented_goal = object_rel_pos.copy()
            object_oriented_goal[2] += 0.03

            for i in range(len(object_oriented_goal)):
                action[i] = object_oriented_goal[i]*6

            action[3] = 0.05 #open

            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1

            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]
        state_inp = torch.from_numpy(env2.observation(obsDataNew)).type(FloatTensor)

        while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
           
            action = [0, 0, 0, 0, 0, 0]
            act_tensor, _ = act(state_inp, model, False, False) 
            
            for i in range(len(object_rel_pos)):
                optimizer.zero_grad()
                expected = torch.from_numpy(np.array(object_rel_pos[i]*6)).type(FloatTensor)
                action[i] = act_tensor[i].cpu().detach().numpy()
                error = criterion(act_tensor[i], expected)
                (error).backward(retain_graph=True)
                ensure_shared_grads(model, shared_model)
                optimizer.step()
                #action[i] = object_rel_pos[i]*6

            optimizer.zero_grad()
            action[5] = act_tensor[3].cpu().detach().numpy()
            error2= criterion(act_tensor[3], torch.from_numpy(np.array(obsDataNew['quat'][0]/6)).type(FloatTensor))
            (error2).backward(retain_graph=True)
            ensure_shared_grads(model, shared_model)
            optimizer.step()
            action[3]= -0.01 
            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1

            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]
            state_inp = torch.from_numpy(env2.observation(obsDataNew)).type(FloatTensor)
            if timeStep >= env._max_episode_steps: break

        while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
            
            action = [0, 0, 0, 0, 0, 0]
            for i in range(len(goal - objectPos)):
                action[i] = (goal - objectPos)[i]*6

            action[3] = -0.01
            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1

            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]
            if timeStep >= env._max_episode_steps: break
        
        while True: #limit the number of timesteps in the episode to a fixed duration
            
            action = [0, 0, 0, 0, 0, 0]
            action[3] = -0.01 # keep the gripper closed

            obsDataNew, reward, done, info = env.step(action)
            timeStep += 1

            objectPos = obsDataNew['observation'][3:6]
            object_rel_pos = obsDataNew['observation'][6:9]

            if timeStep >= env._max_episode_steps: break

def test(rank, args, shared_model, counter):
    
    FloatTensor = torch.cuda.FloatTensor if args.use_cuda else torch.FloatTensor
    env = gym.make("FetchPickAndPlace-v1")
    env2 = gym.wrappers.FlattenDictWrapper(env, dict_keys=['observation', 'desired_goal'])

    model = Actor()
    if args.use_cuda:
        model.cuda()
    model.eval()
    done = True       
    success = 0

    savefile = os.getcwd() + '/train/mario_curves.csv'
    title = ['No. episodes', 'No. of success']
    with open(savefile, 'a', newline='') as sfile:
        writer = csv.writer(sfile)
        writer.writerow(title)   
    
    while True:
        model.load_state_dict(shared_model.state_dict())
        ep_num = 0
        num_ep = counter.value
        while ep_num < 100:
            ep_num +=1            
            lastObs = env.reset()
            goal = lastObs['desired_goal']
            objectPos = lastObs['observation'][3:6]
            object_rel_pos = lastObs['observation'][6:9]
            object_oriented_goal = object_rel_pos.copy()
            object_oriented_goal[2] += 0.03 # first make the gripper go slightly above the object    
            timeStep = 0
            #Ratio, first_step =[], []      
            while np.linalg.norm(object_oriented_goal) >= 0.01 and timeStep <= env._max_episode_steps:
                ##env.render()
                action = [0, 0, 0, 0, 0, 0]
                object_oriented_goal = object_rel_pos.copy()
                object_oriented_goal[2] += 0.03

                for i in range(len(object_oriented_goal)):
                    action[i] = object_oriented_goal[i]*6

                action[3] = 0.05 #open

                obsDataNew, reward, done, info = env.step(action)
                timeStep += 1

                objectPos = obsDataNew['observation'][3:6]
                object_rel_pos = obsDataNew['observation'][6:9]

            state_inp = torch.from_numpy(env2.observation(obsDataNew)).type(FloatTensor)
            #a=timeStep
            while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= env._max_episode_steps :
                ##env.render()
                action = [0, 0, 0, 0, 0, 0]
                
                act_tensor, ratio = act(state_inp, model, False, False)    
                ##Ratio.append(ratio.cpu().detach().numpy())
                for i in range(len(object_oriented_goal)):
                    action[i] = act_tensor[i].cpu().detach().numpy()
                    #action[i] = object_rel_pos[i]*6
                action[5] = act_tensor[3].cpu().detach().numpy()
                action[3]= -0.01 
                obsDataNew, reward, done, info = env.step(action)
                timeStep += 1

                objectPos = obsDataNew['observation'][3:6]
                object_rel_pos = obsDataNew['observation'][6:9]
                state_inp = torch.from_numpy(env2.observation(obsDataNew)).type(FloatTensor)
                if timeStep >= env._max_episode_steps: break
            
            while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= env._max_episode_steps :
                ##env.render()
                action = [0, 0, 0, 0, 0, 0]
                for i in range(len(goal - objectPos)):
                    action[i] = (goal - objectPos)[i]*6

                action[3] = -0.01
                obsDataNew, reward, done, info = env.step(action)
                timeStep += 1

                objectPos = obsDataNew['observation'][3:6]
                object_rel_pos = obsDataNew['observation'][6:9]
                if timeStep >= env._max_episode_steps: break
                    
            while True: #limit the number of timesteps in the episode to a fixed duration
                ##env.render()
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
                #plot_ratio = np.average(np.array(Ratio), 0)
                #lastObs = env.reset()
                if ep_num % 100==0:            
                    print("num episodes {}, success {}".format(num_ep, success))
                    data = [counter.value, success]
                    with open(savefile, 'a', newline='') as sfile:
                        writer = csv.writer(sfile)
                        writer.writerows([data])
                    time.sleep(25)

                
