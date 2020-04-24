#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 20:49:19 2020

@author: yusuf
"""

from copy import deepcopy
import matplotlib.pyplot as plt
import argparse
import numpy as np
from matplotlib import cm
import actorcritic
import torch
import gym
from torch.distributions import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default='CartPole-v0')         
    parser.add_argument("--seed", default=0, type=int)              
    parser.add_argument("--hidden_size", default=128, type=int) 
    parser.add_argument("--max_episode_nums", default=int(2e3), type=int)   
    parser.add_argument("--max_steps", default=int(1e4), type=int)                
    parser.add_argument("--gamma", default=0.9)         # Discount factor
    parser.add_argument("--save_model", action="store_true")       
    parser.add_argument("--load_model", default="")
    parser.add_argument("--actor_act", default="relu") # Activation func of nn relu | sigmoid
    parser.add_argument("--critic_act", default="relu")
    parser.add_argument("--normalizedirs", action="store_false")              
    parser.add_argument('--folder_name', default='results', type=str)
    parser.add_argument('--actor_lr',default=3e-3, type=float)  
    parser.add_argument('--critic_lr',default=3e-3, type=float)  
    args = parser.parse_args()
    
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    file_name = '_seed_'+str(args.seed)+'_env_'+str(args.env_name)+'_'+str(args.max_episode_nums) +'_'+str(args.hidden_size) +'_'+str(args.actor_lr) +'.log'

    # Set seeds
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True
    
    policy = actorcritic.VanillaAC(state_dim, action_dim, args)
    policy.load(f"./models/{file_name}")

    def get_weights(net):
        """ Extract parameters from Q1 of critic"""
        return [p.data for p in net.parameters()]
    
    def normalize_direction(direction, weights, norm='filter'):
        """
            Rescale the direction so that it has similar norm as their corresponding
            model in different levels.
            Args:
              direction: a variables of the random direction for one layer
              weights: a variable of the original model for one layer
              norm: normalization method, 'filter' | 'layer' | 'weight'
        """
        if norm == 'filter':
            # Rescale the filters (weights in group) in 'direction' so that each
            # filter has the same norm as its corresponding filter in 'weights'.
            for d, w in zip(direction, weights):
                d.mul_(w.norm()/(d.norm() + 1e-10))
        elif norm == 'layer':
            # Rescale the layer variables in the direction so that each layer has
            # the same norm as the layer variables in weights.
            direction.mul_(weights.norm()/direction.norm())
        elif norm == 'weight':
            # Rescale the entries in the direction so that each entry has the same
            # scale as the corresponding weight.
            direction.mul_(weights)
        elif norm == 'dfilter':
            # Rescale the entries in the direction so that each filter direction
            # has the unit norm.
            for d in direction:
                d.div_(d.norm() + 1e-10)
        elif norm == 'dlayer':
            # Rescale the entries in the direction so that each layer direction has
            # the unit norm.
            direction.div_(direction.norm())


    def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
        """
            The normalization scales the direction entries according to the entries of weights.
        """
        assert(len(direction) == len(weights))
        for d, w in zip(direction, weights):
            if d.dim() <= 1:
                if ignore == 'biasbn':
                    d.fill_(0) # ignore directions for weights with 1 dimension
                else:
                    d.copy_(w) # keep directions for weights/bias that are only 1 per node
            else:
                normalize_direction(d, w, norm)
    
    
    def perturb_wights(actor_net, N, param_range, r):
        origin_actor = deepcopy(actor_net)
        q1_weights = get_weights(actor_net)
        z = np.zeros((r, r))
        alp = []
        bet = []
        for i in q1_weights:
            alp.append(torch.randn(i.size()))
            bet.append(torch.randn(i.size()))
        if args.normalizedirs:
            normalize_directions_for_weights(alp, q1_weights, 'weight')
            normalize_directions_for_weights(bet, q1_weights, 'weight')    
        for a_i,a in enumerate(np.linspace(-param_range,param_range,r)):
            for b_i,b in enumerate(np.linspace(-param_range,param_range,r)):
                actor_net = deepcopy(origin_actor)
                for j, p in enumerate(actor_net.parameters()):           
                    p.data = p + a*alp[j] + b*bet[j] 
                loss = 0
                for episode in range(N):
                    state = env.reset()
                    rewards = []
                    log_probs = []
                    values = []
                    dones = []
                    for steps in range(int(1e6)):
                        dist = actor_net(torch.FloatTensor(state).to(device))
                        m = Categorical(dist)
                        action = m.sample()
                        log_prob = m.log_prob(action)
                        value = policy.value_estimate(state)
                        new_state, reward, done, _ = env.step(action.item()) 
                        
                        rewards.append(torch.tensor([reward]))
                        log_probs.append(log_prob.unsqueeze(0))
                        values.append(value)
                        dones.append(torch.tensor([1-done], dtype=torch.float, device=device))
                        
                        state = new_state
        
                        if done:
                          loss += policy.get_loss(new_state,log_probs,rewards,args.gamma,dones,values)
                          break
    
                loss /= N # averaging over episodes
                z[a_i, b_i] = loss
        return z
    
    def plot_surface(net1,N, param_range, r):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        # Make data.
        X = np.linspace(-param_range,param_range,r)
        Y = np.linspace(-param_range,param_range,r)
        X, Y = np.meshgrid(X, Y)
        Z = perturb_wights(net1, N, param_range, r)
        
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.xlabel('$\\alpha$')
        plt.ylabel('$\\beta$')
        ax.set_zlabel('Actor loss')
        plt.savefig(f'./plots/{file_name}surface.png')
        
    def plot_contour(net1, N, param_range, r):
        fig, ax = plt.subplots()
        X = np.linspace(-param_range,param_range,r)
        Y = np.linspace(-param_range,param_range,r)
        X, Y = np.meshgrid(X, Y)
        Z = perturb_wights(net1, N, param_range, r)
        
        CS = ax.contour(X, Y, Z, 6,
                         colors='k',  # negative contours will be dashed by default
                         )
        ax.clabel(CS, fontsize=9, inline=1)
        # ax.set_title('Single color - negative contours solid')
        
        plt.savefig(f'./plots/{file_name}contour.png')
    actorcopy = deepcopy(policy.actor)   
    plot_surface(policy.actor,50, 1, 20)
    plot_contour(actorcopy,50, 1, 20)
    