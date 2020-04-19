#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:39:43 2020

@author: yusuf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.softmax(self.l2(x), dim=-1)
        return x 
    

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, 1)
        

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = self.l2(x)
        return x 
    
    
    
class VanillaAC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        args):
        
        self.args=args
        self.actor = Actor(state_dim, action_dim, args.hidden_size).to(device)
        self.critic = Critic(state_dim, args.hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.actor_lr)
        
    def select_action(self, state):
        dist = self.actor(torch.FloatTensor(state).to(device))
        m = Categorical(dist)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob.unsqueeze(0)
    
    def value_estimate(self, state):
        value = self.critic(torch.FloatTensor(state).to(device))
        return value
    
    
    def train(self, new_state,log_probs, rewards, gamma, dones,values):
        new_state = torch.FloatTensor(new_state).to(device)
        next_value = self.critic(new_state)
        log_probs = torch.cat(log_probs)
        returns = []
        for t in reversed(range(len(rewards))):
            next_value = rewards[t] + gamma * next_value * dones[t]
            returns.insert(0, next_value)
        
        returns = torch.cat(returns).detach()
        values = torch.cat(values)
        delta = returns - values
        actor_loss = -(log_probs * delta.detach()).mean()
        critic_loss = -(delta.detach()*values).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()       
        critic_loss.backward()
        self.critic_optimizer.step()
        
    def get_loss(self, new_state, log_probs, rewards, gamma, dones,values):
        with torch.no_grad():
            new_state = torch.FloatTensor(new_state).to(device)
            next_value = self.critic(new_state)
            log_probs = torch.cat(log_probs)
            returns = []
            for t in reversed(range(len(rewards))):
                next_value = rewards[t] + gamma * next_value * dones[t]
                returns.insert(0, next_value)
            
            returns = torch.cat(returns).detach()
            values = torch.cat(values)
            delta = returns - values
            actor_loss = -(log_probs * delta.detach()).mean()
            critic_loss = -(delta.detach()*values).mean()
        return actor_loss
    
    
    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        
