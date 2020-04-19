#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:35:18 2020

@author: yusuf
"""


import argparse
import actorcritic
import gym
import numpy as np
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default='CartPole-v0')         
    parser.add_argument("--seed", default=0, type=int)              
    parser.add_argument("--hidden_size", default=128, type=int) 
    parser.add_argument("--max_episode_nums", default=int(2e3), type=int)   
    parser.add_argument("--max_steps", default=int(1e4), type=int)              
    parser.add_argument("--batch_size", default=256, type=int)      
    parser.add_argument("--gamma", default=0.9)         # Discount factor
    parser.add_argument("--save_model", action="store_true")       
    parser.add_argument("--load_model", default="")
    parser.add_argument("--noplot", action="store_false")   # Cancel plotting log              
    parser.add_argument('--folder_name', default='results', type=str)
    parser.add_argument('--actor_lr',default=3e-3, type=float)  
    parser.add_argument('--critic_lr',default=3e-3, type=float)  
    args = parser.parse_args()
    
    def eval_policy(policy, env_name=args.env_name, seed=args.seed, eval_episodes=10):
        eval_env = gym.make(env_name)
        avg_reward = 0.
        for number in range(eval_episodes):
            
            state, done = eval_env.reset(), False
          
            
            while not done:
                action, _ = policy.select_action(state)
                
                next_state, reward, done, _ = eval_env.step(action)
             
    #            if next_state[0] >= 0.5:
    #                    reward+=10
    #            if next_state[0] > -0.4:
    #                    reward+= (1+state[0])**2
                
                state=next_state
                
                avg_reward += reward
    
        avg_reward /= eval_episodes
    
        print("---------------------------------------")
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        return avg_reward
    
    
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    file_name = '_seed_'+str(args.seed)+'_env_'+str(args.env_name)+'_'+str(args.max_episode_nums) +'.log'

    # Set seeds
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True
    

    all_rewards = []

    
    policy = actorcritic.VanillaAC(state_dim, action_dim, args)
    
    for episode in range(args.max_episode_nums):
        state = env.reset()
        rewards = []
        log_probs = []
        values = []
        dones = []
        for steps in range(args.max_steps):
            action, log_prob = policy.select_action(state)
            value = policy.value_estimate(state)
            new_state, reward, done, _ = env.step(action) 
            
            rewards.append(torch.tensor([reward]))
            log_probs.append(log_prob)
            values.append(value)
            dones.append(torch.tensor([1-done], dtype=torch.float, device=device))
                
            state = new_state

            if done:
                policy.train(new_state,log_probs,rewards,args.gamma,dones,values)
                all_rewards.append(np.sum(rewards))
                if episode%10 == 0:
                    print("episode: {}, reward: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3)))
                    av_rew = eval_policy(policy)
                break

    if args.save_model: 
        if not os.path.exists("./models"):
            os.makedirs("./models")
        policy.save(f"./models/{file_name}")
    if args.noplot and not os.path.exists("./plots"):
        os.makedirs("./plots")
        np.save("./plots/"+file_name+ "plot.npy",np.array(all_rewards))
