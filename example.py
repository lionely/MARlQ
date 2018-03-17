#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:26:15 2018

@author: Jonathan Scott, Jinyoung Lim, So Jin Oh
"""

import gym
import numpy 
import pickle_utilities
from ql_box import *
import itertools
import random
import wrappers

def playAsHuman(env, playTime=1000):
    #TODO: make this work...!
    #TODO: make this run based on time and kill it after the time is over.
    wrapper = wrappers.SetPlayingMode('human')
    env = wrapper(env)
    env.render()
    #env.close()  # closes game

#Should hold down jump, to be able to jump higher."

#TODO: Added docustring but this function is not complete yet, will do after we clear level 1.
"""This function takes an environment and Q table and checks if the optimal actions
at each state is actually being taken. """
def test_algorithm(env,Q):
    stuck_capacity = 5
    stuck = []

    observation = env.reset()
    total_reward = 0
    action = [0]*6
    observation,reward,done,info = env.step(action)
    state = info['distance']
    action_dict = {'up':    [1, 0, 0 ,0, 0, 0],
                   'L':     [0, 1, 0, 0, 0, 0],
                   'down':  [0, 0, 1, 0, 0, 0],
                   'R':     [0, 0, 0, 1, 0, 0],
                   'JUMP':  [0, 0, 0, 0, 1, 0],
                   'B':     [0, 0, 0, 0, 0, 1]}
    for t in itertools.count():
        # selection the action with highest values i.e. best action
        max_q_action = max(Q[state], key=lambda key: Q[state][key])
#        if len(stuck)<stuck_capacity:
#            stuck.append(max_q_action)
#        stuck = [] if t%10==0 else stuck
        #print('t is ',t )
#        if isStuck(stuck,stuck_capacity):
#            print('is stuck :(')
#            max_q_action = 'R' #random.choice(Q[state].keys())
#            stuck = []
        print("Optimal action is: " + max_q_action)
        action = action_dict[str(max_q_action)]
        print("Action is: " , action)
        # apply selected action
        observation, reward, done,info = env.step(action)
        print("Reward: ", reward)
        next_state = info['distance']
        # calculate total reward
        total_reward += reward
        if done:
            print(total_reward)
            break
        state = next_state
    return total_reward
"""Possibly a helper function to test_algorithm"""
def isStuck(stuck,capacity):
    if len(stuck) == capacity:
        stuck = []
    return len(np.unique(stuck)) == 1

if __name__ == "__main__":
    env = gym.make('SuperMarioBros-1-1-Tiles-v0')  # remember need to make the environment each time
    Q = ql_box(env, 1)
    box_Q = loadLatestWith('ql_box')
    
    #Q = ql_distScore(env, 10)

    
