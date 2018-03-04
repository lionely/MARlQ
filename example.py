#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:26:15 2018

@author: NewType
"""

import gym 

env = gym.make('SuperMarioBros-1-1-Tiles-v0')# remember need to make the environment each time
observation = env.reset()
for _ in range(1):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
#env.reset()
    print(observation)
env.close()#closes game


#env = gym.make('SuperMarioBros-1-1-Tiles-v0')
#observation = env.reset()
#done = False
#t = 0
#while not done:
#    action = env.action_space.sample()  # choose random action
#    observation, reward, done, info = env.step(action)  # feedback from environment
#    t += 1
#    if not t % 100:
#        print(t, info)