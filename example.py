#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:26:15 2018

@author: NewType
"""

import gym
import numpy as np

env = gym.make('SuperMarioBros-1-1-Tiles-v0')# remember need to make the environment each time
observation = env.reset()
action = [0, 0, 0 , 1, 0, 0]
print("START", end="")
for i in range(1000):
    env.render()

    observation, reward, done, info = env.step(action)
#env.reset()
    print("________observation________")
    #print(observation)
    marioPosY, marioPosX = np.where(observation == 3)
    if marioPosX.size != 0:
        #print("i: " , i ," mario location index: X==", marioPosX.item(0), " Y==", marioPosY.item(0))
        marioPosX = marioPosX.item(0)
        marioPosY = marioPosY.item(0)
        print("i: " , i ," mario location index: X==", marioPosX, " Y==", marioPosY)

        #lolol
    twoRight = observation[marioPosY, marioPosX + 2]
    print("twoRight : ", twoRight)
    
    if observation[marioPosY, marioPosX + 2] != 0:
        # [Up, L, Down, R, A(JUMP), B]
        action =[0, 0, 0 , 1, 1, 0]
        #env.step(action)
    else:
        action = [0, 0, 0, 1, 0, 0]
    
        
print("DONE", end="")
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