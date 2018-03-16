#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:08:05 2018

@author: Jonathan Scott, Jinyoung Lim, So Jin Oh
"""

import gym
import numpy as np
import pickle
import glob
import sys
import os
import itertools
import random
import wrappers

def bruteForcePolicy(env):
    observation = env.reset()
    action = [0, 0, 0 , 1, 0, 0]
    print("START")
    for i in range(100):
        env.render()

        observation, reward, done, info = env.step(action)
        print("Info has the distance: ",type(info['distance']))
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
    print("DONE")
    env.close()#closes game

if __name__ == "__main__":
    env = gym.make('SuperMarioBros-1-1-Tiles-v0')  # remember need to make the environment each time
    bruteForcePolicy(env)

    #loaded_Q2 = loadLatest()
    #loaded_Q = loadQ('q_248_10.pickle')
    #assert(loaded_Q==Q)