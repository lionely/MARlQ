#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:26:15 2018

@author: Jonathan Scott, Jinyoung Lim, So Jin Oh
"""

import gym
#import numpy 
import pickle_utilities as pu
from ql_box import *
import sys
#import itertools
#import random
import wrappers

#TODO Figure out how to stop mario from getting stuck.
def playAsHuman(env, playTime=1000):
    #TODO: make this run based on time and kill it after the time is over.
    cmd = input()
    if cmd in 'hH':
        wrapper = wrappers.SetPlayingMode('human')
        env = wrapper(env)
    elif cmd in 'aA':
        wrapper = wrappers.SetPlayingMode('algo')
        env = wrapper(env)

   
#TODO collect total reward after every 5 episodes, max distance, episodes ran so far

#params: [1]num of batches [2]num of episodes [3]box size
if __name__ == "__main__":
    env = gym.make('SuperMarioBros-1-1-Tiles-v0')  # remember need to make the environment each time
    # playAsHuman(env)

       
    # #Manually start
    # for i in range(1):
    #     numEp = 1
    #     print(str(i*numEp) + ' episodes have been run.')
    #     Q = ql_box(env, numEp, boxSize=3)



    #Run using terminal
    if len(sys.argv) == 4:
        numBatches = int(sys.argv[1])
        numEpisodes = int(sys.argv[2])
        boxSizeEntered = int(sys.argv[3])
    else:
        numBatches = 1
        numEpisodes = 5
        boxSizeEntered = 3
    print('number of batches: ' + str(numBatches) + ', number of episodes: ' +
          str(numEpisodes) + ', box size: ' + str(boxSizeEntered))

    for i in range(numBatches):
        print(str(i*numEpisodes) + ' episodes have been run.')
        Q = ql_box(env, numEpisodes, boxSize=boxSizeEntered)


    
    