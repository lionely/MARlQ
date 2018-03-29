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
#import wrappers 

#TODO Figure out how to stop mario from getting stuck.
def playAsHuman(env, playTime=1000):
    #TODO: make this work...!
    #TODO: make this run based on time and kill it after the time is over.
    wrapper = wrappers.SetPlayingMode('human')
    env = wrapper(env)
    env.render()
    #env.close()  # closes game

#Should hold down jump, to be able to jump higher."
    

   
#TODO collect total reward after every 5 episodes, max distance, episodes ran so far

#params: [1]num of batches [2]num of episodes [3]box size
if __name__ == "__main__":
    env = gym.make('SuperMarioBros-1-1-Tiles-v0')  # remember need to make the environment each time
    #Q = pu.loadQ('ql_box_245_2.pickle')
    #test_algorithm(env)

    #test_algorithm(env, boxSize=3)


       
    #Manually start
    for i in range(1):
        numEp = 20
        print(str(i*numEp) + ' episodes have been run.')
        Q = ql_box(env, numEp, boxSize=3)
    

    """
    #Run using terminal
    numBatches = int(sys.argv[1])
    numEpisodes = int(sys.argv[2])
    boxSizeEntered = int(sys.argv[3])
    
    print('number of batches: ' + str(numBatches) + ', number of episodes: ' +
          str(numEpisodes) + ', box size: ' + str(boxSizeEntered))
    for i in range(numBatches):
        print(str(i*numEpisodes) + ' episodes have been run.')
        Q = ql_box(env, numEpisodes, boxSize=boxSizeEntered)
    """

    
    