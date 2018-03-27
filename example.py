#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:26:15 2018

@author: Jonathan Scott, Jinyoung Lim, So Jin Oh
"""

import gym
import pickle_utilities as pu
from ql_box import *
import sys
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

#note at 670, decrease max ep to 5000 after reading paper.
#After 730, I changed the reward function to penality of 0.03 from 0.01 
#After 995, changed epsilon decrease conditions
"""params: [1]num of batches [2]num of episodes [3]box size"""
if __name__ == "__main__":
    env = gym.make('SuperMarioBros-1-1-Tiles-v0')  # remember need to make the environment each time
    test_algorithm(env)
    if len(sys.argv) == 4:
        numBatches = int(sys.argv[1])
        numEpisodes = int(sys.argv[2])
        boxSizeEntered = int(sys.argv[3]) 
    else:
        numBatches = 2
        numEpisodes = 5
        boxSizeEntered = 2
    print('number of batches: ' + str(numBatches) + ', number of episodes: ' +
          str(numEpisodes) + ', box size: ' + str(boxSizeEntered))
    for i in range(numBatches):
        print(str(i*numEpisodes) + ' episodes have been run.')
        Q = ql_box(env, numEpisodes, boxSize=boxSizeEntered)


    
    