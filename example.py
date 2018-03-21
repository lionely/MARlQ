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
if __name__ == "__main__":
    env = gym.make('SuperMarioBros-1-1-Tiles-v0')  # remember need to make the environment each time
    #Q = pu.loadQ('ql_box_245_2.pickle')
    #test_algorithm(env)
    for i in range(2):
        print(str(i*5) + ' episodes have been run.')
        Q = ql_box(env, 5)
    