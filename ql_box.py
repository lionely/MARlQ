#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:44:59 2018

@author: NewType
"""

import pickle_utilities as pu
import numpy as np
import itertools
import random

"""Q-learning with box as a state. The box size will be 2 blocks away from Mario as a default.
"""
#worked on this instead of ql_box because accounting for a boxed (limited) environment means that the policy would not be a reinforced-learning?
def ql_box(env, num_episodes, alpha=0.85, discount_factor=0.99, boxSize=2):
    # decaying epsilon, i.e we will divide num of episodes passed
    epsilon = 1.0
    last_episode = 0 #This is so we can run episodes in batches because running many at once takes a lot of time!
    
    # call setdefault for a new state.
    if pu.hasPickleWith("ql_box"):
        Q,last_episode = pu.loadLatestWith("ql_box")
        
    else:
        # not sure if "0000000000003000000000000" is a correct initial box (state) that is comparable to 0
        Q = {"0000000000003000000000000": {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'B': 0}}
        # the state "0000000000003000000000000" represents the box that looks as below
        # 00000
        # 00000
        # 00300
        # 11111
        # 11111
    action = [0, 0, 0, 0, 0, 0]  # Do nothing
    action_dict = {'up':    [1, 0, 0, 0, 0, 0],
                   'L':     [0, 1, 0, 0, 0, 0],
                   'down':  [0, 0, 1, 0, 0, 0],
                   'R':     [0, 0, 0, 1, 0, 0],
                   'JUMP':  [0, 0, 0, 0, 1, 0],
                   'B':     [0, 0, 0, 0, 0, 1]}


    for episode in range(num_episodes):
        print("Starting episode: ",episode)
        observation = env.reset()
        observation, reward, done, info = env.step(action)

        marioPosY, marioPosX = np.where(observation == 3)   #mario position

        # in the beginning of the game, when mario's position is not set (that is we cannot get
        # mario's x and y positions using observation), mario moves right
        # TODO: if there is a more elegant way to deal with the beginning of the game (edge case)... go for it!
        while marioPosX.size == 0:
            action = [0, 0, 0, 1, 0, 0]
            observation, reward, done, info = env.step(action)
            marioPosY, marioPosX = np.where(observation == 3)


        state = getBox(observation, boxSize)

        Q.setdefault(state, {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'B': 0})

        for t in itertools.count():
            # generate a random num between 0 and 1 e.g. 0.35, 0.73 etc..
            # if the generated num is smaller than epsilon, we follow exploration policy
            if np.random.random() <= epsilon:
                # select a random action from set of all actions
                # max_q_action = random.choice(Q[state].keys())      # PYTHON2
                max_q_action = random.choice(list(Q[state].keys()))  # PYTHON3


                action = action_dict[str(max_q_action)]
            # if the generated num is greater than epsilon, we follow exploitation policy
            else:
                # select an action with highest value for current state
                max_q_action = max(Q[state], key=(lambda key: Q[state][key]))
                # not fully sure about lambdas >.< https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary

                action = action_dict[str(max_q_action)]
            # apply selected action, collect values for next_state and reward


            observation, reward, done, info = env.step(action)
            next_state = getBox(observation, boxSize)
            Q.setdefault(next_state, {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'B': 0})
            max_next_state_action = max(Q[next_state], key=lambda key: Q[next_state][key])
            # Calculate the Q-learning target value
            Q_target = reward + discount_factor * Q[next_state][max_next_state_action]
            # Calculate the difference/error between target and current Q
            Q_delta = Q_target - Q[state][str(max_q_action)]
            # Update the Q table, alpha is the learning rate
            Q[state][str(max_q_action)] = Q[state][str(max_q_action)] + (alpha * Q_delta)

            # break if done, i.e. if end of this episode
            if done:
                break
            # make the next_state into current state as we go for next iteration
            state = next_state
        # gradualy decay the epsilon
        if epsilon > 0.1:
            epsilon -= 1.0 / num_episodes

    #TODO: ql_box's len(Q) != maximum distance (don't know what it represents) figure out a way to have consistancy between file names.
    pu.saveQ(Q, num_episodes + last_episode, functionName='ql_box',boxSize=boxSize)
    """
    saved: ql_box_51_5.pickle
        Reloaded modules: wrappers, wrappers.action_space, wrappers.control
        [2018-03-16 14:04:32,550] Making new env: SuperMarioBros-1-1-Tiles-v0

    saved: ql_box_51_1.pickle
        Reloaded modules: wrappers, wrappers.action_space, wrappers.control
        [2018-03-16 14:31:44,025] Making new env: SuperMarioBros-1-1-Tiles-v0
        Loaded: ql_box_51_5.pickle

    saved: ql_box_56_4.pickle
        Reloaded modules: wrappers, wrappers.action_space, wrappers.control
        [2018-03-16 14:48:35,261] Making new env: SuperMarioBros-1-1-Tiles-v0
        Loaded: ql_box_51_1.pickle

    saved: ql_box_75_10.pickle
        Reloaded modules: wrappers, wrappers.action_space, wrappers.control
        [2018-03-16 15:03:24,828] Making new env: SuperMarioBros-1-1-Tiles-v0
        Loaded: ql_box_56_4.pickle
    """
    env.close()
    return Q  # return optimal Q

"""Returns information of a box surrounding the Mario in str type. Used for ql_box."""
def getBox(observation, boxSize):
    marioPosY, marioPosX = np.where(observation == 3)

    # handle edge case where mario's positions are not given
    if marioPosY.size == 0 or marioPosX.size == 0:
        return "0000000000003000000000000"


    marioPosX = marioPosX.item(0)
    marioPosY = marioPosY.item(0)

    box = ""
    for i in range(-boxSize,boxSize+1):
        for j in range(-boxSize, boxSize+1):
            currBoxPos = observation[marioPosY+i, marioPosX+j]

#                currBoxPos = currBoxPos.item(0)
            box += str(currBoxPos)
#    print(box)
    return box