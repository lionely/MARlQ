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

TOTAL_DIST = 3266

"""Q-learning with box as a state. The box size will be 2 blocks away from
Mario as a default.
"""
#worked on this instead of ql_box because accounting for a boxed (limited)
#environment means that the policy would not be a reinforced-learning?
def ql_box(env, num_episodes, learning_rate=0.85, discount_factor=0.99, boxSize=2):
    """(http://178.79.149.207/posts/cartpole-qlearning.html)
    alpha       learning rate
    epsilon     exploration rate
    gamma       discount factor
    """

    # decaying epsilon, i.e we will divide num of episodes passed
    #epsilon = 1.0
    #TODO: uncomment to update Q
    # last_episodeQ = 0
    last_episodeASC = 0    # This is so we can run episodes in batches because
                            # running many at once takes a lot of time!
    funcName = "ql_box_size" + str(boxSize)

    # call setdefault for a new state.
    if pu.hasPickleWith("ql_box", boxSize, "ASC-tables/*.pickle"):
        action_state_count, last_episodeASC = pu.loadLatestASCWith("ql_box", boxSize)

    # TODO: uncomment to update Q
    # if pu.hasPickleWith("ql_box", boxSize, 'Q-tables/*.pickle'):
    #     Q, last_episodeQ = pu.loadLatestQWith("ql_box", boxSize)
    # else:
    #     box = getDefaultBox(boxSize)
    #     # not sure if "0000000000003000000000000" is a correct initial box (state) that is comparable to 0
    #     Q = {box: {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'R_JUMP1': 0, 'R_JUMP2': 0, 'R_JUMP3': 0}}
    #     action_state_count = {box: {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'R_JUMP1': 0, 'R_JUMP2': 0, 'R_JUMP3': 0}}

    box = getDefaultBox(boxSize)
    # not sure if "0000000000003000000000000" is a correct initial box (state) that is comparable to 0
    Q = {box: {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'R_JUMP1': 0, 'R_JUMP2': 0, 'R_JUMP3': 0}}
    action_state_count = {box: {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'R_JUMP1': 0, 'R_JUMP2': 0, 'R_JUMP3': 0}}

    #TODO: uncomment to update Q
    # last_episode = last_episodeQ if (last_episodeQ <= last_episodeASC) else last_episodeASC  # take the smaller episode
    last_episode = last_episodeASC


    action = [0, 0, 0, 0, 0, 0]  # Do nothing
    # remove 'B', which is irrelevant for clearing level 1
    # add R_JUMP, which makes Mario jump and also go right at the same time
    #   L add three times to make Mario prefer R_JUMP more than other actions
    action_dict = {'up':        [1, 0, 0, 0, 0, 0],
                   'L':         [0, 1, 0, 0, 0, 0],
                   'down':      [0, 0, 1, 0, 0, 0],
                   'R':         [0, 0, 0, 1, 0, 0],
                   'JUMP':      [0, 0, 0, 0, 1, 0],
                   'R_JUMP1':   [0, 0, 0, 1, 1, 0],
                   'R_JUMP2':   [0, 0, 0, 1, 1, 0],
                   'R_JUMP3':   [0, 0, 0, 1, 1, 0]}


    for episode in range(num_episodes):
        ### Epsilon Policy ###
        epsilon = 0.3

        # Reset environment each episode
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

        Q.setdefault(state, {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'R_JUMP1': 0, 'R_JUMP2': 0, 'R_JUMP3': 0})
        action_state_count.setdefault(state, {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'R_JUMP1': 0, 'R_JUMP2': 0, 'R_JUMP3': 0})

        for t in itertools.count():
            if np.random.random() <= epsilon:
                # select a random action from set of all actions
                # max_q_action = random.choice(Q[state].keys())      # PYTHON2
                max_q_action = random.choice(list(Q[state].keys()))  # PYTHON3
                action = action_dict[str(max_q_action)]

                # update count for the current state and action NOT sure if this should go after calculating alpha
                action_state_count[state][str(max_q_action)] += 1
                # if the generated num is greater than epsilon, we follow exploitation policy
            else:
                # select an action with highest value for current state
                max_q_action = max(Q[state], key=(lambda key: Q[state][key]))

                # not fully sure about lambdas >.< https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
                action = action_dict[str(max_q_action)]

                # update count for the current state and action NOT sure if this should go after calculating alpha
                action_state_count[state][str(max_q_action)] += 1

                # apply selected action, collect values for next_state and reward
            observation, reward, done, info = env.step(action)

            #print("Qbox reward is: "+str(reward))
            next_state = getBox(observation, boxSize)

            Q.setdefault(next_state, {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'R_JUMP1': 0, 'R_JUMP2': 0, 'R_JUMP3': 0})
            action_state_count.setdefault(next_state, {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'R_JUMP1': 0, 'R_JUMP2': 0, 'R_JUMP3': 0})



            # TODO: uncomment to update Q
            # max_next_state_action = max(Q[next_state], key=lambda key: Q[next_state][key])
            # # Calculate the Q-learning target value
            # Q_target = reward + discount_factor * Q[next_state][max_next_state_action]
            # # Calculate the difference/error between target and current Q
            # Q_delta = Q_target - Q[state][str(max_q_action)]
            # # Calculate alpha
            # alpha = learning_rate / action_state_count[state][max_q_action]
            # # Update the Q table, alpha is the learning rate
            # Q[state][str(max_q_action)] = Q[state][str(max_q_action)] + (alpha * Q_delta)


            # break if done, i.e. if end of this episode
            if done:
                break
            # make the next_state into current state as we go for next iteration
            state = next_state

        ep_dist,ep_reward = info['distance'],info['total_reward'] #last recorded distance , last recorded reward from episodes
        # TODO: uncomment to update Q
        # pu.saveQAndASC(Q, action_state_count, episode + last_episode + 1, functionName='ql_box',boxSize=boxSize)
        pu.saveASC(action_state_count, episode + last_episode + 1, functionName='ql_box', boxSize=boxSize)
        pu.collectData(episode + last_episode +1, ep_reward, ep_dist, functionName=funcName)

    env.close()
    return Q, action_state_count  # return optimal Q table and action_state_count table

"""Returns a defalut box according to the size of the box."""
def getDefaultBox(boxSize):
    strLen = boxSize * boxSize
    boxToStr = "0" * strLen
    list(boxToStr)[int(strLen/2)] = "3"
    #print("boxToStr: " + boxToStr)
    return str(boxToStr)

"""Returns information of a box surrounding the Mario in str type. Used for ql_box."""
def getBox(observation, boxSize):
    marioPosY, marioPosX = np.where(observation == 3)

    # handle edge case where mario's positions are not given
    if marioPosY.size == 0 or marioPosX.size == 0:
        """ boxSize == 2
        0 0 0 0 0
        0 0 0 0 0
        0 0 3 0 0 --> 5x5 with Mario (3) at boxToStr[12] or boxToStr[int(5*5/2)] of len(boxToStr) == 25
        0 0 0 0 0
        0 0 0 0 0
        """
        """ boxSize == 3
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
        0 0 0 3 0 0 0
        0 0 0 0 0 0 0 --> 7x7 with Mario (3) at boxToStr[24] or boxToStr[int(7*7/2)] of len(boxToStr) == 7*7
        0 0 0 0 0 0 0
        0 0 0 0 0 0 0
        """
        return getDefaultBox(boxSize)


    marioPosX = marioPosX.item(0)
    marioPosY = marioPosY.item(0)

    box = ""
    for i in range(-boxSize,boxSize+1):
        for j in range(-boxSize, boxSize+1):
            if (((marioPosY+i) < 0 or (marioPosY+i) >= observation.shape[0]) or
                ((marioPosX+j) < 0 or (marioPosX+j) >= observation.shape[1])):
                currBoxPos = "0"
            else:
                currBoxPos = observation[marioPosY+i, marioPosX+j]

            box += str(currBoxPos)

    return box

"""This function takes an environment and Q table and checks if the optimal actions
at each state is actually being taken. """
def test_algorithm(env, boxSize=3, Q=None):
    if not Q:
        Q = pu.loadLatestQWith('ql_box', boxSize)[0]
    observation = env.reset()
    total_reward = 0
    action = [0]*6
    observation, reward, done, info = env.step(action)
    marioPosY, marioPosX = np.where(observation == 3)
    while marioPosX.size == 0:
            action = [0, 0, 0, 1, 0, 0]
            observation, reward, done, info = env.step(action)
            marioPosY, marioPosX = np.where(observation == 3)

    state = getBox(observation, boxSize)

    action_dict = {'up': [1, 0, 0, 0, 0, 0],
                   'L': [0, 1, 0, 0, 0, 0],
                   'down': [0, 0, 1, 0, 0, 0],
                   'R': [0, 0, 0, 1, 0, 0],
                   'JUMP': [0, 0, 0, 0, 1, 0],
                   'R_JUMP1': [0, 0, 0, 1, 1, 0],
                   'R_JUMP2': [0, 0, 0, 1, 1, 0],
                   'R_JUMP3': [0, 0, 0, 1, 1, 0]}
    for t in itertools.count():
        if np.random.random() <= 0.01:
            max_q_action = random.choice(list(Q[state].keys()))
            # max_q_action = 'R_JUMP1'
            print("***RANDOM***", end=" ")

        else:
            max_q_action = max(Q[state], key=(lambda key: Q[state][key]))
            print("***MAX  Q***", end=" ")

        # if state is not found in the Q table due to the changing environment each episode, make mario R_JUMP
        if state not in Q:
            # max_q_action = 'R_JUMP1'
            max_q_action = random.choice(list(Q[state].keys()))
            print("***FORCED***", end=" ")

        action = action_dict[max_q_action]
        time = info['time']
        dist = info['distance']
        print("TIME: ", time, "   DISTANCE: ", dist, "   ACTION: ", max_q_action, "   REWARD: ", total_reward, "(", reward, ")")

        observation, reward, done,info = env.step(action)
        next_state = getBox(observation, boxSize)
        total_reward += reward

        if done:
            print(total_reward)
            break
        state = next_state
    env.close()
    return total_reward


"""Possibly a helper function to test_algorithm"""
def isStuck(stuck, capacity):
    if len(stuck) == capacity:
        stuck = []
    return len(np.unique(stuck)) == 1
