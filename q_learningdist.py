#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:42:50 2018

@author: NewType
"""
import pickle_utilities as pu
import numpy as np
import itertools
import random


"""Q-learning with distance and score.
"""
#workerd on this instead of ql_box because accounting for a boxed (limited) environment means that the policy would not be a reinforced-learning?
def ql_distScore(env, num_episodes, alpha=0.85, discount_factor=0.99):
    # decaying epsilon, i.e we will divide num of episodes passed
    epsilon = 1.0

    # call setdefault for a new state.
    if hasPickleWith("ql_distScore"):
        Q,last_episode = pu.loadLatestWith("ql_distScore")
    else:
        Q = {0: {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'B': 0}}
    action = [0, 0, 0, 0, 0, 0]  # Do nothing
    action_dict = {'up':    [1, 0, 0, 0, 0, 0],
                   'L':     [0, 1, 0, 0, 0, 0],
                   'down':  [0, 0, 1, 0, 0, 0],
                   'R':     [0, 0, 0, 1, 0, 0],
                   'JUMP':  [0, 0, 0, 0, 1, 0],
                   'B':     [0, 0, 0, 0, 0, 1]}

    for episode in range(num_episodes):
        observation = env.reset()
        observation, reward, done, info = env.step(action)
        state = info['distance'] + info['score'] # adds the distance and score...
        #TODO: there must be a fancier way to compute the total state

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
            next_state = info['distance'] + info['score']
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
    pu.saveQ(Q, num_episodes, functionName='ql_distScore')
    return Q  # return optimal Q