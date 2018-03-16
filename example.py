#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:26:15 2018

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



#TODO Is there a better way to search for extensions with pickle?
def hasPickleWith(functionName):
    database = filter(os.path.isfile, glob.glob('./*.pickle'))
    if database:
        for file in database:
            if functionName in file:
                return True
    return False

#File name based on furthest distance, nb_episodes (q_furthestDistance_numEpisode.pickle)
#https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
def saveQ(Q, num_episodes, functionName,boxSize=""):
    # TODO: make num_episodes consider the previous number of episodes as well.
    # For example, if initially done 10 episodes, num_episodes==10.
    # If do 20 more episodes, num_episodes==30.
    if functionName == 'q_learning':
        with open(functionName + '_' +str(num_episodes)+'.pickle', 'wb') as handle:
            pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(functionName + '_'+str(num_episodes)+'_'+str(boxSize)+'.pickle', 'wb') as handle:
            pickle.dump(Q, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved Q table succesfully for "+str(num_episodes)+" episodes!")
    return
def loadQ(filename):
    with open(filename, 'rb') as handle:
        unserialized_data = pickle.load(handle)
    return unserialized_data
#TODO if there is a better please change it :D
#https://stackoverflow.com/questions/9492481/check-that-a-type-of-file-exists-in-python
#returns max ep num as well so we can pick up where w eleft off.
def loadLatestWith(functionName):
    episode_stamps = []
    f_name = ''
    # for file in _glob.glob("*.pickle"):   #PYTHON2
    for file in glob.glob("*.pickle"):      #PYTHON3    https://stackoverflow.com/questions/44366614/nameerror-name-glob-is-not-defined
        if functionName in file and functionName=='q_learning':
            f_name = str(file)
            #Most recent is defined as the one that makes it the furtherest distance.
            f_name_offset = len(functionName)-1 # to get the index of where it starts
            e_stamp = f_name[f_name_offset+2:]
            end_ = e_stamp.index('_')
            episode_stamps.append(e_stamp[:end_])
        elif functionName in file:
            f_name = str(file)
            box_num = f_name[-1]
            #Most recent is defined as the one that makes it the furtherest distance.
            f_name_offset = len(functionName)-1 # to get the index of where it starts
            e_stamp = f_name[f_name_offset+2:]
            end_ = e_stamp.index('_')
            episode_stamps.append(e_stamp[:end_])

    max_e_stamp = max(episode_stamps)

    #Why are we looping through twice??
    #I looped to find the latest pickle then after we found That
    # we load. There might be a way to load based on the most recent episode. We can look into it!
    for file in glob.glob("*.pickle"):
        if functionName in file and functionName=='q_learning':
            f_name = str(file)
            if max_e_stamp in f_name:
                break
        elif functionName in file and box_num in file:
            f_name = str(file)
            if max_e_stamp in f_name:
                break
    print("Loaded: " + f_name)
    return  (loadQ(f_name) , int(max_e_stamp) )

def playAsHuman(env, playTime=1000):
    #TODO: make this work...!
    #TODO: make this run based on time and kill it after the time is over.
    wrapper = wrappers.SetPlayingMode('human')
    env = wrapper(env)
    env.render()
    #env.close()  # closes game

#Should hold down jump, to be able to jump higher."
"""alpha is the learning rate, gamma the discount factor, closer value in range [0,1] closer to 1 means it considers
future rewards.
Key for state is distance. Value is a dict with possbile actions, initilzaed to probability of 0 at first.
Q = {'x': {'up':0, 'L':0, 'down':0,'R':0,'JUMP':0,'B':0 }} where x is an integer measuring Mario's distance from the goal.
These q values will be updated based on the q function. """
def q_learning(env, num_episodes, alpha=0.85, discount_factor=0.99):
    # decaying epsilon, i.e we will divide num of episodes passed
    epsilon = 1.0
    standing_penalty = 0.08
    #call setdefault for a new state.
    if hasPickleWith("q_learning"):
        Q = loadLatestWith("q_learning")[0]

    else:
        Q = {0: {'up':0, 'L':0, 'down':0,'R':0,'JUMP':0,'B':0 }}
    action = [0, 0, 0, 0, 0, 0] #Do nothing
    action_dict = {'up':    [1, 0, 0, 0, 0, 0],
                   'L':     [0, 1, 0, 0, 0, 0],
                   'down':  [0, 0, 1, 0, 0, 0],
                   'R':     [0, 0, 0, 1, 0, 0],
                   'JUMP':  [0, 0, 0, 0, 1, 0],
                   'B':     [0, 0, 0, 0, 0, 1]}

    for episode in range(num_episodes):
        print("Starting episode: ",episode)
        observation = env.reset()
        observation,reward,done,info = env.step(action)
        """
        The following variables are available in the info dict: https://github.com/ppaquette/gym-super-mario/tree/master/ppaquette_gym_super_mario
            distance        # Total distance from the start (x-axis)
            life            # Number of lives Mario has (3 if Mario is alive, 0 is Mario is dead)
            score           # The current score
            coins           # The current number of coins
            time            # The current time left
            player_status   # Indicates if Mario is small (value of 0), big (value of 1), or can shoot fireballs (2+)
        """
        state = info['distance']
        # putting a default value to a dictionary: https://www.codecademy.com/en/forum_questions/51ae28cf01033cc6d200497d
        Q.setdefault(state, {'up':0, 'L':0, 'down':0,'R':0,'JUMP':0,'B':0 })
        # itertools.count() is similar to 'while True:' but can break for testing based on t
        for t in itertools.count():
            # generate a random num between 0 and 1 e.g. 0.35, 0.73 etc..
            # if the generated num is smaller than epsilon, we follow exploration policy
            if np.random.random() <= epsilon:
                # select a random action from set of all actions
                #max_q_action = random.choice(Q[state].keys())      # done to use action name later
                                                                    # PYTHON2
                max_q_action = random.choice(list(Q[state].keys())) # PYTHON3

                action = action_dict[str(max_q_action)]
            # if the generated num is greater than epsilon, we follow exploitation policy
            else:
                # select an action with highest value for current state
                max_q_action =  max(Q[state], key=(lambda key: Q[state][key])) #not fully sure about lambdas >.< https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
                action = action_dict[str(max_q_action)]
            # apply selected action, collect values for next_state and reward


            observation, reward, done, info = env.step(action)
            next_state = info['distance']
            Q.setdefault(next_state, {'up':0, 'L':0, 'down':0, 'R':0, 'JUMP':0, 'B':0 })
            max_next_state_action = max(Q[next_state], key=lambda key: Q[next_state][key])
            # Calculate the Q-learning target value
            Q_target = reward + discount_factor*Q[next_state][max_next_state_action]
            # Calculate the difference/error between target and current Q
            Q_delta = Q_target - Q[state][str(max_q_action)] - standing_penalty
            # Update the Q table, alpha is the learning rate
            Q[state][str(max_q_action)] = Q[state][str(max_q_action)] + (alpha * Q_delta)

            # break if done, i.e. if end of this episode
            if done:
                break
            # make the next_state into current state as we go for next iteration
            state = next_state
        # gradualy decay the epsilon
        if epsilon > 0.1:
            epsilon -= 1.0/num_episodes

    saveQ(Q,num_episodes, functionName='q_learning')
    env.close()
    return Q    # return optimal Q




"""Q-learning with distance and score.
"""
#workerd on this instead of ql_box because accounting for a boxed (limited) environment means that the policy would not be a reinforced-learning?
def ql_distScore(env, num_episodes, alpha=0.85, discount_factor=0.99):
    # decaying epsilon, i.e we will divide num of episodes passed
    epsilon = 1.0

    # call setdefault for a new state.
    if hasPickleWith("ql_distScore"):
        Q = loadLatestWith("ql_distScore")[0]
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
    saveQ(Q, num_episodes, functionName='ql_distScore')
    return Q  # return optimal Q


"""Q-learning with box as a state. The box size will be 2 blocks away from Mario as a default.
"""
#worked on this instead of ql_box because accounting for a boxed (limited) environment means that the policy would not be a reinforced-learning?
def ql_box(env, num_episodes, alpha=0.85, discount_factor=0.99, boxSize=2):
    # decaying epsilon, i.e we will divide num of episodes passed
    epsilon = 1.0
    last_episode = 0 #This is so we can run episodes in batches because running many at once takes a lot of time!
    
    # call setdefault for a new state.
    if hasPickleWith("ql_box"):
        Q,last_episode = loadLatestWith("ql_box")
        
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
    saveQ(Q, num_episodes + last_episode, functionName='ql_box',boxSize=boxSize)
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

#TODO: Added docustring but this function is not complete yet, will do after we clear level 1.
"""This function takes an environment and Q table and checks if the optimal actions
at each state is actually being taken. """
def test_algorithm(env,Q):
    stuck_capacity = 5
    stuck = []

    observation = env.reset()
    total_reward = 0
    action = [0]*6
    observation,reward,done,info = env.step(action)
    state = info['distance']
    action_dict = {'up':    [1, 0, 0 ,0, 0, 0],
                   'L':     [0, 1, 0, 0, 0, 0],
                   'down':  [0, 0, 1, 0, 0, 0],
                   'R':     [0, 0, 0, 1, 0, 0],
                   'JUMP':  [0, 0, 0, 0, 1, 0],
                   'B':     [0, 0, 0, 0, 0, 1]}
    for t in itertools.count():
        # selection the action with highest values i.e. best action
        max_q_action = max(Q[state], key=lambda key: Q[state][key])
#        if len(stuck)<stuck_capacity:
#            stuck.append(max_q_action)
#        stuck = [] if t%10==0 else stuck
        #print('t is ',t )
#        if isStuck(stuck,stuck_capacity):
#            print('is stuck :(')
#            max_q_action = 'R' #random.choice(Q[state].keys())
#            stuck = []
        print("Optimal action is: " + max_q_action)
        action = action_dict[str(max_q_action)]
        print("Action is: " , action)
        # apply selected action
        observation, reward, done,info = env.step(action)
        print("Reward: ", reward)
        next_state = info['distance']
        # calculate total reward
        total_reward += reward
        if done:
            print(total_reward)
            break
        state = next_state
    return total_reward
"""Possibly a helper function to test_algorithm"""
def isStuck(stuck,capacity):
    if len(stuck) == capacity:
        stuck = []
    return len(np.unique(stuck)) == 1

if __name__ == "__main__":
    env = gym.make('SuperMarioBros-1-1-Tiles-v0')  # remember need to make the environment each time

    Q = ql_box(env, 1)
    box_Q = loadLatestWith('ql_box')
    
    #Q = ql_distScore(env, 10)

    #Q = q_learning(env, 5)
    #loaded_Q2 = loadLatest()
    #loaded_Q = loadQ('q_248_10.pickle')
    #assert(loaded_Q==Q)
