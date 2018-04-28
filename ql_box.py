#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:44:59 2018

@author: Jonathan Scott, Jinyoung Lim, So Jin Oh
"""

import pickle_utilities as pu
import numpy as np
import itertools
import random
from time import sleep





"""Q-learning with box as a state. The box size will be 2 blocks away from Mario as a default.
"""
#TODO reassign episilon based on previous eps, need to record episolon after last episode.
#TODO decrease eps based on (20000)- # of eps ran so far
#worked on this instead of ql_box because accounting for a boxed (limited) environment means that the policy would not be a reinforced-learning?
def ql_box(env, num_episodes, alpha=0.85, discount_factor=0.99, boxSize=2):

    max_episode = 5000.0 #assuming we need at most 5,000 episodes
    last_episode = 0 #This is so we can run episodes in batches because running many at once takes a lot of time!
    
    
    funcName = "ql_box_size" + str(boxSize)
    epsilon = pu.getEpsilon(funcName)
    
    # call setdefault for a new state.
    # if pu.hasPickleWith("ql_box",'2','QAndRewardsAndASC_box3_epsilonsByDistance_Alpha/*.pickle'):
    #     Q,last_episode = pu.loadLatestWith("ql_box")
    #
    # else:
    #     # not sure if "0000000000003000000000000" is a correct initial box (state) that is comparable to 0
    #     Q = {"0000000000003000000000000": {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'B': 0,'R_JUMP': 0}}
        # the state "0000000000003000000000000" represents the box that looks as below
        # 00000
        # 00000
        # 00300
        # 11111
        # 11111

    Q = {"0000000000003000000000000": {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'B': 0, 'R_JUMP': 0}}

    action = [0, 0, 0, 0, 0, 0]  # Do nothing
    action_dict = {'up':    [1, 0, 0, 0, 0, 0],
                   'L':     [0, 1, 0, 0, 0, 0],
                   'down':  [0, 0, 1, 0, 0, 0],
                   'R':     [0, 0, 0, 1, 0, 0],
                   'JUMP':  [0, 0, 0, 0, 1, 0],
                   'B':     [0, 0, 0, 0, 0, 1],
                   'R_JUMP':[0, 0, 0, 1, 1, 0]}


    for episode in range(num_episodes):
        # if pu.hasPickleWith('ql_box','2','bestActions/*pickle'):
        #     bestActions,bestDistance,bestReward = pu.loadBestAction('ql_box')
        #     useBA = False
        #
        # else:
        #     bestActions = []
        #     bestDistance = -1
        #     bestReward = 0
        #     useBA = False
        
        # good_distance = False # A distance I deemed worthy.
        #useBA = len(bestActions)>0#at the start of each ep we want to use best actions, if none then explore til end of ep.
        # total_reward = bestReward
        total_reward = 0
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

        Q.setdefault(state, {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'B': 0, 'R_JUMP': 0})
        useBA = False #TODO: for non-Qtable updating

        while not done:
            
        #for t in itertools.count():
            # generate a random num between 0 and 1 e.g. 0.35, 0.73 etc..
            # if the generated num is smaller than epsilon, we follow exploration policy
            if not useBA:
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
                # bestActions.append(action)
                # apply selected action, collect values for next_state and reward
                observation, reward, done, info = env.step(action)
                currDistance = info['distance']
                total_reward+=reward
               
                #print("Qbox reward is: "+str(reward))
                next_state = getBox(observation, boxSize)
                #mario_loc = next_state.index('3')
#                if mario_loc!=  15 and next_state[mario_loc+1]=='1' and not done:#we will jump more and move right
#                    jump = [0, 0, 0, 0, 1, 0]
#                    right = [0, 0, 0, 1, 0, 0]
#                    jumpAndRight = [0, 0, 0, 1, 1, 0]
#                    #print('getting help.')
#                    
#                                    
#                    observation, reward, done, info = env.step(jump)
#                    bestActions.append(jump)
#                    observation, reward, done, info = env.step(jump)
#                    bestActions.append(jump)
#                    observation, reward, done, info = env.step(jump)
#                    bestActions.append(jump)
#                    observation, reward, done, info = env.step(jump)
#                    bestActions.append(jump)
#                    observation, reward, done, info = env.step(jump)
#                    bestActions.append(jump)
#                    observation, reward, done, info = env.step(jump)
#                    bestActions.append(jump)
#                    observation, reward, done, info = env.step(jumpAndRight)
#                    bestActions.append(jumpAndRight)
#                    observation, reward, done, info = env.step(jumpAndRight)
#                    bestActions.append(jumpAndRight)
#                    observation, reward, done, info = env.step(jumpAndRight)
#                    bestActions.append(jumpAndRight)
#                  
                #next_state = getBox(observation, boxSize)
                    
                
                Q.setdefault(next_state, {'up': 0, 'L': 0, 'down': 0, 'R': 0, 'JUMP': 0, 'B': 0,'R_JUMP': 0})
                max_next_state_action = max(Q[next_state], key=lambda key: Q[next_state][key])
                # Calculate the Q-learning target value
                Q_target = reward + discount_factor * Q[next_state][max_next_state_action]
                # Calculate the difference/error between target and current Q
                Q_delta = Q_target - Q[state][str(max_q_action)]
                # Update the Q table, alpha is the learning rate
                # Q[state][str(max_q_action)] = Q[state][str(max_q_action)] + (alpha * Q_delta) #TODO: uncomment to update the Q Table
                # make the next_state into current state as we go for next iteration
                state = next_state #last_episode+episode

            else:
              currDistance = info['distance']
              # if currDistance < (bestDistance-25):
              #      print("Using BA")
              #      for action in bestActions:
              #          observation, reward, done, info = env.step(action)
              #          if currDistance>= bestDistance or done:#we should start exploring again
              #              break
              #          currDistance = info['distance']
              #      useBA = False
              #      print("Not using BA, exploring.")

            
            
#            print('total reward is: ',total_reward)
#            print('bestReward is: ',bestReward)
#             if (info['distance'] >= bestDistance) and (total_reward > bestReward):
#                 bestDistance = info['distance']
#                 bestReward = total_reward
                # pu.saveBestActions((bestActions,bestDistance,bestReward),'ql_box',boxSize=boxSize,bestDistance=bestDistance)

            # break if done, i.e. if end of this episode
            if done or info['distance']>=3266:
                # pu.saveBestActions((bestActions,bestDistance,bestReward),'ql_box',boxSize=boxSize,bestDistance=bestDistance)
                break
           
            if info['distance']%200==0:
                good_distance = True
            
         # gradualy decay the epsilon and everytime number of iterations is divisible by 50, decreas episilon by 1.5%
        if epsilon > 0.26:
            if (last_episode%5 == 0) and (good_distance):
                epsilon-= 0.0100#since this is rare,take off a huge randomness
            else:
                epsilon -= 1.0/max_episode

    
    ep_dist,ep_reward = info['distance'],info['total_reward'] #last recorded distance , last recorded reward from episodes
    # pu.saveQ(Q, num_episodes + last_episode, functionName='ql_box',boxSize=boxSize) #TODO: uncomment to save Q tables
    pu.collectData(num_episodes + last_episode,ep_reward,ep_dist,epsilon,functionName=funcName)
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
            if(marioPosX+j)<16 and (marioPosY+i<13):
                currBoxPos = observation[marioPosY+i, marioPosX+j]

#                currBoxPos = currBoxPos.item(0)
                box += str(currBoxPos)
#    print(box)
    return box

#TODO: Added docustring but this function is not complete yet, will do after we clear level 1.
"""This function takes an environment and Q table and checks if the optimal actions
at each state is actually being taken. """
def test_algorithm(env,boxSize=2,Q=None):
    if not Q:
        Q = pu.loadLatestWith('ql_box')[0]
    observation = env.reset()
    total_reward = 0
    action = [0]*6
    observation,reward,done,info = env.step(action)
    marioPosY, marioPosX = np.where(observation == 3)
    while marioPosX.size == 0:
            action = [0, 0, 0, 1, 0, 0]
            observation, reward, done, info = env.step(action)
            marioPosY, marioPosX = np.where(observation == 3)


    state = getBox(observation, boxSize)
    #print(state,Q[state])
    #0000000000003001111111111
    action_dict = {'up':    [1, 0, 0 ,0, 0, 0],
                   'L':     [0, 1, 0, 0, 0, 0],
                   'down':  [0, 0, 1, 0, 0, 0],
                   'R':     [0, 0, 0, 1, 0, 0],
                   'JUMP':  [0, 0, 0, 0, 1, 0],
                   'B':     [0, 0, 0, 0, 0, 1]}
    for t in itertools.count():
        # selection the action with highest values i.e. best action
        max_q_action = max(Q[state], key=lambda key: Q[state][key])
        #print("Optimal action is: " + max_q_action)
        action = action_dict[str(max_q_action)]
        #print("Action is: " , action)
        # apply selected action
        observation, reward, done,info = env.step(action)
        #print("Q-box reward: "+str(reward))
        next_state = getBox(observation, boxSize)
        # calculate total reward
        total_reward += reward
        
#        print('reward is ',reward)
#        print(info['total_reward'])
#        print('total_reward var ',total_reward)
        
        if done:
            print(total_reward)
            break
        state = next_state
        #print("This stuck state has q values of: ", Q[state])
    env.close()
    return total_reward


