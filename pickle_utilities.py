#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:32:26 2018

@author: NewType
"""

import pickle
import glob
import sys
import os

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