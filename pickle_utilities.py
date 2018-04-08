#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 17:32:26 2018

@author: Jonathan Scott, Jinyoung Lim, So Jin Oh
"""

import pickle
import glob
import sys
import os
import pandas as pd

def getLastDist(functionName):
    filename = 'reward_stats/'+functionName+'_reward_stats.csv'
    if not os.path.isfile(filename): #if there is no previous episodes ran based on the existance of file, return 0 as the last distance
        return 0
    df = pd.read_csv(filename)
    lastDist = df.iloc[-1]['dist']
    print("Last distance is: " + str(lastDist))
    return lastDist

def getEpsilon(functionName):
    filename = 'reward_stats/'+functionName+'_reward_stats.csv'
    #if there are no previous episodes return 1 since we are just starting.
    if not os.path.isfile(filename):
        return 1
    df = pd.read_csv(filename)
    lastEpsilon = df.iloc[-1]['epsilon']
    print("Last epsilon is: " + str(lastEpsilon))
    return lastEpsilon


def hasPickleWith(functionName, boxSize='',path=''):
    """This function checks to see if there exists pickle files. """
    database = filter(os.path.isfile, glob.glob(path))
    return len(database)>0

#File name based on furthest distance, nb_episodes (q_furthestDistance_numEpisode.pickle)
#https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
def saveQ(Q, num_episodes, functionName,boxSize=""):
    """
    Saves a Q-table 
    """
    if functionName == 'q_learning':
        with open('Q-tables/'+functionName + '_' +str(num_episodes)+'.pickle', 'wb') as handle:
            pickle.dump(Q, handle, protocol=2)
    else:
        with open('Q-tables/'+functionName + '_'+str(num_episodes)+'_'+str(boxSize)+'.pickle', 'wb') as handle:
            pickle.dump(Q, handle, protocol=2)
    print("Saved Q table succesfully for "+str(num_episodes)+" episodes!")
    return

def loadQ(filename):
    with open(filename, 'rb') as handle:
        unserialized_data = pickle.load(handle)
    return unserialized_data


def loadLatestWith(functionName):
    """ 
    #https://stackoverflow.com/questions/9492481/check-that-a-type-of-file-exists-in-python
    #returns max ep num as well so we can pick up where we left off.
    """
    episode_stamps = []
    f_name = ''
    # for file in _glob.glob("*.pickle"):   #PYTHON2
    for file in glob.glob("Q-tables/*.pickle"):      #PYTHON3    https://stackoverflow.com/questions/44366614/nameerror-name-glob-is-not-defined
        if functionName in file and functionName=='q_learning':
            f_name = str(file).replace("Q-tables/","",1)
            #Most recent is defined as the one that makes it the furtherest distance.
            f_name_offset = len(functionName)-1 # to get the index of where it starts
            e_stamp = f_name[f_name_offset+2:]
            end_ = e_stamp.index('_')
            episode_stamps.append(int (e_stamp[:end_]) )
        elif functionName in file:
            f_name = str(file).replace("Q-tables/","",1)
            box_num = f_name[-1]
            #Most recent is defined as the one that makes it the furtherest distance.
            f_name_offset = len(functionName)-1 # to get the index of where it starts
            e_stamp = f_name[f_name_offset+2:]
            end_ = e_stamp.index('_')
            episode_stamps.append(int (e_stamp[:end_]) )

    max_e_stamp = str(max(episode_stamps))
    for file in glob.glob("Q-tables/*.pickle"):
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

"""Saves a csv with Pandas"""
def collectData(episode_num,reward,dist,epsilon,functionName):
    log = {'episode_num': [episode_num] ,
           'reward': [reward],
           'dist': [dist],
           'epsilon':[epsilon]}
    stats_df = pd.DataFrame(data=log)
    filename = 'reward_stats/'+functionName+'_reward_stats.csv'
    if not os.path.isfile(filename):
        stats_df.to_csv(filename, sep=',', encoding='utf-8',index=False)
    stats = pd.read_csv(filename)
    stats = stats.append(stats_df,ignore_index=True)
    stats.to_csv(filename, sep=',', encoding='utf-8',index=False)

    print("Saved Episode stats succesfully for "+str(episode_num)+" episodes!")
    return

def saveBestActions(bestActions,functionName,boxSize="",bestDistance=0):
    """
    Saves the best action for a maximum distance achieved and reward."
    """
    if functionName == 'q_learning':
        with open('bestActions/'+functionName +'_'+str(bestDistance)+'.pickle', 'wb') as handle:
            pickle.dump(bestActions, handle, protocol=2)
    else:
        with open('bestActions/'+functionName +'_'+str(bestDistance)+'_'+str(boxSize)+'.pickle', 'wb') as handle:
            pickle.dump(bestActions, handle, protocol=2)
    #print("Saved Best Actions succesfully.")
    return

def loadBestAction(functionName):
    distance_stamps = []
    f_name = ''
    for file in glob.glob("bestActions/*.pickle"):      
        if functionName in file and functionName=='q_learning':
            f_name = str(file).replace("bestActions/","",1)
            #Most recent is defined as the one that makes it the furtherest distance.
            f_name_offset = len(functionName)-1 # to get the index of where it starts
            dist_stamp = f_name[f_name_offset+2:]
            end_ = dist_stamp.index('_')
            distance_stamps.append(int (dist_stamp[:end_]) )
        elif functionName in file:
            f_name = str(file).replace("bestActions/","",1)
            box_num = f_name[-1]
            #Most recent is defined as the one that makes it the furtherest distance.
            f_name_offset = len(functionName)-1 # to get the index of where it starts
            dist_stamp = f_name[f_name_offset+2:]
            end_ = dist_stamp.index('_')
            distance_stamps.append(int (dist_stamp[:end_]) )

    max_e_stamp = str(max(distance_stamps))
    for file in glob.glob("bestActions/*.pickle"):
        if functionName in file and functionName=='q_learning':
            f_name = str(file)
            if max_e_stamp in f_name:
                break
        elif functionName in file and box_num in file:
            f_name = str(file)
            if max_e_stamp in f_name:
                break
   
           
    print("Loaded Best Actions: " + f_name)
    return loadQ(f_name) 