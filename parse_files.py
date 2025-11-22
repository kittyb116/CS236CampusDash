'''
Code written here is to parse the files from both dashers.csv
and tasklog.csv
'''
import csv
import numpy as np
import pandas as pd
import random

'''
Parse a dasher csv file and returns info as a list of tuples (loc, start, end)
'''
def read_dashers(fname):
    result = []
    # Open the file
    with open(fname) as in_file:
        reader = csv.reader(in_file)
        next(reader) # skip the header
        for line in reader:
        # turn each entry into a tuple of integers
            result.append(tuple(map(int, line)))
    return result

'''
Parses a tasklog csv and returns a tuple with rel information (loc, appear, end, reward)
'''
def read_tasklog(fname, reward_range: tuple):
    result = []
    min_, max_ = reward_range
    # Open the file
    with open(fname) as in_file:
       reader = csv.DictReader(in_file)
       for line in reader:
           end = int(line['minute'])
           start_pos = int(line['VERTEX'])
           # select any appearance time between 0 and time task needs to be completed
           appear = random.randrange(0, end) if end > 0 else 0
           reward = random.randrange(min_, max_)
           result.append((start_pos, appear, end, reward))
    return result

'''
The code below here are some additional preprocessing done to the file
In order to make my simulations easier, I would like to shift all my times
so that the first ever event starts at time = 0 instead of some other arbitrary time

I've already done some preprocessing on the dashers.csv to ensure that each dasher's start_time
is less than their exit_time

Additionally, I require that every task have a appear time that is less than the target time
'''

def adjust_time_zero(dasher_fn, task_fn):
    dasher = pd.read_csv(dasher_fn)
    task = pd.read_csv(task_fn)

    # find the minimum value in both files
    dasher_min = dasher['start_time'].min()
    task_min = task['minute'].min()-1
    
    # find the absolute minimum
    zero_time = min(dasher_min, task_min)

    # modify the times in the dasher file
    dasher['start_time'] = dasher['start_time'] - zero_time
    dasher['exit_time'] = dasher['exit_time'] - zero_time

    # modify the times in the tasklog file
    task['minute'] = task['minute'] - zero_time
    
    # save both the new_files
    new_dasher_fn = dasher_fn.split('.')[0] + '_time_adjusted.csv'
    new_task_fn = task_fn.split('.')[0] + '_time_adjusted.csv'

    dasher.to_csv(new_dasher_fn, index = False)
    task.to_csv(new_task_fn, index = False)

