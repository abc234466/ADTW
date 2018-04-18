'''
First comment

'''
import numpy as np  # including matrix
import math  # including inf
import numbers

# for function sorted key
from operator import itemgetter, attrgetter

TS = [9, 2,9,8,8,5,4,2,1,5]
ComparedTS = [18, 3,8,7,4,1,3,2,1,7]

# Trim the label of the time series
Trim_TS = TS[1:]
Trim_ComparedTS = ComparedTS[1:]


# Origin DTW

def originDTW(ts1, ts2):
    # initializatoin of matrix
    matrix_DTW = np.array([[math.inf for i in range(len(ts1) + 1)] for i in range(len(ts2) + 1)])

    # set the value of origin
    matrix_DTW[0][0] = 0

    for i in range(1, len(ts1) + 1):
        for j in range(1, len(ts2) + 1):
            distance = abs(ts1[i - 1] - ts2[j - 1])
            matrix_DTW[i][j] = distance + min(matrix_DTW[i - 1][j], matrix_DTW[i][j - 1], matrix_DTW[i - 1, j - 1])
    return matrix_DTW[i][j]


#print(originDTW(Trim_TS, Trim_ComparedTS))


def adaptiveWindowDTW(ts1, ts2):
    #
    # boundryLeftRight=np.array([[0, 0] for i in range(len(ts1))])

    # my algorithm

    # initializatoin of matrix
    matrix_ADTW = np.array([[math.inf for i in range(len(ts1))] for i in range(len(ts2))])

    # two indices of time series
    ts1_index = 0
    ts2_index = 0

    # set the value of origin
    matrix_ADTW[ts1_index][ts2_index] = abs(ts1[ts1_index] - ts2[ts2_index])
    current_minimum_distance = matrix_ADTW[ts1_index][ts2_index]

    # two lengths of time series
    ts1_length = len(ts1)  # m
    ts2_length = len(ts2)  # n

    comparedIncreasingQueue = []

    # comparedIncreasingQueueIndex = []

    while True:
        # if #something happen:
        if len(comparedIncreasingQueue) != 0 and matrix_ADTW[ts1_length - 1][ts2_length - 1] < comparedIncreasingQueue[0][0]:
            return matrix_ADTW[ts1_length-1][ts2_length-1]

        else:
            if comparedIncreasingQueue:
                comparedIncreasingQueue.pop(0)
            if ts1_index + 1 < ts1_length and ts2_index + 1 < ts2_length:
                distance_dia = abs(ts1[ts1_index + 1] - ts2[ts2_index + 1])
                if current_minimum_distance + distance_dia < matrix_ADTW[ts1_index + 1][ts2_index + 1]:
                    matrix_ADTW[ts1_index + 1][ts2_index + 1] = matrix_ADTW[ts1_index][ts2_index] + distance_dia
                    comparedIncreasingQueue = updateQueue(comparedIncreasingQueue, current_minimum_distance + distance_dia, ts1_index + 1, ts2_index + 1)
            if ts1_index + 1 < ts1_length:
                distance_down = abs(ts1[ts1_index + 1] - ts2[ts2_index])
                if current_minimum_distance + distance_down < matrix_ADTW[ts1_index + 1][ts2_index]:
                    matrix_ADTW[ts1_index + 1][ts2_index] = matrix_ADTW[ts1_index][ts2_index] + distance_down
                    comparedIncreasingQueue = updateQueue(comparedIncreasingQueue, current_minimum_distance + distance_down, ts1_index + 1, ts2_index)
            if ts2_index + 1 < ts2_length:
                distance_right = abs(ts1[ts1_index] - ts2[ts2_index + 1])
                if current_minimum_distance + distance_right < matrix_ADTW[ts1_index][ts2_index + 1]:
                    matrix_ADTW[ts1_index][ts2_index + 1] = matrix_ADTW[ts1_index][ts2_index] + distance_right
                    comparedIncreasingQueue = updateQueue(comparedIncreasingQueue, current_minimum_distance + distance_right, ts1_index, ts2_index + 1)
            distance_minimum = min(distance_dia, distance_down, distance_right)

            current_minimum_distance = current_minimum_distance + distance_minimum
            ts1_index = comparedIncreasingQueue[0][1]
            ts2_index = comparedIncreasingQueue[0][2]
            # sort the queue
            # sorted(comparedIncreasingQueue, key = itemgetter(0))
            # the first step


def updateQueue(queue, inputValue, ts1_index, ts2_index):
    insertList = [inputValue, ts1_index, ts2_index]
    if insertList in queue:
        return queue
    else:
        queue.insert(0, insertList)
        queue = sorted(queue, key=itemgetter(0))
        return queue
    #     try:
    #         queue.insert(queue.index(inputValue), insertList)
    #     except:
    #         firstValueList = queue[:][0]
    #         for i in range(0, len(firstValueList)):
    #             if inputValue <= firstValueList[i]:
    #                 queue.insert(i, insertList)
    #
    #
    #
    #
    # if queue:
    #     if inputValue >= queue[len(queue)-1][0]:
    #         queue.insert(len(queue), [inputValue, [ts1_index, ts2_index]])
    #         return queue
    #     else:
    #         for i in range(0, len(queue)):
    #             if inputValue < queue[i][0]:
    #                 queue.insert(i, [inputValue, [ts1_index,ts2_index]])
    #             elif i == len(queue) - 1:
    #                 queue.insert(i+1, [inputValue, [ts1_index,ts2_index]])
    #         return queue


print(adaptiveWindowDTW(Trim_TS, Trim_ComparedTS))
