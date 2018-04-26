'''
First comment

'''
import numpy as np  # including matrix
import math  # including inf
import time
import numbers

# for function sorted key
from operator import itemgetter, attrgetter

TS = [9, -0.86728, -0.85935, -0.90491, -0.93489, -0.94114, -0.95455, -0.97459, -0.98291, -0.9838, -0.98984, -0.99807,
      -1.0024, -1.0062, -1.0126, -1.0187, -1.0219, -1.0239, -1.0262, -1.0278, -1.0262, -1.0223, -1.02, -1.0194, -1.0177,
      -1.0149, -1.0151, -1.0163, -1.0176, -1.0152, -1.0154, -1.0145, -1.0146, -1.0095, -1.0041, -0.99684, -0.99111,
      -0.98579, -0.97912, -0.9555, -0.88506, -0.74187, -0.5201, -0.24511, 0.047446, 0.3207, 0.55772, 0.75961, 0.94007,
      1.0993, 1.2171, 1.2646, 1.2331, 1.143, 1.0264, 0.90448, 0.7784, 0.64456, 0.50064, 0.35298, 0.21007, 0.080998,
      -0.031882, -0.1312, -0.21783, -0.29466, -0.36019, -0.41768, -0.45864, -0.48529, -0.49594, -0.50157, -0.50174,
      -0.49816, -0.48962, -0.47862, -0.46806, -0.45706, -0.44625, -0.43238, -0.4185, -0.40396, -0.39297, -0.38334,
      -0.37905, -0.37982, -0.38702, -0.39758, -0.4125, -0.43267, -0.45687, -0.4803, -0.49751, -0.50792, -0.50866,
      -0.50495, -0.49558, -0.49266, -0.49105, -0.49565, -0.49692, -0.50196, -0.50481, -0.50211, -0.48218, -0.44693,
      -0.40418, -0.35601, -0.2998, -0.2091, -0.049675, 0.2321, 0.6404, 1.1288, 1.6003, 1.9829, 2.2503, 2.4285, 2.5523,
      2.6416, 2.6974, 2.7156, 2.7051, 2.667, 2.6063, 2.5179, 2.4129, 2.2898, 2.1503, 1.9936, 1.8343, 1.6876, 1.5631,
      1.4713, 1.4093, 1.3789, 1.358, 1.3466, 1.328, 1.3117, 1.2924, 1.2805, 1.2729, 1.268, 1.2612, 1.2493, 1.2392,
      1.2309, 1.2372, 1.2574, 1.3039, 1.3669, 1.4428, 1.5153, 1.583, 1.6498, 1.7236, 1.8061, 1.8826, 1.9335, 1.9316,
      1.8665, 1.7341, 1.5523, 1.3327, 1.0962, 0.85633, 0.63143, 0.42859, 0.24507, 0.075226, -0.083841, -0.22557,
      -0.35114, -0.45708, -0.55049, -0.62429, -0.68543, -0.7288, -0.76416, -0.78726, -0.79964, -0.79806, -0.78398,
      -0.75336, -0.6932, -0.59575, -0.46283, -0.31299, -0.16183, -0.03404, 0.053763, 0.092858, 0.10112, 0.096811,
      0.090684, 0.081882, 0.071976, 0.063568, 0.060536, 0.067846, 0.087287, 0.11341, 0.13398, 0.14008, 0.12502, 0.08758,
      0.02985, -0.040089, -0.11459, -0.18894, -0.25992, -0.32855, -0.39606, -0.466, -0.53294, -0.59353, -0.64278,
      -0.68533, -0.71932, -0.74483, -0.75939, -0.76711, -0.77212, -0.77479, -0.77247, -0.76227, -0.74828, -0.73313,
      -0.72092, -0.70929, -0.70026, -0.69203, -0.68683, -0.68266, -0.68041, -0.67831, -0.67508, -0.66805, -0.65695,
      -0.6434, -0.62904, -0.61631, -0.60635, -0.60119, -0.59829, -0.59688, -0.5945, -0.59072, -0.58315, -0.57156,
      -0.55792, -0.54535, -0.53518, -0.52604, -0.51699, -0.50585, -0.49528, -0.48979, -0.4907, -0.49429, -0.50522,
      -0.52929, -0.55947, -0.59183, -0.64158, -0.70929, -0.75715, -0.77593, -0.8217, -0.92773]
# TS = [9, -79,-76,-73,-69,-6,-63,-61]
ComparedTS = [9, -0.81704, -0.7331, -0.62782, -0.49572, -0.33489, -0.14328, 0.077817, 0.32273, 0.58623, 0.86202, 1.1402,
              1.409, 1.6562, 1.8708, 2.0462, 2.1814, 2.2777, 2.337, 2.3604, 2.3474, 2.2983, 2.2156, 2.1047, 1.9732,
              1.8298, 1.6821, 1.5356, 1.3929, 1.2539, 1.1168, 0.97972, 0.84133, 0.70163, 0.5619, 0.42341, 0.28756,
              0.15633, 0.030907, -0.086822, -0.19325, -0.28775, -0.37099, -0.44154, -0.50288, -0.55919, -0.60734,
              -0.64944, -0.68914, -0.72084, -0.74642, -0.77102, -0.78907, -0.80158, -0.81462, -0.82374, -0.82926,
              -0.83723, -0.84361, -0.84646, -0.85046, -0.85335, -0.85325, -0.8547, -0.85653, -0.8551, -0.85318, -0.851,
              -0.84543, -0.83971, -0.83603, -0.83013, -0.82232, -0.81317, -0.79532, -0.76505, -0.72024, -0.65066,
              -0.5499, -0.41674, -0.24633, -0.041712, 0.18476, 0.42319, 0.65994, 0.87865, 1.0726, 1.2386, 1.3741,
              1.4852, 1.5787, 1.6588, 1.733, 1.8066, 1.8804, 1.9567, 2.0361, 2.1164, 2.1965, 2.2737, 2.3417, 2.3937,
              2.423, 2.423, 2.3918, 2.3322, 2.2493, 2.1508, 2.0448, 1.9357, 1.8261, 1.7169, 1.6057, 1.4913, 1.3738,
              1.2522, 1.1282, 1.0045, 0.88026, 0.75705, 0.63776, 0.52154, 0.4105, 0.30946, 0.2178, 0.1357, 0.065827,
              0.0046436, -0.050112, -0.095075, -0.13226, -0.16306, -0.18265, -0.19389, -0.2013, -0.20165, -0.19899,
              -0.1988, -0.19646, -0.19444, -0.19683, -0.19914, -0.20288, -0.21124, -0.21988, -0.22919, -0.24159,
              -0.25397, -0.26735, -0.28502, -0.30541, -0.32869, -0.35593, -0.38403, -0.41137, -0.43885, -0.46553,
              -0.49129, -0.51709, -0.54114, -0.56133, -0.57814, -0.5922, -0.60425, -0.61673, -0.63031, -0.64285,
              -0.65415, -0.66523, -0.67607, -0.68919, -0.70681, -0.72626, -0.74562, -0.76412, -0.77858, -0.7893,
              -0.79904, -0.80675, -0.81274, -0.81859, -0.8218, -0.8217, -0.81948, -0.81255, -0.79947, -0.7802, -0.75171,
              -0.71234, -0.66291, -0.60334, -0.53546, -0.46369, -0.39122, -0.32103, -0.25638, -0.19821, -0.1464,
              -0.10158, -0.063726, -0.033104, -0.011303, 0.00062289, 0.0025969, -0.0050291, -0.020476, -0.040994,
              -0.064514, -0.089478, -0.11464, -0.13947, -0.16332, -0.18513, -0.2045, -0.22162, -0.23703, -0.25179,
              -0.26669, -0.28172, -0.29704, -0.31318, -0.3306, -0.35055, -0.37401, -0.40028, -0.42898, -0.45986,
              -0.49117, -0.52302, -0.55609, -0.5886, -0.62042, -0.65188, -0.68096, -0.70766, -0.73273, -0.75464,
              -0.77372, -0.79142, -0.80722, -0.82178, -0.83603, -0.8484, -0.85811, -0.86577, -0.87112, -0.87518,
              -0.87965, -0.88412, -0.8879, -0.89106, -0.893, -0.8938, -0.89463, -0.89539, -0.89555, -0.89557, -0.89569,
              -0.89623, -0.89821, -0.90157, -0.90536, -0.90961, -0.91489, -0.92182, -0.93115, -0.94213, -0.9522,
              -0.95905, -0.96174, -0.96086]
# ComparedTS = [18, -87,-84,-82,-81,-79,-76,-74]
width = 10
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

    print(matrix_DTW)
    return matrix_DTW[i][j]


start_time = time.time()
print(originDTW(Trim_TS, Trim_ComparedTS))
print(time.time() - start_time)


def adaptiveWindowDTW(ts1, ts2):
    #
    # boundryLeftRight=np.array([[0, 0] for i in range(len(ts1))])

    # my algorithm

    # initializatoin of matrix
    matrix_ADTW = [[math.inf for i in range(len(ts1))] for i in range(len(ts2))]

    # two indices of time series
    ts1_index = 0
    ts2_index = 0

    # set the value of origin
    matrix_ADTW[ts1_index][ts2_index] = abs(ts1[ts1_index] - ts2[ts2_index])
    current_minimum_distance = matrix_ADTW[ts1_index][ts2_index]

    # two lengths of time series
    ts1_length = len(ts1)  # m
    ts2_length = len(ts2)  # n

    comparedIncreasingQueue = [[matrix_ADTW[ts1_index][ts2_index], 0, 0]]

    # comparedIncreasingQueueIndex = []

    while True:
        # if #something happen:
        if len(comparedIncreasingQueue) != 0 and matrix_ADTW[ts1_length - 1][ts2_length - 1] < \
                comparedIncreasingQueue[0][0]:
            count = 0
            for i in matrix_ADTW:
                for j in i:
                    if j != math.inf:
                        count += 1

            print("Total {0}, we use {1}, usage {2}%".format(ts1_length * ts2_length, count,
                                                             count * 100 / (ts1_length * ts2_length)))
            # print(matrix_ADTW)
            return matrix_ADTW[ts1_length - 1][ts2_length - 1]
        else:

            comparedIncreasingQueue.pop(0)
            # right direction
            if ts1_index + 1 < ts1_length and ts2_index + 1 < ts2_length:
                distance_dia = abs(ts1[ts1_index + 1] - ts2[ts2_index + 1])

                cumulativeDia = matrix_ADTW[ts1_index][ts2_index] + distance_dia
                if cumulativeDia < matrix_ADTW[ts1_index + 1][ts2_index + 1]:
                    matrix_ADTW[ts1_index + 1][ts2_index + 1] = cumulativeDia
                    insertList = [cumulativeDia, ts1_index, ts2_index]
                    if insertList not in comparedIncreasingQueue:
                        comparedIncreasingQueue.insert(0, insertList)

            # down direction
            if ts1_index + 1 < ts1_length:
                distance_down = abs(ts1[ts1_index + 1] - ts2[ts2_index])

                cumulativeDown = matrix_ADTW[ts1_index][ts2_index] + distance_down
                if cumulativeDown < matrix_ADTW[ts1_index + 1][ts2_index]:
                    matrix_ADTW[ts1_index + 1][ts2_index] = cumulativeDown
                    insertList = [cumulativeDown, ts1_index, ts2_index]
                    if insertList not in comparedIncreasingQueue:
                        comparedIncreasingQueue.insert(0, insertList)

            #right direction
            if ts2_index + 1 < ts2_length:
                distance_right = abs(ts1[ts1_index] - ts2[ts2_index + 1])

                cumulativeRight = matrix_ADTW[ts1_index][ts2_index] + distance_right
                if cumulativeRight < matrix_ADTW[ts1_index][ts2_index + 1]:
                    matrix_ADTW[ts1_index][ts2_index + 1] = cumulativeRight

                    insertList = [cumulativeRight, ts1_index, ts2_index]
                    if insertList not in comparedIncreasingQueue:
                        comparedIncreasingQueue.insert(0, insertList)

            distance_minimum = min(distance_dia, distance_down, distance_right)

            current_minimum_distance = current_minimum_distance + distance_minimum
            # sort the queue
            sorted(comparedIncreasingQueue, key=itemgetter(0))
            print(comparedIncreasingQueue)
            ts1_index = comparedIncreasingQueue[0][1]
            ts2_index = comparedIncreasingQueue[0][2]

            # the first step


def updateQueue(queue, inputValue, ts1_index, ts2_index):
    if insertList in queue:
        return queue
    else:
        queue.insert(0, insertList)
        queue = sorted(queue, key=itemgetter(0))
        return queue


def quicksort(queue):
    if len(queue) <= 1:
        return queue
    midcell = queue[len(queue) // 2]
    return quicksort([i for i in queue if i < midcell]) + [midcell] + quicksort([i for i in queue if i > midcell])


#
print(adaptiveWindowDTW(Trim_TS, Trim_ComparedTS))
# print(time.time() - start_time)
