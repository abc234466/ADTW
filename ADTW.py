'''
First comment

'''
import numpy as np  # including matrix
import math  # including inf
import time
import numbers

from heapq import *  # include heap queue data structure

# for function sorted key
from operator import itemgetter, attrgetter

TS = [0, -0.57437, -0.52338, -0.44876, -0.46651, -0.59004, -0.73919, -0.80092, -0.81515, -0.84054, -0.86631, -0.87463,
      -0.84165, -0.8246, -0.82119, -0.87321, -0.95265, -0.99158, -0.99607, -0.97312, -0.96291, -0.97581, -0.991,
      -1.0034, -1.0318, -1.0289, -1.0003, -0.96894, -0.93858, -0.94099, -0.94074, -0.95839, -0.98225, -0.97491,
      -0.97035, -0.94755, -0.9466, -0.92715, -0.85785, -0.84109, -0.79947, -0.71206, -0.66973, -0.62611, -0.5308, -0.45,
      -0.39838, -0.37412, -0.38281, -0.37574, -0.32695, -0.29111, -0.29491, -0.26288, -0.20025, -0.16225, -0.12687,
      0.0041787, 0.13998, 0.21599, 0.31917, 0.3241, 0.29486, 0.32027, 0.30927, 0.29324, 0.35811, 0.48667, 0.68689,
      0.75718, 0.68731, 0.72988, 0.756, 0.83825, 0.83549, 0.7726, 0.7612, 0.68591, 0.64449, 0.61831, 0.63251, 0.68554,
      0.75817, 0.76966, 0.71291, 0.6277, 0.53471, 0.49493, 0.44773, 0.50884, 0.67876, 0.68074, 0.70983, 0.66441,
      0.57923, 0.55792, 0.48399, 0.5362, 0.55923, 0.5414, 0.55987, 0.45549, 0.42395, 0.43482, 0.38289, 0.35787, 0.32245,
      0.32855, 0.30438, 0.24037, 0.26538, 0.36893, 0.4629, 0.48232, 0.51493, 0.5247, 0.54878, 0.59439, 0.64871, 0.68087,
      0.73012, 0.87583, 0.84585, 0.83694, 0.91175, 0.88035, 0.83189, 0.77887, 0.69327, 0.64209, 0.57543, 0.48546,
      0.44086, 0.4511, 0.43631, 0.38398, 0.3661, 0.3977, 0.41469, 0.36438, 0.39034, 0.42047, 0.40196, 0.45055, 0.53085,
      0.57445, 0.71293, 0.8312, 0.84039, 0.81422, 0.753, 0.86442, 0.9814, 1.059, 1.1149, 1.0779, 1.1557, 1.2545, 1.1055,
      1.0041, 1.0152, 1.105, 1.0665, 0.86485, 0.8216, 0.84299, 0.81286, 0.72833, 0.67071, 0.61214, 0.59016, 0.4402,
      0.26667, 0.1254, 0.013672, -0.093055, -0.15143, -0.16624, -0.23237, -0.23758, -0.23032, -0.23634, -0.13697,
      -0.038152, 0.015485, -0.026498, -0.0077584, 0.046598, -0.0065994, -0.11331, -0.13581, -0.0078213, 0.041536,
      0.14153, 0.18556, 0.22912, 0.36653, 0.42455, 0.49413, 0.57844, 0.74013, 0.81693, 0.88616, 0.92862, 1.0556, 1.1756,
      1.1968, 1.3013, 1.4791, 1.5914, 1.5908, 1.4715, 1.2686, 1.1835, 1.3008, 1.4991, 1.6466, 1.6034, 1.4668, 1.4387,
      1.6441, 1.8007, 1.6191, 1.2327, 0.99935, 0.93658, 0.91436, 0.90126, 0.94476, 1.0469, 1.1296, 1.1422, 1.3731,
      1.413, 1.2952, 1.1572, 0.69969, 0.41586, 0.29386, 0.025245, -0.2437, -0.46344, -0.66026, -0.85575, -0.98252,
      -1.1116, -1.1931, -1.2402, -1.2808, -1.305, -1.3215, -1.3253, -1.3511, -1.4024, -1.4679, -1.5513, -1.6449,
      -1.7223, -1.7909, -1.8413, -1.8743, -1.8953, -1.9127, -1.9274, -1.9351, -1.9419, -1.9489, -1.9547, -1.9578,
      -1.9625, -1.9668, -1.9673, -1.9704, -1.9756, -1.9798, -1.9821, -1.9849, -1.9896, -1.992, -1.9937, -1.9964,
      -2.0006, -2.0039, -2.0059, -2.0082, -2.0104, -2.0112]

ComparedTS = [0, -0.59474, -0.54581, -0.55004, -0.62582, -0.71261, -0.83124, -0.92234, -0.92832, -0.93328, -0.96047,
              -0.97991, -0.98305, -0.98849, -0.99826, -1.0019, -1.0158, -1.0189, -1.014, -1.0094, -0.99988, -1.0129,
              -1.0226, -1.0282, -1.0544, -1.0633, -1.0529, -1.0471, -1.0564, -1.0761, -1.1019, -1.1137, -1.1082,
              -1.0806, -1.0598, -1.0589, -1.0409, -1.0161, -0.96924, -0.90919, -0.86976, -0.83657, -0.79945, -0.73514,
              -0.65988, -0.62568, -0.62419, -0.59193, -0.54263, -0.52214, -0.49248, -0.45815, -0.37529, -0.27192,
              -0.20179, -0.15238, -0.053538, 0.098364, 0.16873, 0.28901, 0.35378, 0.32521, 0.37096, 0.45916, 0.56032,
              0.60268, 0.63201, 0.65801, 0.74457, 0.82525, 0.86189, 0.89943, 0.86572, 0.92097, 1.0044, 1.0228, 0.95148,
              0.84451, 0.79566, 0.7365, 0.69361, 0.70052, 0.74184, 0.81824, 0.75476, 0.64439, 0.66659, 0.62743, 0.59496,
              0.65076, 0.68329, 0.65915, 0.6044, 0.59596, 0.55578, 0.49742, 0.47694, 0.44376, 0.44419, 0.40895, 0.36305,
              0.30724, 0.24913, 0.24098, 0.24536, 0.26627, 0.24837, 0.21098, 0.21622, 0.21828, 0.25447, 0.32042,
              0.36379, 0.39251, 0.41478, 0.46643, 0.52176, 0.59594, 0.65372, 0.71132, 0.71519, 0.6418, 0.64346, 0.65991,
              0.67135, 0.66234, 0.61276, 0.56657, 0.54291, 0.50065, 0.43842, 0.38458, 0.3416, 0.30987, 0.26841, 0.28265,
              0.28937, 0.28172, 0.28467, 0.25837, 0.27278, 0.32456, 0.3687, 0.4565, 0.55917, 0.61597, 0.70338, 0.7647,
              0.7648, 0.76799, 0.80993, 0.90607, 0.96013, 1.057, 1.0641, 1.0447, 1.1157, 1.0988, 1.0484, 0.94257,
              0.90492, 0.89548, 0.80692, 0.76108, 0.73732, 0.70644, 0.65272, 0.5895, 0.56895, 0.52483, 0.43619, 0.30388,
              0.17379, 0.037461, -0.088546, -0.19105, -0.27859, -0.28107, -0.27524, -0.30451, -0.29659, -0.27751,
              -0.19329, -0.15035, -0.10119, -0.050733, -0.040088, -0.066913, -0.087491, -0.081317, -0.063804, 0.03511,
              0.077251, 0.17092, 0.27262, 0.32324, 0.43022, 0.49557, 0.59898, 0.68809, 0.85104, 0.96217, 1.0774, 1.1461,
              1.2123, 1.4326, 1.5736, 1.6917, 1.7661, 1.742, 1.6948, 1.687, 1.6412, 1.5265, 1.4244, 1.3248, 1.4068,
              1.4341, 1.4815, 1.6138, 1.7697, 1.7598, 1.6162, 1.4279, 1.2653, 1.1079, 0.9962, 0.90867, 0.91963, 1.0007,
              1.0476, 1.1737, 1.3064, 1.4171, 1.3574, 1.1063, 0.79759, 0.52058, 0.31609, 0.065971, -0.17664, -0.34354,
              -0.49705, -0.70832, -0.85447, -0.95096, -0.99646, -1.0069, -0.99181, -0.97493, -0.94116, -0.90389,
              -0.92808, -0.98852, -1.0896, -1.2276, -1.4023, -1.5444, -1.6624, -1.7456, -1.8026, -1.8417, -1.8653,
              -1.8828, -1.8951, -1.9056, -1.9136, -1.9212, -1.9275, -1.9317, -1.9365, -1.9404, -1.9439, -1.9473,
              -1.9504, -1.9545, -1.9572, -1.9598, -1.9622, -1.9644, -1.9673, -1.9695, -1.9714, -1.9735, -1.9758,
              -1.9774, -1.9789]

# short data
# TS = [9, -79,-76,-73,-69,-6,-63,-61]
# ComparedTS = [18, -87,-84,-82,-81,-79,-76,-74]


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

    # print(matrix_DTW)
    return matrix_DTW[i][j]


print("---OriginDTW---")
start_time = time.time()
print('Total Distance is {0:.3f}'.format(originDTW(Trim_TS, Trim_ComparedTS)))
print('Total time is {0:.5f}'.format(time.time() - start_time))


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
            '''
            for i in matrix_ADTW:
                for j in i:
                    if j != math.inf:
                        count += 1

            print("Total {0}, we use {1}, usage {2}%".format(ts1_length * ts2_length, count,
                                                             count * 100 / (ts1_length * ts2_length)))            
            '''
            # print(matrix_ADTW)
            return matrix_ADTW[ts1_length - 1][ts2_length - 1]
        else:

            # diagonal direction
            if ts1_index + 1 < ts1_length and ts2_index + 1 < ts2_length:
                distance_dia = abs(ts1[ts1_index + 1] - ts2[ts2_index + 1])

                cumulativeDia = matrix_ADTW[ts1_index][ts2_index] + distance_dia
                if cumulativeDia < matrix_ADTW[ts1_index + 1][ts2_index + 1]:
                    matrix_ADTW[ts1_index + 1][ts2_index + 1] = cumulativeDia
                    insertList = [cumulativeDia, ts1_index + 1, ts2_index + 1]
                    heappush(comparedIncreasingQueue, insertList)

            # down direction
            if ts1_index + 1 < ts1_length:
                distance_down = abs(ts1[ts1_index + 1] - ts2[ts2_index])

                cumulativeDown = matrix_ADTW[ts1_index][ts2_index] + distance_down
                if cumulativeDown < matrix_ADTW[ts1_index + 1][ts2_index]:
                    matrix_ADTW[ts1_index + 1][ts2_index] = cumulativeDown
                    insertList = [cumulativeDown, ts1_index + 1, ts2_index]
                    heappush(comparedIncreasingQueue, insertList)

            # right direction
            if ts2_index + 1 < ts2_length:
                distance_right = abs(ts1[ts1_index] - ts2[ts2_index + 1])

                cumulativeRight = matrix_ADTW[ts1_index][ts2_index] + distance_right
                if cumulativeRight < matrix_ADTW[ts1_index][ts2_index + 1]:
                    matrix_ADTW[ts1_index][ts2_index + 1] = cumulativeRight

                    insertList = [cumulativeRight, ts1_index, ts2_index + 1]
                    heappush(comparedIncreasingQueue, insertList)

            distance_minimum = min(distance_dia, distance_down, distance_right)

            current_minimum_distance = current_minimum_distance + distance_minimum
            # sort the queue
            # comparedIncreasingQueue = sorted(comparedIncreasingQueue, key=itemgetter(0))
            minimum_cell = heappop(comparedIncreasingQueue)

            # print(comparedIncreasingQueue)
            ts1_index = minimum_cell[1]
            ts2_index = minimum_cell[2]


print("---ADTW---")
start_time = time.time()
print('Total Distance is {0:.3f}'.format(adaptiveWindowDTW(Trim_TS, Trim_ComparedTS)))
print('Total time is {0:.5f}'.format(time.time() - start_time))


def updateQueue(queue, inputValue, ts1_index, ts2_index):
    if insertList in queue:
        return queue
    else:
        queue.insert(0, insertList)
        queue = sorted(queue, key=itemgetter(0))
        return queue


'''
def quicksort(queue):
    if len(queue) <= 1:
        return queue
    midcell = queue[len(queue) // 2]
    return quicksort([i for i in queue if i < midcell]) + [midcell] + quicksort([i for i in queue if i > midcell])
'''
