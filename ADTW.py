'''
First comment

'''
import numpy as np  # including matrix
import math  # including inf
import time
import numbers

# for function sorted key
from operator import itemgetter, attrgetter

TS = [9,-0.79042,-0.76517,-0.73354,-0.69963,-0.66774,-0.63863,-0.61181,-0.58728,-0.5643,-0.54438,-0.52935,-0.5175,-0.50887,-0.50228,-0.49419,-0.48777,-0.48244,-0.47509,-0.47222,-0.46985,-0.46459,-0.46352,-0.46143,-0.45761,-0.45909,-0.46014,-0.45923,-0.45788,-0.44819,-0.43071,-0.40776,-0.3747,-0.33276,-0.2768,-0.19645,-0.090298,0.047421,0.2205,0.4261,0.67133,0.96248,1.3002,1.6832,2.0961,2.5103,2.8951,3.2194,3.4628,3.621,3.6996,3.7129,3.6775,3.6059,3.5102,3.3997,3.2801,3.1584,3.0381,2.9193,2.8006,2.676,2.5378,2.3817,2.2061,2.0162,1.8208,1.6266,1.4393,1.2597,1.0847,0.91676,0.75766,0.60808,0.47629,0.36106,0.25887,0.17309,0.096471,0.026894,-0.032236,-0.087307,-0.1369,-0.17931,-0.22133,-0.25955,-0.29338,-0.32834,-0.35884,-0.38534,-0.41342,-0.437,-0.45785,-0.48019,-0.49755,-0.51223,-0.52641,-0.536,-0.54405,-0.55164,-0.55638,-0.56113,-0.56549,-0.56759,-0.56959,-0.56966,-0.56603,-0.56026,-0.54948,-0.5322,-0.50996,-0.48015,-0.44343,-0.4024,-0.35675,-0.30783,-0.2563,-0.20004,-0.13875,-0.0727,-0.0028484,0.066369,0.13059,0.18612,0.22915,0.25895,0.2759,0.27963,0.27232,0.25625,0.23374,0.20941,0.18572,0.16234,0.13869,0.11209,0.080644,0.045686,0.0078176,-0.0317,-0.070879,-0.10997,-0.14719,-0.18032,-0.20995,-0.23491,-0.2559,-0.27537,-0.29212,-0.30676,-0.32039,-0.33221,-0.34381,-0.35484,-0.36277,-0.36673,-0.36361,-0.35116,-0.33013,-0.29944,-0.26041,-0.21724,-0.1718,-0.12815,-0.090836,-0.059735,-0.036372,-0.021981,-0.01589,-0.020298,-0.036849,-0.064752,-0.10373,-0.1509,-0.20149,-0.25293,-0.30208,-0.34683,-0.38841,-0.4262,-0.45982,-0.49066,-0.51698,-0.53882,-0.55823,-0.57462,-0.58842,-0.59955,-0.60572,-0.60698,-0.60415,-0.59771,-0.58913,-0.57675,-0.55773,-0.53128,-0.49612,-0.4541,-0.40953,-0.36356,-0.31831,-0.27612,-0.23687,-0.20282,-0.1761,-0.15669,-0.1454,-0.142,-0.14501,-0.15405,-0.16792,-0.18515,-0.20586,-0.22969,-0.25636,-0.28638,-0.31836,-0.35074,-0.38312,-0.41367,-0.44169,-0.4673,-0.48928,-0.50759,-0.52279,-0.53471,-0.54438,-0.55209,-0.55728,-0.56031,-0.56033,-0.5569,-0.55111,-0.54236,-0.53119,-0.51937,-0.50656,-0.49434,-0.48419,-0.47553,-0.46873,-0.46286,-0.45629,-0.4493,-0.44203,-0.43526,-0.43086,-0.42919,-0.43047,-0.43509,-0.44231,-0.45208,-0.46455,-0.47856,-0.49294,-0.50625,-0.51695,-0.52585,-0.5354,-0.54844,-0.5674,-0.59232,-0.62175,-0.65471,-0.69032,-0.72764,-0.76425,-0.79486,-0.81473,-0.82271,-0.82133]
#TS = [9, -79,-76,-73,-69,-6,-63,-61]
ComparedTS = [9,-0.87406,-0.8438,-0.82536,-0.81015,-0.79103,-0.7657,-0.74347,-0.72379,-0.70604,-0.68873,-0.66598,-0.64613,-0.62478,-0.60472,-0.5884,-0.57013,-0.55668,-0.53885,-0.51669,-0.48978,-0.45123,-0.40553,-0.34266,-0.26196,-0.16141,-0.034225,0.11943,0.31252,0.55019,0.83667,1.1703,1.5337,1.9083,2.2639,2.576,2.8307,3.0245,3.1704,3.2767,3.3506,3.3915,3.3948,3.3636,3.2975,3.2021,3.0815,2.9383,2.7801,2.6076,2.4265,2.24,2.0546,1.8795,1.7151,1.5634,1.4177,1.2779,1.1489,1.033,0.94175,0.87118,0.82348,0.79465,0.77478,0.7646,0.7489,0.7255,0.68867,0.63446,0.57364,0.50401,0.43453,0.36404,0.28879,0.21323,0.13161,0.050432,-0.02959,-0.1072,-0.17629,-0.24058,-0.29663,-0.34702,-0.39266,-0.4305,-0.46426,-0.49215,-0.5176,-0.54165,-0.56231,-0.58267,-0.6005,-0.61845,-0.63528,-0.64697,-0.65388,-0.6532,-0.64934,-0.64493,-0.64068,-0.63846,-0.63525,-0.63268,-0.6313,-0.6314,-0.63482,-0.63858,-0.64288,-0.6476,-0.65259,-0.66201,-0.67396,-0.68819,-0.70277,-0.7145,-0.72688,-0.73952,-0.75455,-0.77121,-0.78551,-0.79871,-0.80931,-0.81965,-0.82943,-0.83431,-0.83168,-0.81509,-0.78144,-0.7266,-0.64691,-0.54387,-0.42064,-0.28784,-0.15664,-0.037201,0.061686,0.13826,0.19094,0.22269,0.23828,0.24162,0.24039,0.23611,0.22968,0.21826,0.19497,0.1599,0.11143,0.055447,-0.0013555,-0.05622,-0.10257,-0.14158,-0.17017,-0.18705,-0.19381,-0.18836,-0.17584,-0.15557,-0.12966,-0.098455,-0.057909,-0.0087795,0.05542,0.13428,0.22713,0.33134,0.43752,0.54156,0.63498,0.71474,0.78037,0.83038,0.86831,0.89179,0.90006,0.89145,0.86316,0.81806,0.75658,0.68283,0.60116,0.5145,0.42885,0.34462,0.26288,0.18242,0.10151,0.02328,-0.052236,-0.12156,-0.18451,-0.24135,-0.28965,-0.33165,-0.3642,-0.38835,-0.4034,-0.40704,-0.40356,-0.39212,-0.37771,-0.36168,-0.34318,-0.32698,-0.3125,-0.30492,-0.30474,-0.30919,-0.31885,-0.33015,-0.34589,-0.36645,-0.39007,-0.41628,-0.44053,-0.46383,-0.48724,-0.51192,-0.53924,-0.56551,-0.58891,-0.60718,-0.62008,-0.62964,-0.6356,-0.63921,-0.63996,-0.63834,-0.63633,-0.63398,-0.63348,-0.63464,-0.63806,-0.6447,-0.65323,-0.66399,-0.67479,-0.68453,-0.693,-0.69984,-0.70725,-0.71577,-0.72628,-0.73826,-0.75012,-0.76141,-0.77066,-0.7776,-0.78168,-0.78233,-0.78036,-0.77621,-0.77191,-0.769,-0.76804,-0.76914,-0.77092,-0.77414,-0.78048,-0.79167,-0.80831,-0.82832,-0.85112,-0.87714,-0.9065,-0.93625,-0.95921,-0.97052,-0.97086]
#ComparedTS = [18, -87,-84,-82,-81,-79,-76,-74]
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
        if len(comparedIncreasingQueue) != 0 and matrix_ADTW[ts1_length - 1][ts2_length - 1] < \
                comparedIncreasingQueue[0][0]:
            print(matrix_ADTW)
            return matrix_ADTW[ts1_length - 1][ts2_length - 1]

        else:
            if comparedIncreasingQueue:
                comparedIncreasingQueue.pop(0)
            if ts1_index + 1 < ts1_length and ts2_index + 1 < ts2_length:
                distance_dia = abs(ts1[ts1_index + 1] - ts2[ts2_index + 1])
                if current_minimum_distance + distance_dia < matrix_ADTW[ts1_index + 1][ts2_index + 1]:
                    matrix_ADTW[ts1_index + 1][ts2_index + 1] = matrix_ADTW[ts1_index][ts2_index] + distance_dia
                    comparedIncreasingQueue = updateQueue(comparedIncreasingQueue,
                                                          matrix_ADTW[ts1_index][ts2_index] + distance_dia,
                                                          ts1_index + 1, ts2_index + 1)
            if ts1_index + 1 < ts1_length :
                distance_down = abs(ts1[ts1_index + 1] - ts2[ts2_index])
                if current_minimum_distance + distance_down < matrix_ADTW[ts1_index + 1][ts2_index]:
                    matrix_ADTW[ts1_index + 1][ts2_index] = matrix_ADTW[ts1_index][ts2_index] + distance_down
                    comparedIncreasingQueue = updateQueue(comparedIncreasingQueue,
                                                          matrix_ADTW[ts1_index][ts2_index] + distance_down,
                                                          ts1_index + 1, ts2_index)
            if ts2_index + 1 < ts2_length :
                distance_right = abs(ts1[ts1_index] - ts2[ts2_index + 1])
                if current_minimum_distance + distance_right < matrix_ADTW[ts1_index][ts2_index + 1]:
                    matrix_ADTW[ts1_index][ts2_index + 1] = matrix_ADTW[ts1_index][ts2_index] + distance_right
                    comparedIncreasingQueue = updateQueue(comparedIncreasingQueue,
                                                          matrix_ADTW[ts1_index][ts2_index] + distance_right, ts1_index,
                                                          ts2_index + 1)
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


start_time = time.time()
print(adaptiveWindowDTW(Trim_TS, Trim_ComparedTS))
print(time.time() - start_time)
