'''
First comment

'''
import numpy as np # including matrix
import math # including inf
TS =[9,3,8,7,4,1,3,2,1,7]
ComparedTS=[18,2,9,8,8,5,4,2,1,5]

# Trim the label of the time series
Trim_TS=TS[1:]
Trim_ComparedTS=ComparedTS[1:]

#Origin DTW

def originDTW(ts1, ts2):
    matrix_DTW = np.array([[math.inf for i in range(len(ts1) + 1)] for i in range(len(ts2) + 1)])
    matrix_DTW[0][0] = 0

    for i in range(1, len(ts1)+1):
        for j in range(1, len(ts2)+1):
            distance = abs(ts1[i-1] - ts2[j-1])
            matrix_DTW[i][j] = distance + min(matrix_DTW[i-1][j], matrix_DTW[i][j-1], matrix_DTW[i-1, j-1])
    return matrix_DTW[i][j]

print(originDTW(Trim_TS, Trim_ComparedTS))


