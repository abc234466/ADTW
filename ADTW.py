'''
First comment
'''

# import method
import numpy as np  # including matrix
import math  # including inf
import time  # calculate the execution time
import os  # list data of directory
import random # use random choice

from heapq import *  # include heap queue data structure
from sortedcontainers import *
from collections import *

'''
TS = [4, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094,
      0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094,
      0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094,
      0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57094, 0.57122, 0.57528,
      0.59379, 0.6123, 0.62808, 0.63162, 0.63278, 0.63458, 0.64561, 0.66153, 0.67902, 0.69045, 0.69462, 0.69501,
      0.69816, 0.71077, 0.72928, 0.74779, 0.7663, 0.78482, 0.80333, 0.82184, 0.84035, 0.85887, 0.87738, 0.89589, 0.9144,
      0.93291, 0.95143, 0.96994, 0.98845, 1.007, 1.0255, 1.044, 1.0625, 1.081, 1.0995, 1.118, 1.1243, 1.1275, 1.1275,
      1.1357, 1.1488, 1.1673, 1.1814, 1.1893, 1.1893, 1.1893, 1.1893, 1.1893, 1.1893, 1.1952, 1.2106, 1.2287, 1.2431,
      1.2501, 1.2512, 1.2512, 1.2512, 1.2512, 1.2523, 1.2559, 1.2716, 1.2902, 1.3087, 1.2983, 1.2803, 1.2618, 1.2543,
      1.2512, 1.2512, 1.2356, 1.211, 1.174, 1.1416, 1.1137, 1.0952, 1.075, 1.0507, 1.0137, 0.97663, 0.94287, 0.92278,
      0.90377, 0.88751, 0.88199, 0.88013, 0.88292, 0.90104, 0.93134, 0.96836, 1.0054, 1.0424, 1.0794, 1.1165, 1.1535,
      1.1905, 1.2275, 1.2553, 1.2764, 1.2949, 1.3067, 1.313, 1.313, 1.2914, 1.25, 1.1574, 1.0753, 1.0038, 1.0038,
      1.0038, 1.0038, 0.94828, 0.89274, 0.8372, 0.76737, 0.69541, 0.62136, 0.53708, 0.44884, 0.35628, 0.27003, 0.18928,
      0.11523, 0.17029, -0.19995, -0.10692, -0.18097, -0.24818, -0.3049, -0.36057, -0.41611, -0.47165, -0.52718,
      -0.57599, -0.61262, -0.63392, -0.65429, -0.67832, -0.71309, -0.75012, -0.78714, -0.82417, -0.86119, -0.89821,
      -0.93524, -0.97226, -1.0093, -1.0463, -1.0833, -1.1204, -1.1636, -1.2113, -1.2669, -1.3161, -1.3567, -1.3752,
      -1.3934, -1.4079, -1.4079, -1.4079, -1.4079, -1.4079, -1.4079, -1.4061, -1.3951, -1.3791, -1.3606, -1.3421,
      -1.3236, -1.3055, -1.2901, -1.2842, -1.2842, -1.2842, -1.2948, -1.3126, -1.3311, -1.3496, -1.3681, -1.3866,
      -1.4051, -1.4236, -1.4422, -1.4607, -1.4792, -1.4977, -1.5142, -1.5284, -1.5099, -1.4914, -1.4729, -1.4544,
      -1.4359, -1.4173, -1.4111, -1.4079, -1.4079, -1.3996, -1.3866, -1.3681, -1.354, -1.3461, -1.3461, -1.3453,
      -1.3374, -1.3189, -1.3004, -1.2818, -1.2633, -1.2448, -1.2305, -1.2234, -1.2224, -1.2172, -1.2022, -1.1688,
      -1.1341, -1.1042, -1.0987, -1.0987, -1.0987, -1.0987, -1.0987, -1.0987, -1.0987, -1.0987, -1.0987, -1.0909,
      -1.0786, -1.0601, -1.0463, -1.0369, -1.0369, -1.0334, -1.0219, -0.98487, -0.94784, -0.91082, -0.87379, -0.83677,
      -0.80199, -0.77797, -0.7576, -0.73769, -0.71012, -0.67646, -0.64013, -0.60822, -0.58469, -0.56618, -0.54766,
      -0.52915, -0.51064, -0.49213, -0.47362, -0.4551, -0.43659, -0.41808, -0.39957, -0.38106, -0.36686, -0.35663,
      -0.35663, -0.35455, -0.35033, -0.33182, -0.31331, -0.2948, -0.23609, -0.23609, -0.23609, -0.23609, -0.27635,
      -0.32262, -0.37476, -0.39811, -0.41027, -0.41027, -0.41027, -0.41027, -0.41027, -0.41027, -0.41027, -0.41027,
      -0.41027, -0.41027, -0.41027, -0.41027, -0.41027, -0.41027, -0.41027, -0.41027, -0.41027, -0.41027, -0.41027,
      -0.41027, -0.41027, -0.41027, -0.41027, -0.41027, -0.41027, -0.41027, -0.41027, -0.41027, -0.41027, -0.43672,
      -0.47572, -0.52786, -0.56253, -0.58444, -0.58444, -0.57552, -0.55449, -0.50235, -0.45178, -0.42247, -0.47461,
      -0.52675, -0.57119, -0.58118, -0.58444, -0.58444, -0.58444, -0.58444, -0.58444, -0.58444, -0.58444, -0.58333,
      -0.57446, -0.53896, -0.48682, -0.43467, -0.41229, -0.41027, -0.41027, -0.38786, -0.34814, -0.296, -0.24386,
      -0.19171, -0.13957, -0.09639, -0.34517, -0.34517, -0.34517, -0.34517, -0.34517, -0.34517, -0.34517, -0.11443,
      0.13187, 0.39258, 0.48091, 0.11226, 0.11226, 0.1355, 0.17217, 0.22431, 0.26404, 0.28644, 0.28644, 0.28847,
      0.31085, 0.36299, 0.41513, 0.46727, 0.51942, 0.57156, 0.60019, 0.58799, 0.5416, 0.50405, 0.49407, 0.53606,
      0.58493, 0.62709, 0.63479, 0.63479, 0.63479, 0.63479, 0.63479, 0.63479, 0.63479, 0.63479, 0.63479, 0.63479,
      0.63479, 0.63479, 0.63479, 0.63479, 0.63479, 0.62999, 0.61371, 0.56157, 0.50943, 0.47564, 0.5189, 0.56823,
      0.62037, 0.67251, 0.72466, 0.7768, 0.82894, 0.88108, 0.93517, 1.0017, 1.0919, 1.1958, 1.2967, 1.3681, 1.4203,
      1.4724, 1.5245, 1.5767, 1.6288, 1.681, 1.7331, 1.7852, 1.8374, 1.8895, 1.9417, 1.9879, 2.0282, 2.0282, 2.0282,
      2.0282, 2.0803, 2.1325, 2.1846, 2.1965, 2.2024, 2.2024, 2.2024, 2.2024, 2.2024, 2.1846, 2.1514, 2.0992, 2.0399,
      1.9616, 1.8574, 1.7531, 1.668, 1.6126, 1.56, 1.522, 1.5076, 1.5057, 1.5057, 1.5057, 1.5057, 1.5057, 1.5057,
      1.5057, 1.5043, 1.4998, 1.4569, 1.4047, 1.3526, 1.3004, 1.2483, 1.1962, 1.1176, 1.0264, 0.92213, 0.80037, 0.66586,
      0.50943, 0.36192, 0.22653, 0.12225, 0.33731, -0.40619, -0.12626, -0.1784, -0.22284, -0.23282, -0.23609, -0.23101,
      -0.19995, -0.1551, -0.10584, -0.40395, -0.34517, -0.35072, -0.39509, -0.1074, -0.15954, -0.21168, -0.26382,
      -0.31597, -0.36811, -0.42025, -0.47239, -0.52453, -0.59214, -0.6732, -0.77748, -0.8728, -0.95942, -1.0116,
      -1.0608, -1.107, -1.107, -1.107, -1.107, -1.107, -1.107, -1.107, -1.1414, -1.1846, -1.2368, -1.2657, -1.2811,
      -1.2811, -1.2687, -1.239, -1.1869, -1.1367, -1.107, -1.107, -1.107, -1.107, -1.107, -1.107, -1.107, -1.107,
      -1.107, -1.1143, -1.1353, -1.1824, -1.2346, -1.2867, -1.3388, -1.391, -1.4431, -1.4545, -1.4553, -1.4553, -1.4553,
      -1.4553, -1.4553, -1.4553, -1.4553, -1.4553, -1.4422, -1.4165, -1.3644, -1.317, -1.2811, -1.2811, -1.2811,
      -1.2628, -1.1674, -1.0659, -0.96798, -0.9003, -0.84293, -0.79472, -0.76809, -0.75862, -0.75862, -0.75862,
      -0.75862, -0.75862, -0.75862, -0.75862, -0.75862, -0.75862, -0.73259, -0.68762, -0.63548, -0.60221, -0.58444,
      -0.58444, -0.58444, -0.58444, -0.58444, -0.58444, -0.58444, -0.58444, -0.58444, -0.58444, -0.7514, -0.7514,
      -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514,
      -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514,
      -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7514,
      -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.77007, -0.8141, -0.92324, -1.0291, -1.0904, -0.98129, -0.87215,
      -0.77913, -0.75824, -0.7514, -0.74078, -0.67576, -0.58189, -0.47878, -0.41144, -0.38684, -0.38451, -0.36594,
      -0.29163, -0.18249, -0.40239, -0.16817, -0.14696, -0.14696, -0.14696, -0.14696, -0.14696, -0.14696, -0.14696,
      -0.14696, -0.053182, 0.13169, 0.1426, 0.23968, 0.32372, 0.21458, 0.49161, -0.054075, -0.11682, -0.14696, -0.14696,
      -0.14696, -0.14696, -0.14696, -0.14696, -0.14696, -0.14696, -0.16385, 0.29424, 0.17511, 0.28, 0.3423, 0.3423,
      0.3423, 0.37713, 0.46769, 0.5745, 0.65904, 0.70083, 0.70686, 0.70686, 0.70686, 0.70686, 0.70686, 0.70686, 0.70686,
      0.70686, 0.70686, 0.62162, 0.51413, 0.40499, 0.36096, 0.3423, 0.3423, 0.29644, 0.22387, 0.11473, 0.1299, -0.14696,
      -0.14696, -0.14696, -0.14696, -0.14696, -0.14696, -0.24297, -0.14132, -0.24751, -0.35665, -0.46579, -0.57493,
      -0.69228, -0.85483, -1.0533, -1.2715, -1.4898, -1.7081, -1.9256, -2.1369, -2.2863, -2.3954, -2.5046, -2.5048,
      -2.4256, -2.3165, -2.1678, -1.9868, -1.7685, -1.5247, -1.2576, -0.9302, -0.63963, -0.38684, -0.38684, -0.38684,
      -0.38684, -0.2777, -0.16856, -0.33273, 0.21296, 0.15885, 0.26799, 0.37713, 0.48627, 0.5954, 0.74174, 0.92049,
      1.1388, 1.342, 1.5057, 1.6148, 1.7239, 1.8331, 1.9422, 2.0513, 2.1013, 2.0525, 1.9515, 1.8424, 1.7332, 1.6241,
      1.5259, 1.4493, 1.436, 1.436, 1.436, 1.436, 1.436, 1.436, 1.5861, 1.7843, 2.0026, 2.1655, 2.3021, 2.4113, 2.4838,
      2.5297, 2.5297, 2.5297, 2.5297, 2.5297, 2.528, 2.5042, 2.395, 2.2859, 2.1606, 1.9632, 1.7518, 1.5442, 1.3909,
      1.2665, 1.1573, 1.0482, 0.93907, 0.83225, 0.74169, 0.70686, 0.70686, 0.70686, 0.70686, 0.70686, 0.70686, 0.70686,
      0.70686, 0.70686, 0.70686, 0.70686, 0.70686, 0.68811, 0.65113, 0.542, 0.43889, 0.3423, 0.3423, 0.3423, 0.3423,
      0.24571, 0.1426, 0.13169, -0.053182, -0.14696, -0.14696, -0.14696, -0.14696, -0.14696, -0.14696, -0.14696,
      -0.14696, -0.16817, -0.40239, -0.18249, -0.29163, -0.36594, -0.38451, -0.38684, -0.41144, -0.47878, -0.58189,
      -0.69103, -0.80017, -0.9093, -1.0116, -1.0999, -1.116, -1.116, -1.116, -1.0307, -0.92324, -0.8141, -0.77007,
      -0.7514, -0.7514, -0.7514, -0.7514, -0.7514, -0.7239, -0.67013, -0.56099, -0.4619, -0.38684, -0.38684, -0.38684,
      -0.38684, -0.38684, -0.38684, -0.38684, -0.38684, -0.38684, -0.37862, -0.32521, -0.2359, -0.13083, -0.29499,
      -0.14696, -0.14696, -0.14696, -0.14696, -0.14696, -0.14696, -0.41936, -0.17088, -0.28002, -0.34964, -0.38684,
      -0.38684, -0.38684, -0.38684, -0.38684, -0.38684, -0.38684, -0.38684, -0.38684, -0.38684]

ComparedTS = [4,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.9837,0.98191,0.97774,0.97308,0.96171,0.95033,0.93895,0.92757,0.9162,0.90482,0.89344,0.88206,0.87069,0.84937,0.82206,0.79446,0.76601,0.73757,0.70913,0.68068,0.65224,0.6238,0.59125,0.54005,0.48885,0.419,0.34505,0.2711,0.19715,0.12319,0.24572,-0.12404,-0.48961,-0.17064,-0.23747,-0.29436,-0.35124,-0.40813,-0.46502,-0.5219,-0.57879,-0.63568,-0.69256,-0.74945,-0.7929,-0.83538,-0.87309,-0.90722,-0.94135,-0.97548,-1.0096,-1.0437,-1.0779,-1.1098,-1.1394,-1.1666,-1.178,-1.1894,-1.2008,-1.2121,-1.2235,-1.2349,-1.2463,-1.2576,-1.269,-1.2714,-1.2697,-1.2667,-1.2611,-1.2554,-1.2497,-1.244,-1.2383,-1.2326,-1.2251,-1.2118,-1.1986,-1.1719,-1.1435,-1.115,-1.0866,-1.0582,-1.0297,-1.0013,-0.97283,-0.94438,-0.92264,-0.90955,-0.8966,-0.88522,-0.87384,-0.86247,-0.85109,-0.83971,-0.82833,-0.81696,-0.81085,-0.80478,-0.80255,-0.80255,-0.80255,-0.80255,-0.80255,-0.80255,-0.80255,-0.80275,-0.80313,-0.80558,-0.81696,-0.82833,-0.83971,-0.85109,-0.86247,-0.87384,-0.88522,-0.8966,-0.90798,-0.92189,-0.93669,-0.95235,-0.96941,-0.98648,-1.0035,-1.0206,-1.0377,-1.0547,-1.0715,-1.0876,-1.1038,-1.1154,-1.1268,-1.1382,-1.1496,-1.1609,-1.1723,-1.1837,-1.1951,-1.2064,-1.2158,-1.223,-1.23,-1.2356,-1.2413,-1.247,-1.2527,-1.2584,-1.2641,-1.2696,-1.2719,-1.2741,-1.2709,-1.2652,-1.2595,-1.2538,-1.2482,-1.2425,-1.2368,-1.2311,-1.2254,-1.2209,-1.2202,-1.2194,-1.2193,-1.2193,-1.2193,-1.2193,-1.2193,-1.2193,-1.2193,-1.2154,-1.2104,-1.2027,-1.1913,-1.1799,-1.1685,-1.1571,-1.1458,-1.1344,-1.123,-1.1116,-1.1003,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70322,0.70467,0.70623,0.7157,0.7313,0.7469,0.76249,0.77809,0.79369,0.80929,0.82488,0.84048,0.85483,0.86003,0.86523,0.86648,0.86648,0.86648,0.86648,0.86648,0.86648,0.86648,0.87017,0.87563,0.88675,0.91015,0.93354,0.95694,0.98034,1.0037,1.0271,1.0505,1.0739,1.0973,1.1044,1.1091,1.1114,1.1114,1.1114,1.1114,1.1114,1.1114,1.1114,1.115,1.1236,1.1332,1.1566,1.18,1.2034,1.2268,1.2502,1.2736,1.297,1.3204,1.3438,1.3535,1.3551,1.3562,1.3562,1.3562,1.3562,1.3562,1.3562,1.3562,1.3513,1.324,1.2967,1.2471,1.1925,1.1379,1.0833,1.0287,0.9741,0.91951,0.86453,0.80942,0.75158,0.68919,0.6268,0.56441,0.50202,0.43963,0.37724,0.31485,0.25246,0.19007,0.12768,0.32599,0.092949,-0.2979,-0.12187,-0.18426,-0.24665,-0.30904,-0.37143,-0.43382,-0.49621,-0.5586,-0.62099,-0.68338,-0.74577,-0.80816,-0.87055,-0.93294,-0.99532,-1.0577,-1.1201,-1.1498,-1.1643,-1.1742,-1.1742,-1.1742,-1.1742,-1.1742,-1.1742,-1.1742,-1.1729,-1.1677,-1.1625,-1.1482,-1.1326,-1.117,-1.1014,-1.0858,-1.0702,-1.0546,-1.039,-1.0234,-1.0078,-0.99221,-0.97661,-0.96101,-0.94541,-0.92982,-0.91422,-0.89862,-0.88302,-0.86743,-0.85905,-0.85074,-0.84767,-0.84767,-0.84767,-0.84767,-0.84767,-0.84767,-0.84767,-0.84781,-0.84807,-0.84975,-0.85755,-0.86535,-0.87315,-0.88094,-0.88874,-0.89654,-0.90434,-0.91214,-0.91994,-0.9347,-0.95186,-0.97141,-0.99481,-1.0182,-1.0416,-1.065,-1.0884,-1.1118,-1.1348,-1.1569,-1.179,-1.195,-1.2106,-1.2262,-1.2418,-1.2574,-1.273,-1.2886,-1.3042,-1.3198,-1.3353,-1.3509,-1.3665,-1.3821,-1.3977,-1.4133,-1.4289,-1.4445,-1.4601,-1.4755,-1.4864,-1.4973,-1.5007,-1.5007,-1.5007,-1.5007,-1.5007,-1.5007,-1.5007,-1.5007,-1.5007,-1.5007,-1.5007,-1.5007,-1.5007,-1.5007,-1.5007,-1.5007,-1.5007,-1.5007,-1.5007,-1.498,-1.4946,-1.4892,-1.4814,-1.4736,-1.4658,-1.458,-1.4502,-1.4424,-1.4346,-1.4269,-1.4191,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1789,1.1775,1.176,1.1668,1.1517,1.1366,1.1214,1.1063,1.0912,1.076,1.0609,1.0458,1.0319,1.0268,1.0218,1.0206,1.0206,1.0206,1.0206,1.0206,1.0206,1.0206,1.0134,1.0028,0.98123,0.93584,0.89045,0.84506,0.79967,0.75428,0.70889,0.6635,0.61811,0.57272,0.559,0.54992,0.54548,0.54548,0.54548,0.54548,0.54548,0.54548,0.54548,0.52646,0.48208,0.43251,0.31147,0.19043,0.34648,-0.25872,-0.17269,-0.29373,-0.41477,-0.53581,-0.65685,-0.76908,-0.876,-0.98265,-1.0886,-1.1945,-1.3004,-1.4063,-1.5122,-1.6181,-1.7131,-1.7585,-1.8039,-1.7997,-1.7845,-1.7694,-1.7543,-1.7391,-1.724,-1.7089,-1.6923,-1.6751,-1.6474,-1.602,-1.5566,-1.5112,-1.4658,-1.4204,-1.375,-1.3296,-1.2842,-1.2389,-1.2113,-1.1851,-1.1652,-1.1501,-1.135,-1.1198,-1.1047,-1.0896,-1.0744,-1.0578,-1.0397,-1.02,-0.98971,-0.95945,-0.92919,-0.89893,-0.86867,-0.83841,-0.80815,-0.77789,-0.74763,-0.71737,-0.68711,-0.65685,-0.62659,-0.59633,-0.56607,-0.53581,-0.50555,-0.47529,-0.44988,-0.4398,-0.42971,-0.45512,-0.48538,-0.51564,-0.5459,-0.57616,-0.60642,-0.63668,-0.66694,-0.6972,-0.72152,-0.73816,-0.75469,-0.76982,-0.78495,-0.80008,-0.81521,-0.83034,-0.84547,-0.8606,-0.86172,-0.86272,-0.85354,-0.83841,-0.82328,-0.80815,-0.79302,-0.77789,-0.76276,-0.74791,-0.73328,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.72065,-0.71812,-0.7156,-0.70123,-0.6861,-0.67097,-0.65584,-0.64071,-0.62558,-0.61045,-0.59532,-0.58019,-0.57607,-0.58313,-0.59129,-0.60642,-0.62155,-0.63668,-0.65181,-0.66694,-0.68207,-0.69696,-0.70755,-0.71814,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.7214,-0.71613,-0.70957,-0.69921,-0.68408,-0.66895,-0.65382,-0.63869,-0.62356,-0.60843,-0.5933,-0.57817,-0.56304]

# short data
# TS = [9, -79,-76,-73,-69,-6,-63,-61]
# ComparedTS = [18, -87,-84,-82,-81,-79,-76,-74]


# Trim the label of the time series
Trim_TS = TS[1:]
Trim_ComparedTS = ComparedTS[1:]
'''

# global matrix_ADTW matrix
matrix_ADTW = []

# usage of cells

count = 0


# ----- Origin DTW -----

def originDTW(ts1, ts2):
    # initialization of matrix
    matrix_DTW = np.array([[math.inf for i in range(len(ts1) + 1)] for i in range(len(ts2) + 1)])

    # set the value of origin
    matrix_DTW[0][0] = 0

    # Dynamic Programming -> Divide and Conquer & Memorization
    for i in range(1, len(ts1) + 1):
        for j in range(1, len(ts2) + 1):
            distance = abs(ts1[i - 1] - ts2[j - 1])
            matrix_DTW[i][j] = distance + min(matrix_DTW[i - 1][j], matrix_DTW[i][j - 1], matrix_DTW[i - 1, j - 1])
    # print(matrix_DTW)
    return matrix_DTW[i][j]


# ----- MyDTW -----
def adaptiveWindowDTW(ts1, ts2, cur_mini_distance):
    # initialization of matrix
    # global matrix_ADTW
    matrix_ADTW = [[math.inf for i in range(len(ts1))] for i in range(len(ts2))]

    # two indices of time series
    ts1_index = 0
    ts2_index = 0

    # set the value of origin point [0, 0]
    matrix_ADTW[0][0] = abs(ts1[0] - ts2[0])
    # current_minimum_distance = matrix_ADTW[0][0]

    # two lengths of time series
    ts1_length = len(ts1)  # length of m
    ts2_length = len(ts2)  # length of n

    # unexpanded cells queuing for expansion
    unexpandedCellQueue = [[math.inf, math.inf, math.inf]]
    minimum_value = -1
    while cur_mini_distance > minimum_value:
        templist=[]
        # global totalUsedCells
        # count = 0
        # if #something happen:
        # diagonal direction
        if ts1_index + 1 < ts1_length and ts2_index + 1 < ts2_length:
            # accumulate current distance with diagonal distance
            distance_dia = abs(ts1[ts1_index + 1] - ts2[ts2_index + 1])
            cumulativeDia = matrix_ADTW[ts1_index][ts2_index] + distance_dia

            # check whether the cell should be add into unexpandedCellQueue
            if matrix_ADTW[ts1_index + 1][ts2_index + 1] == math.inf:
                matrix_ADTW[ts1_index + 1][ts2_index + 1] = cumulativeDia
                insertList = [cumulativeDia, ts1_index + 1, ts2_index + 1]
                templist.append(insertList)
                # heappush(unexpandedCellQueue, insertList)
                # totalUsedCells += 1

        # down direction
        if ts1_index + 1 < ts1_length:
            # accumulate current distance with down distance
            distance_down = abs(ts1[ts1_index + 1] - ts2[ts2_index])
            cumulativeDown = matrix_ADTW[ts1_index][ts2_index] + distance_down

            # check whether the cell should be add into unexpandedCellQueue
            if matrix_ADTW[ts1_index + 1][ts2_index] == math.inf:
                matrix_ADTW[ts1_index + 1][ts2_index] = cumulativeDown
                insertList = [cumulativeDown, ts1_index + 1, ts2_index]
                templist.append(insertList)
                heappush(unexpandedCellQueue, insertList)
                # totalUsedCells += 1

        # right direction
        if ts2_index + 1 < ts2_length:
            # accumulate current distance with right distance
            distance_right = abs(ts1[ts1_index] - ts2[ts2_index + 1])
            cumulativeRight = matrix_ADTW[ts1_index][ts2_index] + distance_right

            # check whether the cell should be add into unexpandedCellQueue
            if matrix_ADTW[ts1_index][ts2_index + 1] == math.inf:
                matrix_ADTW[ts1_index][ts2_index + 1] = cumulativeRight
                insertList = [cumulativeRight, ts1_index, ts2_index + 1]
                heappush(unexpandedCellQueue, insertList)
                # totalUsedCells += 1

        # find the current minimum cell
        minimum_cell = heappop(unexpandedCellQueue)
        # replace the minimum value with new heappop value
        minimum_value = minimum_cell[0]
        # replace the ts1_index & ts2_index with the value of current minimum_cell
        ts1_index = minimum_cell[1]
        ts2_index = minimum_cell[2]

        # check whether surpass current allowed distance

        if matrix_ADTW[ts1_length - 1][ts2_length - 1] <= minimum_value:
            '''
            count = 0

            for i in matrix_ADTW:
                for j in i:
                    if j != math.inf:
                        count += 1

            print("Total {0}, we use {1}, usage {2}%".format(ts1_length * ts2_length, count,
                                                             count * 100 / (ts1_length * ts2_length)))            
            '''
            return matrix_ADTW[ts1_length - 1][ts2_length - 1]

            # worst case -> fill all cells
            # if len(unexpandedCellQueue) == 1:
            # return matrix_ADTW[ts1_length - 1][ts2_length - 1]


'''


main program


'''

pre_dir = 'UCR_TS_Archive_2015'
# pre_dir = 'TEST'

### experiment all data
for file in os.listdir(pre_dir):
    print('Dataset {} begins.'.format(file))
    # read TEST & TRAIN data set
    test_data_dir = pre_dir + '/' + file + '/' + file + '_TEST'
    train_data_dir = pre_dir + '/' + file + '/' + file + '_TRAIN'

    # open test & train data
    with open(test_data_dir) as test_data, open(train_data_dir) as train_data:

        # two variables -> total time for DTW & ADTW
        all_dtw_total_time = 0
        all_adtw_total_time = 0
        # set up the matched & unmatched points
        matched_point = 0
        unmatched_point = 0

        # totalUsedCells = 0

        # initial the train data list -> preprocess the train data
        processed_train_data = []

        while True:
            # read whole line data -> transform into data we want
            train_data_one_line = train_data.readline().split(',')
            # print(train_data_one_line)
            if train_data_one_line == ['']:
                break
            else:
                processed_train_data.append(train_data_one_line)

        # store the total experiment
        totalExperiment = 0

        # i = 0
        ### test data while loop
        while True:

            '''
            initialization -> read one line test data -> adjust the data < string to list >
            e.g. 
                '1, 2, 3, 4, 5' split(',') -> ['1', '2', '3', '4', '5'] -> [1, 2, 3, 4, 5]
            '''

            test_data_one_line = test_data.readline().split(',')
            # print(test_data_one_line)
            # print(test_data_one_line)
            # store the total experimental rounds


            # end of data
            if test_data_one_line == ['']:
                # print('Matched number is {0}, and unmatched number is {1}.'.format(matched_point, unmatched_point))
                # print('Total time of DTW is {0:.3f} '.format(all_dtw_total_time))
                print('Total time of ADTW of {0} is {1:.3f}'.format(file, all_adtw_total_time))
                # print('Accuracy is {0:.3f}%.'.format(matched_point * 100 / (matched_point + unmatched_point)))
                # totalUsedCells += totalExperiment
                # totalCells = len(train_data_list) * len(train_data_list) * totalExperiment
                # print("Total {0}, we use {1}, usage {2}%".format(totalCells, totalUsedCells,
                #                                                  totalUsedCells * 100 / (
                #                                                      totalCells)))
                print('---End of Test---')
                break
            else:

                '''
                    adjust the data :
                    ['1', '2', '3', '4', '5'] -> [1, 2, 3, 4, 5]
                '''
                test_data_list = []
                for value in test_data_one_line[1:]:
                    test_data_list.append(float(value))

                # get the 'test' data label
                test_label = test_data_one_line[0]

                # store the current minimum distance -> in order to know who is the closet class
                current_minimum_distance = math.inf

                # scores between DTW and ADTW
                adtwpoint = 0
                dtwpoint = 0

                # set up a current minimum distance to infinity
                current_comparision_distance = math.inf

                train_data_index = 0

                # store the backup processed train data
                processed_train_data_backup = processed_train_data[:]

            ### train data while loop
            while True:

                '''
                initialization -> read one line test data -> adjust the data < string to list >
                e.g. 
                    '1, 2, 3, 4, 5' split(',') -> ['1', '2', '3', '4', '5']
                '''
                # end of current test data -> begin next test data
                if len(processed_train_data_backup) == 0:
                    # i += 1
                    # print the information of compared time series
                    # print("The most similar class is {0}".format(train_label))
                    #
                    # # Accuracy of Test Set
                    # if test_label == train_label:
                    #     matched_point += 1
                    # else:
                    #     unmatched_point += 1
                    #
                    # # comparison of DTW & ADTW
                    # print("DTW wins {0}, ADTW wins {1}".format(dtwpoint, adtwpoint))
                    break
                else:

                    # accumulate the total experiments round
                    # totalExperiment += 1

                    # print('Train DataSet TEST : {0}, TRAIN : {1} '.format(i, train_data_index))
                    # split_train_data_one_line = train_data_one_line

                    # random choice the train data -> speed up the execution
                    train_data_this_line = random.choice(processed_train_data_backup)
                    processed_train_data_backup.remove(train_data_this_line)
                    '''
                        process the data :
                        ['1', '2', '3', '4', '5'] -> [1, 2, 3, 4, 5]
                    '''
                    train_data_list = []
                    for value in train_data_this_line[1:]:
                        train_data_list.append(float(value))

                    # print(train_data_list, end='')
                    # print("---Origin DTW---")

                    # calculate the execution time
                    # dtw_start_time = time.time()
                    # dtw_distance = originDTW(test_data_list, train_data_list)
                    # dtw_total_time = time.time() - dtw_start_time
                    #
                    # all_dtw_total_time += dtw_total_time

                    # print('Total Distance of DTW is {0:.3f}'.format(dtw_distance))
                    # print('Total time is {0:.5f}'.format(dtw_total_time))

                    # print("---ADTW---")
                    # calculate the execution time

                    # beginning of ADTW
                    adtw_start_time = time.time()

                    '''
                    function adaptiveWindowDTW(A, B, C) ->
                    A : current test data <list>
                    B : current train data <list>
                    C : current allowed maximum distance
                    '''
                    adtw_distance = adaptiveWindowDTW(test_data_list, train_data_list, current_comparision_distance)
                    adtw_total_time = time.time() - adtw_start_time
                    # end of ADTW

                    # total ADTW execution time
                    all_adtw_total_time += adtw_total_time

                    # adtw_distance != None -> We arrive at the cell matrix_ADTW[m][n]
                    if adtw_distance != None:

                        # update the current allowed maximum distance
                        if adtw_distance < current_comparision_distance:
                            current_comparision_distance = adtw_distance

                        # 1 nearest neighbor (1NN) -> determine the most similar class
                        if adtw_distance < current_minimum_distance:
                            # if we find another smaller value -> update adtw_distance
                            current_minimum_distance = adtw_distance
                            train_label = train_data_this_line[0]
                            # print('Total Distance of ADTW is {0:.3f}'.format(adtw_distance))

                    # adtw_distance == None -> We can't find the better answer, so we just move on to next training data and compare it.
                    else:
                        pass
                        # print('We do not need to complete this round.')

                        # #print(test_label, train_label)
                        # # show the ADTW total time
                        # #print('Total time is {0:.5f}'.format(adtw_total_time))
                        #
                        # if adtw_total_time < dtw_total_time:
                        #     #print(i)
                        #     adtwpoint += 1
                        # else:
                        #     dtwpoint += 1

                        # print('------------------------------------')

                # calculate the total used cells

                # for i in matrix_ADTW:
                #     for j in i:
                #         if j != math.inf:
                #             count += 1

                # next train data
'''
# Original Dynamic Time Warping
print("---OriginDTW---")

# calculate the execution time
start_time = time.time()
print('Total Distance is {0:.3f}'.format(originDTW(Trim_TS, Trim_ComparedTS)))
print('Total time is {0:.5f}'.format(time.time() - start_time))

# My DTW -> Adaptive Dynamic Time Warping
print("---ADTW---")
# calculate the execution time
start_time = time.time()
print('Total Distance is {0:.3f}'.format(adaptiveWindowDTW(Trim_TS, Trim_ComparedTS)))
print('Total time is {0:.5f}'.format(time.time() - start_time))

'''

'''
def updateQueue(queue, inputValue, ts1_index, ts2_index):
    if inputValue in queue:
        return queue
    else:
        queue.insert(0, inputValue)
        queue = sorted(queue, key=itemgetter(0))
        return queue

def quicksort(queue):
    if len(queue) <= 1:
        return queue
    midcell = queue[len(queue) // 2]
    return quicksort([i for i in queue if i < midcell]) + [midcell] + quicksort([i for i in queue if i > midcell])
'''
