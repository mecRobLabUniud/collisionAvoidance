#!/usr/bin/env python3

from utils.kalman_filter import KalmanFilter as kfil
from utils.speed_kalman_filter import KalmanFilter as skfil
import numpy as np


arr1 = [[0.036002252250909805, -0.3748200535774231, 0.346359521150589], [-0.07815882563591003, -0.3534291386604309, 0.35554203391075134]]
arr2 = [[0.097, -0.37, 0.35], [-0.018, -0.35, 0.36]]
arr3 = [[-0.01078223530203104, -0.4673510491847992, 0.34931594133377075], [np.nan, np.nan, np.nan]]
arr4 = [[np.nan, np.nan, np.nan], [0.2507195472717285, 0.054586492478847504, 0.04815799370408058]]
arr5 = [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]

# skeletons = [arr1, arr2, arr3, arr4, arr5]
skeletons = [arr2, arr4]

kfs = []
skfs = []
for i in range(len(skeletons)):
    kfs.append(kfil())
    skfs.append(skfil())

for i in range(10):
    for n, array in enumerate(skeletons):
        print(f"\n stage {i} ======================")
        print(array)
        
        conf = [0.9, 0.9]

        
        res = kfs[n].step(array, conf)
        print("result kfil class: ", res)

        res = skfs[n].step(array, conf)
        print("result SPEED kfil class: ", res)
        