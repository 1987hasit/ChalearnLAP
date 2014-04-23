# -*- coding:utf-8 -*-  

'''
Created on 24/03/2014

@author: Bin Liang
'''


import numpy as np
from sklearn import preprocessing
import os
from utils.ChalearnLAPSample import GestureSample


def normalizeList(inputList, normType):
    ''' normalize input list to 0-1 
        inputList: row vector
        type: min_max, zero_mean
    '''
    inputListArr = np.array(inputList, dtype = float)
    size = inputListArr.size
    
    # reshape to column vector for scaling
    columnVec = inputListArr.reshape(size, 1)
    
    if normType == 'min_max':
        min_max_scaler = preprocessing.MinMaxScaler()
        # max min scale to range [0, 1]
        columnVecNormed = min_max_scaler.fit_transform(columnVec)
        
        # reshape back to row vector
        normalizedArray = columnVecNormed.reshape(1, size)
        normalizedList = normalizedArray.tolist()[0]
    
    elif normType == 'zero_mean':
        columnVecNormed = preprocessing.scale(columnVec)
        normalizedArray = columnVecNormed.reshape(1, size)
        normalizedList = normalizedArray.tolist()[0]
        
    return normalizedList


def getGestureLengthList(trainDataDir):
    ''' get average length of gestures from training samples '''
    
    # Get the list of training samples    
    samples = os.listdir(trainDataDir)
    
    lengthList = []
    
    for samplFile in samples:
        if not samplFile.endswith(".zip"):
            continue
        
        smp = GestureSample(os.path.join(trainDataDir, samplFile))
        
        # get labels
        labels = smp.getLabels()
        labelArray = np.array(labels)
        resultArray = labelArray[:, 2] - labelArray[:, 1] + 1
        resultList = resultArray.tolist()
        
        lengthList += resultList
        
    return lengthList


def groupConsecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


def removeall(path):
    ''' remove all files and folders in the path '''
    if not os.path.isdir(path):
        return
    
    files = os.listdir(path)

    for x in files:
        fullpath = os.path.join(path, x)
        if os.path.isfile(fullpath):
            os.remove(fullpath)
        elif os.path.isdir(fullpath):
            removeall(fullpath)
            os.rmdir(fullpath)
