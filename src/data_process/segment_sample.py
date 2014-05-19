'''
Created on 08/05/2014

@author:  Bin Liang

'''
import numpy as np
import os
from common import GestureType
from utils.public_utils import groupConsecutives
from numpy import mean
import matplotlib.pyplot as plt
from utils.ChalearnLAPSample import GestureSample
from timeit import itertools
from scipy.spatial.distance import jaccard, euclidean
import config
import common


def getTestGestureLabels(testDataDir):
    ''' Split unlabeled gesture samples and save the files '''
    
    # label dictionary
    labelsDict = {}
    
    if os.path.exists("pred_labels.npy"):
        labelsDict = np.load("pred_labels.npy")
        return labelsDict.item()    # convert to dictionary
    
    # Get the list of training samples    
    samples = os.listdir(testDataDir)
    scoreList = []
    
    # Access to each sample
    for samplFile in samples:
        if not samplFile.endswith(".zip"):
            continue
        
        print "\t Labeling file", samplFile
        
        # Create the object to access the sample
        smp = GestureSample(os.path.join(testDataDir, samplFile))
        numFrames = smp.getNumFrames()  # total number of frames in the sample
        
        firstFrameNum = 0
        
        gestureTypeList = []
        
        #######################################################################
        # calculate the score list
        for i in xrange(1, numFrames + 1):
            
            # if out of the sequence, break the loop
            if i + config.SEGMENT_WINDOW_SIZE - 1 > numFrames:
                break
            
            skel = smp.getSkeleton(i)            
            hipCenterCors = skel.getWorldCoordinates().get('HipCenter')
            
            # decide the first frame number
            if firstFrameNum == 0:
                if np.sum(hipCenterCors) == 0:
                    continue
                else:
                    firstFrameNum = i
                          
            gestureType = getGestureType(i, config.SEGMENT_WINDOW_SIZE, smp)
            gestureTypeList.append(gestureType)
        
        # insert blank frames into type list
        for f in xrange(1, firstFrameNum):
            gestureTypeList.insert(0, 0)
            
        # get labels
        labels = smp.getLabels()
        labelList = []
        
        # if the ground truth labels exist
        if len(labels) > 0:
            labelArray = np.array(labels)
            
            for row in labelArray:
#                 labelList.append(row[1])
#                 labelList.append(row[2])
                labelList.append([row[1], row[2]])
        
        processedGestureTypeList = processGestureTypeList(gestureTypeList)
        predLabelList = getLabelListFromProcessedGestureTypeList(processedGestureTypeList)
        # save pred_labels.npy
        labelsDict[smp.seqID] = predLabelList
                
#         # show predict label result 
#         dsipLabelList = list(itertools.chain.from_iterable(predLabelList))    
#           
#         # show processed gesture type list
#         dsipGestureTypeList = list(itertools.chain.from_iterable(processedGestureTypeList))
#                 
#         plotScoreList(dsipGestureTypeList, dsipLabelList)
        
#         # jaccard score
#         u = np.zeros(numFrames)
#         v = np.zeros(numFrames)
#              
#         for item in labelList:
#             startIdx = item[0] - 1
#             endIdx = item[1] - 1
#              
#             u[startIdx : endIdx + 1] = 1
#              
#         for item in predLabelList:
#             startIdx = item[0] - 1
#             endIdx = item[1] - 1
#              
#             v[startIdx : endIdx + 1] = 1
#              
#          
#          
#         score = jaccard(u, v)
#         scoreList.append(score)
#          
#     finalScore = np.mean(scoreList)
    
    np.save("pred_labels", labelsDict)
    return labelsDict


def processGestureTypeList(gestureTypeList):
    ''' process gesture type list, remove sudden pulse '''
    
    min_thresh_interval = 10
    max_thresh_interval = 60
    
    groupedGestureTypeList = groupConsecutives(gestureTypeList, 0)
    finalGroupedGestureTypeList = []
    
    # remove extreme short interval
    for i in xrange(len(groupedGestureTypeList)):
        gestureInterval = groupedGestureTypeList[i]
        gestureType = mean(gestureInterval)
        
        if gestureType != GestureType.rest:
            # if geture type is not rest
            intervalLen = len(gestureInterval)
            
            if intervalLen <= min_thresh_interval:
                # if the interval is extremely short, set its type to rest
                for j in xrange(intervalLen):
                    gestureInterval[j] = GestureType.rest
                
                finalGroupedGestureTypeList.append(gestureInterval)
                
            elif intervalLen >= max_thresh_interval:
                middle = len(gestureInterval) / 2
                gestureInterval1 = gestureInterval[:middle - 1]
                gestureInterval2 = gestureInterval[middle + 1:]
                
                finalGroupedGestureTypeList.append(gestureInterval1)
                finalGroupedGestureTypeList.append(gestureInterval2)
            else:
                finalGroupedGestureTypeList.append(gestureInterval)
        else:
            finalGroupedGestureTypeList.append(gestureInterval)
        
    processedGroupedGestureTypeList = finalGroupedGestureTypeList
    
    return processedGroupedGestureTypeList


def getLabelListFromProcessedGestureTypeList(groupedGestureTypeList):
    ''' get label list from  processed gesture type list '''
    
    labelList = []
    realIdx = 0
    for i in xrange(len(groupedGestureTypeList)):
        
        gestureInterval = groupedGestureTypeList[i]
        
        if i != 0: 
            realIdx += len(groupedGestureTypeList[i-1])
        
        if (mean(gestureInterval) != GestureType.rest) and (len(gestureInterval) > 10):
            # if gesture type is not rest and the gesture interval is larger than 10
            startIdx = realIdx + 1  # frame number is larger than index number
            endIdx = startIdx + len(gestureInterval) - 1
            labelList.append([startIdx, endIdx])
            
    return labelList


def plotScoreList(scoreList, labelList):
    ''' plot score list '''
    scoreListLen = len(scoreList)
    xLabel = range(1, scoreListLen + 1)
    #plt.plot(xLabel, scoreList, 'bo', xLabel, scoreList, 'k')
    
    plt.figure()
    plt.plot(xLabel, scoreList, 'k')
    plt.ylabel('score')    
    plt.xlabel('frame number')
    
    # ground truth label
    plt.vlines(labelList, 0, 1, colors = 'r', linestyles='dashed')
    
    plt.show()
    

def getGestureType(frameNum, sizeWindow, gestureSample):
    ''' get gesture type '''
    
    lrHandPixelDiffWindowed, height = getLrHandPosDiffWindowed(frameNum, sizeWindow, gestureSample)
    
    thresh1 = height / 5.
    thresh2 = height / 10.
    
    if lrHandPixelDiffWindowed > thresh1:
        # left hand dominant
        gestureType = GestureType.left_hand
    elif lrHandPixelDiffWindowed < - thresh1:
        # right hand dominant
        gestureType = GestureType.right_hand
    else:
        # rest and both hands cases
        handLines = []
        hipLines = []
        
        for i in xrange(frameNum, frameNum + sizeWindow):
            skeleton = gestureSample.getSkeleton(i)
            pixelCoordinates = skeleton.getPixelCoordinates()
            
            leftHipCor = common.VIDEO_HEIGHT - pixelCoordinates.get('HipLeft')[1]
            rightHipCor = common.VIDEO_HEIGHT - pixelCoordinates.get('HipRight')[1]
            leftHandCor = common.VIDEO_HEIGHT - pixelCoordinates.get('HandLeft')[1]
            rightHandCor = common.VIDEO_HEIGHT - pixelCoordinates.get('HandRight')[1]
            
            handLineCor = (leftHandCor + rightHandCor) / 2.
            hipLineCor = (leftHipCor + rightHipCor) / 2.
            
            handLines.append(handLineCor)
            hipLines.append(hipLineCor)
        
        handLine = mean(handLines)
        hipLine = mean(hipLines)
        
        if handLine - hipLine > thresh2:
            gestureType = GestureType.both_hands
        else:
            gestureType = GestureType.rest
    
    return gestureType


def getLrHandPosDiffWindowed(frameNum, sizeWindow, gestureSample):
    ''' using sliding window to get hands position difference '''
    
    winDiffs = []
    heightList = []
    
    for i in xrange(frameNum, frameNum + sizeWindow):
        skeleton = gestureSample.getSkeleton(i)
        pixelCoordinates = skeleton.getPixelCoordinates()
        
        shoulderCenterCor = pixelCoordinates.get('ShoulderCenter')
        hipCenterCor = pixelCoordinates.get('HipCenter')
        
        height = euclidean(hipCenterCor, shoulderCenterCor)
        
        diff = getLrHandPosDiff(skeleton)
        winDiffs.append(diff)
        heightList.append(height)
        
    lrHandPixelDiffWindowed = mean(winDiffs)
    height = mean(heightList)
    
    return lrHandPixelDiffWindowed, height


def getLrHandPosDiff(skeleton):
    ''' get the difference between left hand position and right hand position '''
    
    #worldCoordinates = skeleton.getWorldCoordinates()
    pixelCoordinates = skeleton.getPixelCoordinates()
    
    #leftHandWorldCor = worldCoordinates.get('HandLeft')
    #rightHandWorldCor = worldCoordinates.get('HandRight')
    
    leftHandPixelCor = pixelCoordinates.get('HandLeft')
    rightHandPixelCor = pixelCoordinates.get('HandRight')
    
    # y direction difference, if lrHandPixelDiff > 0, left hand is higher than right hand, vice-versa
    lrHandPixelDiff = - (leftHandPixelCor[1] - rightHandPixelCor[1])  
    
    return lrHandPixelDiff