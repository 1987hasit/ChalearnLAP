# -*- coding:utf-8 -*-  

'''
Created on 24/03/2014

@author: Bin Liang
'''

import numpy as np
from skimage.feature import hog
import math
from sklearn import preprocessing
import cv2
from scipy.spatial.distance import mahalanobis, euclidean

from utils.public_utils import normalizeList
from numpy import mean
from common import GestureType, VIDEO_WIDTH, VIDEO_HEIGHT


def normQuaternions(quaternions):
    ''' normalize quaternions '''
    
    s = 0
    
    for i in quaternions:
        s += i ** 2
    
    magnitude = math.sqrt(s)
    
    normQuaternions = []
    
    for i in quaternions:
        normQuaternions.append(i / magnitude)
        
    return normQuaternions


def extractHoGFeatuer(image, widthGridNum, heightGridNum, orientationNum):
    ''' Extract hog feature from image '''
    
#     widthGridNum = 32 # Horizontal grid number
#     heightGridNum = 32 # Vertical grid number
#     orientationNum = 8  # Orientation number
    
    height, width = image.shape
    
    featureVector, hogImage = hog(image, orientations = orientationNum, 
                                  pixels_per_cell = (width/widthGridNum, height/heightGridNum),
                                  cells_per_block=(1, 1), visualise = True)
    
    return featureVector, hogImage


def extractPHOGFeature(image, level1, level2):
    ''' Extract PHOG feature from image '''
    height, width = image.shape
    orientationNum = 9
    pHOGVec = np.array([])
    widthBase = 4
    heightBase = 3
    
    for l in xrange(level1, level2 + 1):
        multi = 2 ** (l - 1)
        widthGridNum = widthBase * multi
        heightGridNum = heightBase * multi
        
        featureVector, hogImage = hog(image, orientations = orientationNum, 
                                  pixels_per_cell = (width/widthGridNum, height/heightGridNum),
                                  cells_per_block=(1, 1), visualise = True)
        
        featureVector = featureVector[np.newaxis].transpose()   # transpose to column vector
        
        if pHOGVec.size == 0:
            pHOGVec = featureVector.copy()
        else:
            pHOGVec = np.vstack((pHOGVec, featureVector))
        
    return pHOGVec


def extractMtmFeature(two_d_mtm):
    ''' hog feature extraction from 2DMTM '''
    mtmFeatureVec = np.array([])
    level1 = 1
    level2 = 3
    
    for i in xrange(len(two_d_mtm)):
        image = two_d_mtm[i]
        pHOGVec = extractPHOGFeature(image, level1, level2)
        
        if mtmFeatureVec.size == 0:
            mtmFeatureVec = pHOGVec.copy()
        else:
            mtmFeatureVec = np.vstack((mtmFeatureVec, pHOGVec))
        
    return mtmFeatureVec    


def extractTdFeature(gesture, gestureSample):
    ''' extract time domain features from gesture '''
    
    startFrame = gesture.startFrame
    endFrame = gesture.endFrame
    
    gestureFeatures = np.array([])
    
    for i in xrange(startFrame, endFrame + 1):
        skeleton = gestureSample.getSkeleton(i)
        worldCoordinates = skeleton.getWorldCoordinates()
        hipCenterCoordinate = worldCoordinates.get('HipCenter')
        if sum(hipCenterCoordinate) == 0:
            # if no skeleton data
            continue
        
        # 1. NRWP feature
        # extract normalized relative world position (NRWP) feature in each frame
        NRWPFeatureVec = extractNRWPFeature(gestureSample, i)    # [33 x 1]
        
        # 2. WPD feature
        # extract world pair distance (WPD) feature in each frame
        WPDFeatureVec = extractWPDFeature(gestureSample, i)
        
        # 3. RPP feature
        # extract normalized relative pixel position (NRPP) feature in each frame
        NRPPFeatureVec = extractNRPPFeature(gestureSample, i)    # [22 x 1]
        
        # 4. PPD feature
        # extract pixel pair distance (PPD) feature in each frame
        PPDFeatureVec = extractPPDFeature(gestureSample, i)
        
        # 5. orientation feature
        # extract orientation feature in each frame
        oriFeatureVec = extractOriFeature(gestureSample, i) # [44 x 1]
        
        currentGestureFeatureVec = np.vstack((NRWPFeatureVec, WPDFeatureVec, NRPPFeatureVec, PPDFeatureVec, oriFeatureVec))  
        
        # check validate data
        if np.isnan(currentGestureFeatureVec).any():
            continue
        
        if gestureFeatures.size == 0:
            gestureFeatures = currentGestureFeatureVec.copy()
        else:
            gestureFeatures = np.hstack((gestureFeatures, currentGestureFeatureVec))
            
    gestureFeatures = gestureFeatures.transpose() #  n_frame * n_feature 
    
    return  gestureFeatures


def extractNRWPFeature(gestureSample, frameNum):
    ''' extract normalized relative world position (NRWP) feature from one frame '''
    
    skeleton = gestureSample.getSkeleton(frameNum)
    worldCoordinates = skeleton.getWorldCoordinates()

    # hip center
    hipCenterCoordinate = worldCoordinates.get('HipCenter')
    
    # spine
    headCoordinate = worldCoordinates.get('Head')
    shoulderCenterCoordinate = worldCoordinates.get('ShoulderCenter')
    spineCoordinate = worldCoordinates.get('Spine')
    
    # left arm
    shoulderLeftCoordinate = worldCoordinates.get('ShoulderLeft')
    elbowLeftCoordinate = worldCoordinates.get('ElbowLeft')
    wristLeftCoordinate = worldCoordinates.get('WristLeft')
    handLeftCoordinate = worldCoordinates.get('HandLeft')
    
    # right arm
    shoulderRightCoordinate = worldCoordinates.get('ShoulderRight')
    elbowRightCoordinate = worldCoordinates.get('ElbowRight')
    wristRightCoordinate = worldCoordinates.get('WristRight')
    handRightCoordinate = worldCoordinates.get('HandRight')
    
    # relative positions of spine
    centerRP1 = np.array(headCoordinate) - np.array(hipCenterCoordinate)
    centerRP2 = np.array(shoulderCenterCoordinate) - np.array(hipCenterCoordinate)
    centerRP3 = np.array(spineCoordinate) - np.array(hipCenterCoordinate)
    
    # relative positions of left arm to hip center 
    leftRP1 = np.array(shoulderLeftCoordinate) - np.array(hipCenterCoordinate)
    leftRP2 = np.array(elbowLeftCoordinate) - np.array(hipCenterCoordinate)
    leftRP3 = np.array(wristLeftCoordinate) - np.array(hipCenterCoordinate)
    leftRP4 = np.array(handLeftCoordinate) - np.array(hipCenterCoordinate)
    
    # relative positions of right arm to hip center 
    rightRP1 = np.array(shoulderRightCoordinate) - np.array(hipCenterCoordinate)
    rightRP2 = np.array(elbowRightCoordinate) - np.array(hipCenterCoordinate)
    rightRP3 = np.array(wristRightCoordinate) - np.array(hipCenterCoordinate)
    rightRP4 = np.array(handRightCoordinate) - np.array(hipCenterCoordinate)
     
    # whole length
    originalCor = np.array([0., 0., 0.])
    wholeLen = euclidean(centerRP1, originalCor) + euclidean(centerRP2, originalCor) + euclidean(centerRP3, originalCor) \
             + euclidean(leftRP1, originalCor) + euclidean(leftRP2, originalCor) \
             + euclidean(leftRP3, originalCor) + euclidean(leftRP4, originalCor) \
             + euclidean(rightRP1, originalCor) + euclidean(rightRP2, originalCor) \
             + euclidean(rightRP3, originalCor) + euclidean(rightRP4, originalCor)
    
    nCenterRP1 = centerRP1 / wholeLen
    nCenterRP2 = centerRP2 / wholeLen
    nCenterRP3 = centerRP3 / wholeLen
    
    nLeftRP1 = leftRP1 / wholeLen
    nLeftRP2 = leftRP2 / wholeLen
    nLeftRP3 = leftRP3 / wholeLen
    nLeftRP4 = leftRP4 / wholeLen
    
    nRightRP1 = rightRP1 / wholeLen
    nRightRP2 = rightRP2 / wholeLen
    nRightRP3 = rightRP3 / wholeLen
    nRightRP4 = rightRP4 / wholeLen
     
    NRWPFeatureTmpArr = np.vstack((nCenterRP1, nCenterRP2, nCenterRP3, \
                                   nLeftRP1, nLeftRP2, nLeftRP3, nLeftRP4, \
                                   nRightRP1, nRightRP2, nRightRP3, nRightRP4))
    
    NRWPFeatureVec = NRWPFeatureTmpArr.reshape(33, 1) # NRWP feature vector
    
    return NRWPFeatureVec


def extractWPDFeature(gestureSample, frameNum):
    ''' extract world pair distance (WPD) feature in each frame '''
    skeleton = gestureSample.getSkeleton(frameNum)
    worldCoordinates = skeleton.getWorldCoordinates()
    
    coorList = []
    
    # hip center
    hipCenterCoordinate = worldCoordinates.get('HipCenter')
    coorList.append(hipCenterCoordinate)
    
    # spine
    headCoordinate = worldCoordinates.get('Head')
    shoulderCenterCoordinate = worldCoordinates.get('ShoulderCenter')
    spineCoordinate = worldCoordinates.get('Spine')
    coorList.append(headCoordinate)
    coorList.append(shoulderCenterCoordinate)
    coorList.append(spineCoordinate)
    
    # left arm
    shoulderLeftCoordinate = worldCoordinates.get('ShoulderLeft')
    elbowLeftCoordinate = worldCoordinates.get('ElbowLeft')
    wristLeftCoordinate = worldCoordinates.get('WristLeft')
    handLeftCoordinate = worldCoordinates.get('HandLeft')
    coorList.append(shoulderLeftCoordinate)
    coorList.append(elbowLeftCoordinate)
    coorList.append(wristLeftCoordinate)
    coorList.append(handLeftCoordinate)
    
    # right arm
    shoulderRightCoordinate = worldCoordinates.get('ShoulderRight')
    elbowRightCoordinate = worldCoordinates.get('ElbowRight')
    wristRightCoordinate = worldCoordinates.get('WristRight')
    handRightCoordinate = worldCoordinates.get('HandRight')
    coorList.append(shoulderRightCoordinate)
    coorList.append(elbowRightCoordinate)
    coorList.append(wristRightCoordinate)
    coorList.append(handRightCoordinate)
    
    length = len(coorList)
    WPDList = []
    for i in xrange(length):
        coor1 = coorList[i]
        for j in xrange(i + 1, length):
            coor2 = coorList[j]
            dist = euclidean(coor1, coor2)
            WPDList.append(dist)
    
    # normalize
    wholeLen = sum(WPDList)
    WPDTmpArr = np.array(WPDList)
    WPDTmpArr = WPDTmpArr / float(wholeLen)
    nDim = WPDTmpArr.size
    WPDFeatureVec = WPDTmpArr.reshape(nDim, 1)
    
    return WPDFeatureVec

def extractNRPPFeature(gestureSample, frameNum):
    ''' extract normalized relative pixel position feature from one frame '''
    
    skeleton = gestureSample.getSkeleton(frameNum)
    pixelCoordinates = skeleton.getPixelCoordinates()

    # hip center
    hipCenterCoordinate = pixelCoordinates.get('HipCenter')
    
    # spine
    headCoordinate = pixelCoordinates.get('Head')
    shoulderCenterCoordinate = pixelCoordinates.get('ShoulderCenter')
    spineCoordinate = pixelCoordinates.get('Spine')
    
    # left arm
    shoulderLeftCoordinate = pixelCoordinates.get('ShoulderLeft')
    elbowLeftCoordinate = pixelCoordinates.get('ElbowLeft')
    wristLeftCoordinate = pixelCoordinates.get('WristLeft')
    handLeftCoordinate = pixelCoordinates.get('HandLeft')
    
    # right arm
    shoulderRightCoordinate = pixelCoordinates.get('ShoulderRight')
    elbowRightCoordinate = pixelCoordinates.get('ElbowRight')
    wristRightCoordinate = pixelCoordinates.get('WristRight')
    handRightCoordinate = pixelCoordinates.get('HandRight')
    
    # relative positions of spine
    centerRP1 = np.array(headCoordinate) - np.array(hipCenterCoordinate)
    centerRP2 = np.array(shoulderCenterCoordinate) - np.array(hipCenterCoordinate)
    centerRP3 = np.array(spineCoordinate) - np.array(hipCenterCoordinate)
    
    # relative positions of left arm to hip center 
    leftRP1 = np.array(shoulderLeftCoordinate) - np.array(hipCenterCoordinate)
    leftRP2 = np.array(elbowLeftCoordinate) - np.array(hipCenterCoordinate)
    leftRP3 = np.array(wristLeftCoordinate) - np.array(hipCenterCoordinate)
    leftRP4 = np.array(handLeftCoordinate) - np.array(hipCenterCoordinate)
    
    # relative positions of right arm to hip center 
    rightRP1 = np.array(shoulderRightCoordinate) - np.array(hipCenterCoordinate)
    rightRP2 = np.array(elbowRightCoordinate) - np.array(hipCenterCoordinate)
    rightRP3 = np.array(wristRightCoordinate) - np.array(hipCenterCoordinate)
    rightRP4 = np.array(handRightCoordinate) - np.array(hipCenterCoordinate)
     
    # whole length
    originalCor = np.array([0., 0.])
    wholeLen = euclidean(centerRP1, originalCor) + euclidean(centerRP2, originalCor) + euclidean(centerRP3, originalCor) \
             + euclidean(leftRP1, originalCor) + euclidean(leftRP2, originalCor) \
             + euclidean(leftRP3, originalCor) + euclidean(leftRP4, originalCor) \
             + euclidean(rightRP1, originalCor) + euclidean(rightRP2, originalCor) \
             + euclidean(rightRP3, originalCor) + euclidean(rightRP4, originalCor)
    
    nCenterRP1 = centerRP1 / wholeLen
    nCenterRP2 = centerRP2 / wholeLen
    nCenterRP3 = centerRP3 / wholeLen
    
    nLeftRP1 = leftRP1 / wholeLen
    nLeftRP2 = leftRP2 / wholeLen
    nLeftRP3 = leftRP3 / wholeLen
    nLeftRP4 = leftRP4 / wholeLen
    
    nRightRP1 = rightRP1 / wholeLen
    nRightRP2 = rightRP2 / wholeLen
    nRightRP3 = rightRP3 / wholeLen
    nRightRP4 = rightRP4 / wholeLen
     
    NRPPTmpArr = np.vstack((nCenterRP1, nCenterRP2, nCenterRP3, \
                            nLeftRP1, nLeftRP2, nLeftRP3, nLeftRP4, \
                            nRightRP1, nRightRP2, nRightRP3, nRightRP4))
    
    NRPPFeatureVec = NRPPTmpArr.reshape(22, 1) # left arm RP feature
    
    return NRPPFeatureVec


def extractPPDFeature(gestureSample, frameNum):
    ''' extract pixel pair distance (WPD) feature in each frame '''
    skeleton = gestureSample.getSkeleton(frameNum)
    pixelCoordinates = skeleton.getPixelCoordinates()
    
    coorList = []

    # hip center
    hipCenterCoordinate = pixelCoordinates.get('HipCenter')
    coorList.append(hipCenterCoordinate)
    
    # spine
    headCoordinate = pixelCoordinates.get('Head')
    shoulderCenterCoordinate = pixelCoordinates.get('ShoulderCenter')
    spineCoordinate = pixelCoordinates.get('Spine')
    coorList.append(headCoordinate)
    coorList.append(shoulderCenterCoordinate)
    coorList.append(spineCoordinate)
    
    # left arm
    shoulderLeftCoordinate = pixelCoordinates.get('ShoulderLeft')
    elbowLeftCoordinate = pixelCoordinates.get('ElbowLeft')
    wristLeftCoordinate = pixelCoordinates.get('WristLeft')
    handLeftCoordinate = pixelCoordinates.get('HandLeft')
    coorList.append(shoulderLeftCoordinate)
    coorList.append(elbowLeftCoordinate)
    coorList.append(wristLeftCoordinate)
    coorList.append(handLeftCoordinate)
    
    # right arm
    shoulderRightCoordinate = pixelCoordinates.get('ShoulderRight')
    elbowRightCoordinate = pixelCoordinates.get('ElbowRight')
    wristRightCoordinate = pixelCoordinates.get('WristRight')
    handRightCoordinate = pixelCoordinates.get('HandRight')
    coorList.append(shoulderRightCoordinate)
    coorList.append(elbowRightCoordinate)
    coorList.append(wristRightCoordinate)
    coorList.append(handRightCoordinate)
    
    length = len(coorList)
    PPDList = []
    for i in xrange(length):
        coor1 = coorList[i]
        for j in xrange(i + 1, length):
            coor2 = coorList[j]
            dist = euclidean(coor1, coor2)
            PPDList.append(dist)
    
    # normalize
    wholeLen = sum(PPDList)
    PPDTmpArr = np.array(PPDList)
    PPDTmpArr = PPDTmpArr / float(wholeLen)
    nDim = PPDTmpArr.size
    PPDFeatureVec = PPDTmpArr.reshape(nDim, 1)
    
    return PPDFeatureVec


def extractOriFeature(gestureSample, frameNum):
    ''' extract relative orientation feature in each frame '''
    skeleton = gestureSample.getSkeleton(frameNum)
    joinOrientations = skeleton.getJoinOrientations()
    
    headOri = normQuaternions(joinOrientations.get('Head'))    
    shoulderCenterOri = normQuaternions(joinOrientations.get('ShoulderCenter'))
    spineOri = normQuaternions(joinOrientations.get('Spine'))
    
    shoulderLeftOri = normQuaternions(joinOrientations.get('ShoulderLeft'))
    elbowLeftOri = normQuaternions(joinOrientations.get('ElbowLeft'))
    wristLeftOri = normQuaternions(joinOrientations.get('WristLeft'))
    handLeftOri = normQuaternions(joinOrientations.get('HandLeft'))
    
    shoulderRightOri = normQuaternions(joinOrientations.get('ShoulderRight'))
    elbowRightOri = normQuaternions(joinOrientations.get('ElbowRight'))
    wristRightOri = normQuaternions(joinOrientations.get('WristRight'))
    handRightOri = normQuaternions(joinOrientations.get('HandRight'))
        
    oriTmpArr = np.vstack((headOri, shoulderCenterOri, spineOri, \
                           shoulderLeftOri, elbowLeftOri, wristLeftOri, handLeftOri, \
                           shoulderRightOri, elbowRightOri, wristRightOri, handRightOri))
    
    oriFeatureVec = oriTmpArr.reshape(44, 1)
    
    return oriFeatureVec
    

def extractHandDepthHogFeature(gestureSample, frameNum):
    ''' extract hand depth hog feature in each frame '''
    skeleton = gestureSample.getSkeleton(frameNum)
    pixelCoordinates = skeleton.getPixelCoordinates()
    
    handRegionWidth = 40
    handRegionHeight = 40
    
    handLeftPixelCoor = pixelCoordinates.get('HandLeft')
    handRightPixelCoor = pixelCoordinates.get('HandRight')
    #depthImg, _ = gestureSample.getGestureRegion(frameNum)
    
    _, depthImg = gestureSample.getDepth(frameNum)
    
    leftHandDepthHogFeatureVec = np.array([])
    rightHandDepthHogFeatureVec = np.array([])
    leftHandMask = np.array([])
    rightHandMask = np.array([])
    
    if handLeftPixelCoor[1]-handRegionHeight/2 >= 0 and handLeftPixelCoor[1]+handRegionHeight/2 < VIDEO_HEIGHT \
    and handRightPixelCoor[1]-handRegionHeight/2 >= 0 and handRightPixelCoor[1]+handRegionHeight/2 < VIDEO_HEIGHT \
    and handLeftPixelCoor[0]-handRegionWidth/2 >= 0 and handLeftPixelCoor[0]+handRegionWidth/2 < VIDEO_WIDTH \
    and handRightPixelCoor[0]-handRegionWidth/2 >= 0 and handRightPixelCoor[0]+handRegionWidth/2 < VIDEO_WIDTH:
    
        leftHandDepthRegion = depthImg[handLeftPixelCoor[1]-handRegionHeight/2 : handLeftPixelCoor[1]+handRegionHeight/2, 
                                handLeftPixelCoor[0]-handRegionWidth/2 : handLeftPixelCoor[0]+handRegionWidth/2]
        
        rightHandDepthRegion = depthImg[handRightPixelCoor[1]-handRegionHeight/2 : handRightPixelCoor[1]+handRegionHeight/2, 
                                handRightPixelCoor[0]-handRegionWidth/2 : handRightPixelCoor[0]+handRegionWidth/2]
    
    
        if leftHandDepthRegion.size == handRegionWidth * handRegionHeight and rightHandDepthRegion.size == handRegionWidth * handRegionHeight:
            # if hand depth region can be segmented
            
            leftHandMask = getHandMask(leftHandDepthRegion)
            rightHandMask = getHandMask(rightHandDepthRegion)
            leftHandDepthRegion[np.where(leftHandMask == 0)] = 0
            rightHandDepthRegion[np.where(rightHandMask == 0)] = 0
            
            widthGridNum = 3
            heightGridNum = 3
            orientationNum = 9
            featureDim = widthGridNum * heightGridNum * orientationNum
            
            leftHandDepthHogFeatureVec, leftHandDepthHogImg = extractHoGFeatuer(leftHandDepthRegion, widthGridNum, heightGridNum, orientationNum)
            rightHandDepthHogFeatureVec, rightHandDepthHogImg = extractHoGFeatuer(rightHandDepthRegion, widthGridNum, heightGridNum, orientationNum)
            
            leftHandDepthHogFeatureVec = leftHandDepthHogFeatureVec.reshape((featureDim, 1))
            rightHandDepthHogFeatureVec = rightHandDepthHogFeatureVec.reshape((featureDim, 1))
    
    return leftHandDepthHogFeatureVec, rightHandDepthHogFeatureVec, leftHandMask, rightHandMask

 
def getHandMask(handRegion):
    ''' get hand mask (binary image) from given hand region '''
    threshold, _ = cv2.threshold(handRegion, 127, 255, cv2.THRESH_OTSU)
    maskImg = np.copy(handRegion)
    maskImg[np.where(handRegion >= threshold)] = 0
    
    filteredMaskImg = cv2.medianBlur(maskImg, 5)
    
    return filteredMaskImg


def extractHandGrayHogFeature(gestureSample, frameNum, leftHandMask, rightHandMask):
    ''' extract hand depth hog feature in each frame '''
    skeleton = gestureSample.getSkeleton(frameNum)
    pixelCoordinates = skeleton.getPixelCoordinates()
    
    handRegionWidth = 40
    handRegionHeight = 40
    
    handLeftPixelCoor = pixelCoordinates.get('HandLeft')
    handRightPixelCoor = pixelCoordinates.get('HandRight')
    rgb = gestureSample.getRGB(frameNum)
    gray = cv2.cvtColor(rgb, cv2.cv.CV_RGB2GRAY)
    
    leftHandGrayHogFeatureVec = np.array([])
    rightHandGrayHogFeatureVec = np.array([])
    
    if handLeftPixelCoor[1]-handRegionHeight/2 >= 0 and handLeftPixelCoor[1]+handRegionHeight/2 < VIDEO_HEIGHT \
    and handRightPixelCoor[1]-handRegionHeight/2 >= 0 and handRightPixelCoor[1]+handRegionHeight/2 < VIDEO_HEIGHT \
    and handLeftPixelCoor[0]-handRegionWidth/2 >= 0 and handLeftPixelCoor[0]+handRegionWidth/2 < VIDEO_WIDTH \
    and handRightPixelCoor[0]-handRegionWidth/2 >= 0 and handRightPixelCoor[0]+handRegionWidth/2 < VIDEO_WIDTH:
    
        leftHandGrayRegion = gray[handLeftPixelCoor[1]-handRegionHeight/2 : handLeftPixelCoor[1]+handRegionHeight/2, 
                                handLeftPixelCoor[0]-handRegionWidth/2 : handLeftPixelCoor[0]+handRegionWidth/2]
        
        rightHandGrayRegion = gray[handRightPixelCoor[1]-handRegionHeight/2 : handRightPixelCoor[1]+handRegionHeight/2, 
                                handRightPixelCoor[0]-handRegionWidth/2 : handRightPixelCoor[0]+handRegionWidth/2]
    
    
    
        if leftHandGrayRegion.size == handRegionWidth * handRegionHeight and rightHandGrayRegion.size == handRegionWidth * handRegionHeight:
            # if hand depth region can be segmented
            
            leftHandGrayRegion[np.where(leftHandMask == 0)] = 0
            rightHandGrayRegion[np.where(rightHandMask == 0)] = 0
            
            widthGridNum = 3
            heightGridNum = 3
            orientationNum = 9
            featureDim = widthGridNum * heightGridNum * orientationNum
            
            leftHandGrayHogFeatureVec, leftHandGrayHogImg = extractHoGFeatuer(leftHandGrayRegion, widthGridNum, heightGridNum, orientationNum)
            rightHandGrayHogFeatureVec, rightHandGrayHogImg = extractHoGFeatuer(rightHandGrayRegion, widthGridNum, heightGridNum, orientationNum)
            
            leftHandGrayHogFeatureVec = leftHandGrayHogFeatureVec.reshape((featureDim, 1))
            rightHandGrayHogFeatureVec = rightHandGrayHogFeatureVec.reshape((featureDim, 1))
    
    return leftHandGrayHogFeatureVec, rightHandGrayHogFeatureVec

def extractStaFeature(timeDomainFeatures):
    ''' extract statistic skeleton features '''
    
    skeletonDataArr = timeDomainFeatures.transpose()
    nDim, _ = skeletonDataArr.shape
    meanVec = np.mean(skeletonDataArr, axis = 1).reshape(nDim, 1)
    varVec = np.var(skeletonDataArr, axis = 1).reshape(nDim, 1)
    maxVec = np.max(skeletonDataArr, axis = 1).reshape(nDim, 1)
    minVec = np.min(skeletonDataArr, axis = 1).reshape(nDim, 1)
    
    staFeaVec = np.vstack((meanVec, varVec, maxVec, minVec))
    return staFeaVec