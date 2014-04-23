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
from math import ceil

from representation.gesture_rep import computeMHI
from utils.public_utils import normalizeList
from numpy import mean
from common import GestureType


def extractSingleSkeletonFeature(skeleton):
    ''' Extract skeleton feature from world coordinates in one frame  '''
    
    worldCoordinates = skeleton.getWorldCoordinates() 
    skeletonArray = np.array([item for item in worldCoordinates.itervalues()])
    
    hipCenterCoordinates = worldCoordinates.get('HipCenter')
    jointNumber = 20
    tmpArray = np.ones((jointNumber, 1)) * np.array([hipCenterCoordinates])
    
    # Relative joint distance from hip center
    skeletonFeature = skeletonArray - tmpArray  # [20 x 3]
    skeletonFeature = skeletonFeature.reshape(60, 1)    # [60 x 1]
    
    return skeletonFeature


def extractSelectedSkeletonFeature(skeleton):
    ''' Extract selected skeleton feature from world coordinates in one frame '''
    
    worldCoordinates = skeleton.getWorldCoordinates()
    
    hipCenterCoordinates = worldCoordinates.get('HipCenter')
    
    #shoulderLeftCoordinates = worldCoordinates.get('ShoulderLeft')
    elbowLeftCoordinates = worldCoordinates.get('ElbowLeft')
    wristLeftCoordinates = worldCoordinates.get('WristLeft')
    
    
    #shoulderRightCoordinates = worldCoordinates.get('ShoulderRight')
    elbowRightCoordinates = worldCoordinates.get('ElbowRight')
    wristRightCoordinates = worldCoordinates.get('WristRight')
    
    
    skeletonArray = np.vstack((elbowLeftCoordinates, wristLeftCoordinates, elbowRightCoordinates, wristRightCoordinates))
    
    jointNumber = 4
    tmpArray = np.ones((jointNumber, 1)) * np.array([hipCenterCoordinates])
    
    # Relative joint distance from hip center
    selectedSkelFeature = skeletonArray - tmpArray  # [4 x 3]
    selectedSkelFeature = selectedSkelFeature.reshape(12, 1)    # [12 x 1]    
    
    return selectedSkelFeature


def extractSelectedOrientationFeature(skeleton):
    ''' Extract selected skeleton feature from world coordinates in one frame
        The quaternion is the orientation of bone that is relative ti the child joint 
    '''
    
    joinOrientations = skeleton.getJoinOrientations()
        
    #hipCenterOri = normQuaternions(joinOrientations.get('HipCenter'))
    
    #shoulderLeftOri = normQuaternions(joinOrientations.get('ShoulderLeft'))
    elbowLeftOri = normQuaternions(joinOrientations.get('ElbowLeft'))
    wristLeftOri = normQuaternions(joinOrientations.get('WristLeft'))
    
    #shoulderRightOri = normQuaternions(joinOrientations.get('ShoulderRight'))
    elbowRightOri = normQuaternions(joinOrientations.get('ElbowRight'))
    wristRightOri = normQuaternions(joinOrientations.get('WristRight'))
        
    #relativeOri1 = np.dot(elbowLeftOri, hipCenterOri)
    #relativeOri2 = np.dot(wristLeftOri, hipCenterOri)
    #relativeOri3 = np.dot(elbowRightOri, hipCenterOri)
    #relativeOri4 = np.dot(wristRightOri, hipCenterOri)
    
    orientationList = []
    orientationList.append(elbowLeftOri)
    orientationList.append(wristLeftOri)
    orientationList.append(elbowRightOri)
    orientationList.append(wristRightOri)

    orientationArray = np.array(orientationList)
    
    selectedOriFeature = orientationArray.reshape(4, 1)   # [4 x 1]    
    
    return selectedOriFeature


def extractContourFeature(binaryImg):
    ''' extract contour features from binary image '''
    
    contours, _ = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # why no contours shown in the image?????????
    # draw countour image
    # cv2.drawContours(binaryImg, contours, -1,(0,0,255), 3)
    # cv2.imshow('contour', binaryImg), cv2.waitKey(0)
    
    cnt = contours[0]
    M = cv2.moments(cnt)
    huMoments = cv2.HuMoments(M)
    
    return huMoments


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


def compareQuternionMatrix(quternionMatrix1, quternionMatrix2):
    ''' compare two quternion matrices '''
    
    row, _ = quternionMatrix1.shape
    
    resultVec = []
    
    for i in range(row):
        quaternion1 = quternionMatrix1[i, :]
        quaternion2 = quternionMatrix2[i, :]
        
        result = np.dot(quaternion1, quaternion2)
        resultVec.append(result)
    
    return resultVec


def extractSelectedPixelFeature(skeleton):
    ''' Extract selected skeleton feature from world coordinates in one frame '''
    
    ''' Extract selected skeleton feature from world coordinates in one frame '''
    
    pixelCoordinates = skeleton.getPixelCoordinates()
    
    hipCenterCoordinates = pixelCoordinates.get('HipCenter')
    
    #shoulderLeftCoordinates = worldCoordinates.get('ShoulderLeft')
    elbowLeftCoordinates = pixelCoordinates.get('ElbowLeft')
    wristLeftCoordinates = pixelCoordinates.get('WristLeft')
    
    
    #shoulderRightCoordinates = worldCoordinates.get('ShoulderRight')
    elbowRightCoordinates = pixelCoordinates.get('ElbowRight')
    wristRightCoordinates = pixelCoordinates.get('WristRight')
    
    
    pixelArray = np.vstack((elbowLeftCoordinates, wristLeftCoordinates, elbowRightCoordinates, wristRightCoordinates))
    
    jointNumber = 4
    tmpArray = np.ones((jointNumber, 1)) * np.array([hipCenterCoordinates])
    
    # Relative joint distance from hip center
    selectedPixelFeature = pixelArray - tmpArray  # [2 x 3]
    selectedPixelFeature = selectedPixelFeature.reshape(8, 1)    # [8 x 1]
    
    return selectedPixelFeature


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
       

def getScoreListFromFeatureArray(featureArray, distanceType):
    ''' calculate scores from an array, each column is a feature vector for one frame 
        distance type: mahalanobis, euclidean
    '''
    
    # scale array using zero mean and unit variance
    transposedFeatureArray = featureArray.transpose()   # each column has the same type of feature
    transposedFeatureArrayScaled = preprocessing.scale(transposedFeatureArray)
    
    # transpose again
    featureArrayScaled = transposedFeatureArrayScaled.transpose()
    
    # calculate scores
    scoreList = []
    
    if distanceType == 'euclidean':
        # euclidean distance
        initFeatureVec = featureArrayScaled[:, 0]
        _, col = featureArrayScaled.shape
        
        for i in range(col):
            currentFeatureVec = featureArrayScaled[:, i]
            #score = LA.norm((currentFeatureVec - initFeatureVec), 2)
            score = euclidean(currentFeatureVec, initFeatureVec)
            scoreList.append(score)
       
        # using max min scale to range [0, 1]
        #scoreList = normalizeList(scoreList, 'zero_mean')
    
    elif distanceType == 'mahalanobis':
        # mahalanobis distance
        initFeatureVec = transposedFeatureArrayScaled[0]
        row, _ = transposedFeatureArrayScaled.shape
        
        # calculate covariance
        covar = np.cov(transposedFeatureArrayScaled, rowvar=0)
        if(covar.shape[1:2] == (1,)):
            invcovar = np.linalg.inv(covar.reshape(1,1))
        else:
            invcovar = np.linalg.inv(covar)
        
        for i in range(row):
            currentFeatureVec = transposedFeatureArrayScaled[i]
            score = mahalanobis(currentFeatureVec, initFeatureVec, invcovar)
            scoreList.append(score)
        
        # using max min scale to range [0, 1]
        scoreList = normalizeList(scoreList, 'zero_mean')
        
    return scoreList    


# Self added
def extractGestureFeatures(gesture, gestureSample):
    """ Extract features from one gesture of a gesture sequence (gestureSample) """
    
    startFrame = gesture.getStartFrame()
    endFrame = gesture.getEndFrame()
            
    # window size of the gesture sequence
    #windowSize = endFrame - start_frame + 1
    windowSize = 5
    numWindow = int(ceil( (endFrame - startFrame + 1) / float(windowSize) ))
    
    # features initialization
    finalSkeletonFeatures = np.zeros((60, numWindow))
    
    widthGridNum = 32; heightGridNum = 32; orientationNum = 8
    depthFeatureSize = widthGridNum * heightGridNum * orientationNum  # depth (HOG) feature size
    finalDepthFeatures = np.zeros((depthFeatureSize, numWindow))
    
    # window video initialization
    # xoy image height and width
    firstXoyFrm, _ = gestureSample.getGestureRegion(1)
    heightLen, widthLen = firstXoyFrm.shape
    
    # xoz yoz image height and width
    firstXozFrm, firstYozFrm = gestureSample.getProjection(1)
    heightLenXoz, widthLenXoz = firstXozFrm.shape
    heightLenYoz, widthLenYoz = firstYozFrm.shape
    
    winXoyVideo = np.zeros((heightLen, widthLen, windowSize))
    winXozVideo = np.zeros((heightLenXoz, widthLenXoz, windowSize))
    winYozVideo = np.zeros((heightLenYoz, widthLenYoz, windowSize))
    
    for i in range(numWindow):
        winStartFrame = startFrame + i * windowSize # window start frame
        winEndFrame = winStartFrame + windowSize - 1    # window end frame
        if winEndFrame > endFrame:
            winEndFrame = endFrame
        
        # Skeleton features in one window, size [60 x windowSize]
        winSekeltonFeatures = np.zeros((60, windowSize))
            
        # process in one window
        for numFrame in range(winStartFrame, winEndFrame + 1):
            winNumFrame = numFrame - winStartFrame  # frame number in one window
            
            # skeleton feature
            skeleton = gestureSample.getSkeleton(numFrame)
            singleSkeletonFeature = extractSingleSkeletonFeature(skeleton)
            
            singleSkeletonFeature.shape = (60,)
            
            winSekeltonFeatures[:, winNumFrame] = singleSkeletonFeature
            
            # depth feature
            xoyImg, _ = gestureSample.getGestureRegion(numFrame)
            xozImg, yozImg = gestureSample.getProjection(numFrame)
            
            winXoyVideo[:, :, winNumFrame] = xoyImg
            winXozVideo[:, :, winNumFrame] = xozImg
            winYozVideo[:, :, winNumFrame] = yozImg
        
        # max pooling of skeleton features in each window    
        finalSkeletonFeatures[:, i] = np.amax(winSekeltonFeatures, axis = 1)
        
        mhiXoy = computeMHI(winXoyVideo, thresh = 10)
#         mhiXoz = computeMHI(winXozVideo, thresh = 10)
#         mhiYoz = computeMHI(winYozVideo, thresh = 10)
        
        xoyHogFeature, hogImage = extractHoGFeatuer(mhiXoy, widthGridNum, heightGridNum, orientationNum)
        finalDepthFeatures[:, i] = xoyHogFeature
    
    return finalSkeletonFeatures, finalDepthFeatures    
    

def extractSingleGestureFeatures(gesture, gestureSample):
    ''' extract features from a single gesture in one sample file '''
    
    featureArray = np.array([])
    startFrame = gesture.start_frame
    endFrame = gesture.endFrame
    
    for i in range(startFrame, endFrame + 1):
        
        skel = gestureSample.getSkeleton(i)
        hipCenterCors = skel.getWorldCoordinates().get('HipCenter')
        
        # remove some frame which has no skeleton data
        if np.sum(hipCenterCors) == 0:
            continue
        
        skeletonFeature = extractSelectedSkeletonFeature(skel)
        skeletonFeatureVec = skeletonFeature.reshape(12, 1)
        
        orientationFeature = extractSelectedOrientationFeature(skel)
        orientationFeatureVec = orientationFeature.reshape(4, 1)
        
        pixelFeature = extractSelectedPixelFeature(skel)
        pixelFeatureVec = pixelFeature.reshape(8, 1)
        
        finalFeatureVec = np.vstack((skeletonFeatureVec, orientationFeatureVec, pixelFeatureVec))
        
        if featureArray.size == 0:
            featureArray = finalFeatureVec
        else:
            featureArray = np.hstack((featureArray, finalFeatureVec))
            
    featureArrayScaled = preprocessing.scale(featureArray, axis = 1)
    return featureArrayScaled   # shape(n_feature, n_frame)


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
    

def getLrHandPosDiffWindowed(frameNum, sizeWindow, gestureSample):
    ''' using sliding window to get hands position difference '''
    
    winDiffs = []
    heightList = []
    
    for i in range(frameNum, frameNum + sizeWindow):
        skeleton = gestureSample.getSkeleton(i)
        pixelCoordinates = skeleton.getPixelCoordinates()
        height = 360 - pixelCoordinates.get('Head')[1] # 360 is the height of the image
        
        diff = getLrHandPosDiff(skeleton)
        winDiffs.append(diff)
        heightList.append(height)
        
    lrHandPixelDiffWindowed = mean(winDiffs)
    height = mean(heightList)
    
    return lrHandPixelDiffWindowed, height


def getGestureType(frameNum, sizeWindow, gestureSample):
    ''' get gesture type '''
    
    lrHandPixelDiffWindowed, height = getLrHandPosDiffWindowed(frameNum, sizeWindow, gestureSample)
    
    thresh1 = height / 5.
    thresh2 = - height / 30.
    
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
        
        for i in range(frameNum, frameNum + sizeWindow):
            skeleton = gestureSample.getSkeleton(i)
            pixelCoordinates = skeleton.getPixelCoordinates()
            
            leftHipCor = 360 - pixelCoordinates.get('HipLeft')[1]
            rightHipCor = 360 - pixelCoordinates.get('HipRight')[1]
            leftHandCor = 360 - pixelCoordinates.get('HandLeft')[1]
            rightHandCor = 360 - pixelCoordinates.get('HandRight')[1]
            
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


def getGestureTypeForGesture(gesture, gestureSample):
    ''' get gesture type for gesture in the training stage '''
    startFrame = gesture.startFrame
    endFrame = gesture.endFrame
    gestureTypes = []
    
    for i in range(startFrame, endFrame + 1):
        skeleton = gestureSample.getSkeleton(i)
        pixelCoordinates = skeleton.getPixelCoordinates()

        heightCor = 360 - pixelCoordinates.get('Head')[1]
        diffCor = getLrHandPosDiff(skeleton)
        
        thresh1 = heightCor / 5.
    
        if diffCor > thresh1:
            # left hand
            gestureType = GestureType.left_hand
        elif diffCor < - thresh1:
            # right hand
            gestureType = GestureType.right_hand
        else:
            gestureType = GestureType.both_hands
        
        gestureTypes.append(gestureType)
        
    
    gestureTypesArr = np.array(gestureTypes)
    numLeftHand = gestureTypesArr[np.where(gestureTypesArr == GestureType.left_hand)].size
    numRightHand = gestureTypesArr[np.where(gestureTypesArr == GestureType.right_hand)].size
    numBothHands = gestureTypesArr[np.where(gestureTypesArr == GestureType.both_hands)].size
    
    predTypeDict = {numLeftHand : GestureType.left_hand, numRightHand : GestureType.right_hand, numBothHands : GestureType.both_hands}
    
    maxNum = max(numLeftHand, numRightHand, numBothHands)
    gestureType = predTypeDict.get(maxNum)
    
    return gestureType


def extractGestureFeature(gesture, gestureSample):
    ''' extract gesture features from gesture '''
    
    startFrame = gesture.startFrame
    endFrame = gesture.endFrame
    
    leftGestureFeatures = np.array([])
    rightGestureFeatures = np.array([])
    
    for i in range(startFrame, endFrame + 1):
        skeleton = gestureSample.getSkeleton(i)
        worldCoordinates = skeleton.getWorldCoordinates()
        hipCenterCoordinate = worldCoordinates.get('HipCenter')
        if sum(hipCenterCoordinate) == 0:
            # if no skeleton data
            continue
        
        # extract relative position (RP) feature in each frame
        leftRpFeatureVec, rightRpFeatureVec = extractRPFeature(gestureSample, i)    # [12 x 1] and [12 x 1]
        
        # extract orientation feature in each frame
        leftOriFeatureVec, rightOriFeatureVec = extractOriFeature(gestureSample, i) # [16 x 1] and [16 x 1]
        
        # extract hand depth hog feature in each frame
        leftHandDepthHogFeatureVec, rightHandDepthHogFeatureVec = extractHandDepthHogFeature(gestureSample, i) # [81 x1] and [81 x 1]
        
        # extract hand gray hog feature in each frame
        leftHandGrayHogFeatureVec, rightHandGrayHogFeatureVec = extractHandGrayHogFeature(gestureSample, i) # [81 x1] and [81 x 1]
        
        if leftHandDepthHogFeatureVec.size == 0 or rightHandDepthHogFeatureVec.size == 0 \
            or leftHandGrayHogFeatureVec.size == 0 or rightHandGrayHogFeatureVec.size == 0:
            continue 
        
        currentLeftGestureFeatureVec = np.vstack((leftRpFeatureVec, leftOriFeatureVec, leftHandDepthHogFeatureVec, leftHandGrayHogFeatureVec))  
        currentRigthGestureFeatureVec = np.vstack((rightRpFeatureVec, rightOriFeatureVec, rightHandDepthHogFeatureVec, rightHandGrayHogFeatureVec))
        
        if leftGestureFeatures.size == 0:
            leftGestureFeatures = currentLeftGestureFeatureVec
        else:
            leftGestureFeatures = np.hstack((leftGestureFeatures, currentLeftGestureFeatureVec))
            
        if rightGestureFeatures.size == 0:
            rightGestureFeatures = currentRigthGestureFeatureVec
        else:
            rightGestureFeatures = np.hstack((rightGestureFeatures, currentRigthGestureFeatureVec))
                
    leftGestureFeatures = leftGestureFeatures.transpose() #  n_frame * n_feature 
    rightGestureFeatures = rightGestureFeatures.transpose() #  n_frame * n_feature
    
    return  leftGestureFeatures, rightGestureFeatures


def extractRPFeature(gestureSample, frameNum):
    ''' extract relative position feature from one frame '''
    
    skeleton = gestureSample.getSkeleton(frameNum)
    worldCoordinates = skeleton.getWorldCoordinates()

    hipCenterCoordinate = worldCoordinates.get('HipCenter')

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
     
    leftTmpArr = np.vstack((leftRP1, leftRP2, leftRP3, leftRP4))
    rightTmpArr = np.vstack((rightRP1, rightRP2, rightRP3, rightRP4))
    
    leftRpFeatureVec = leftTmpArr.reshape((12, 1)) # left arm RP feature
    rightRpFeatureVec = rightTmpArr.reshape((12, 1)) # right arm RP feature
    
    return leftRpFeatureVec, rightRpFeatureVec


def extractOriFeature(gestureSample, frameNum):
    ''' extract relative orientation feature in each frame '''
    skeleton = gestureSample.getSkeleton(frameNum)
    joinOrientations = skeleton.getJoinOrientations()
        
    shoulderLeftOri = normQuaternions(joinOrientations.get('ShoulderLeft'))
    elbowLeftOri = normQuaternions(joinOrientations.get('ElbowLeft'))
    wristLeftOri = normQuaternions(joinOrientations.get('WristLeft'))
    handLeftOri = normQuaternions(joinOrientations.get('HandLeft'))
    
    shoulderRightOri = normQuaternions(joinOrientations.get('ShoulderRight'))
    elbowRightOri = normQuaternions(joinOrientations.get('ElbowRight'))
    wristRightOri = normQuaternions(joinOrientations.get('WristRight'))
    handRightOri = normQuaternions(joinOrientations.get('HandRight'))
        
    leftTmpArr = np.vstack((shoulderLeftOri, elbowLeftOri, wristLeftOri, handLeftOri))
    rightTmpArr = np.vstack((shoulderRightOri, elbowRightOri, wristRightOri, handRightOri))
    
    leftOriFeatureVec = leftTmpArr.reshape((16, 1))   # left arm ori feature
    rightOriFeatureVec = rightTmpArr.reshape((16, 1))   # right arm ori feature     
    
    return leftOriFeatureVec, rightOriFeatureVec
    

def extractHandDepthHogFeature(gestureSample, frameNum):
    ''' extract hand depth hog feature in each frame '''
    skeleton = gestureSample.getSkeleton(frameNum)
    pixelCoordinates = skeleton.getPixelCoordinates()
    
    handRegionWidth = 40
    handRegionHeight = 40
    
    handLeftPixelCoor = pixelCoordinates.get('HandLeft')
    handRightPixelCoor = pixelCoordinates.get('HandRight')
    depthImg, _ = gestureSample.getGestureRegion(frameNum)
    
    leftHandDepthRegion = depthImg[handLeftPixelCoor[1]-handRegionHeight/2 : handLeftPixelCoor[1]+handRegionHeight/2, 
                            handLeftPixelCoor[0]-handRegionWidth/2 : handLeftPixelCoor[0]+handRegionWidth/2]
    
    rightHandDepthRegion = depthImg[handRightPixelCoor[1]-handRegionHeight/2 : handRightPixelCoor[1]+handRegionHeight/2, 
                            handRightPixelCoor[0]-handRegionWidth/2 : handRightPixelCoor[0]+handRegionWidth/2]
    
    leftHandDepthHogFeatureVec = np.array([])
    rightHandDepthHogFeatureVec = np.array([])
    
    if leftHandDepthRegion.size == handRegionWidth * handRegionHeight and rightHandDepthRegion.size == handRegionWidth * handRegionHeight:
        # if hand depth region can be segmented
        
        widthGridNum = 3
        heightGridNum = 3
        orientationNum = 9
        featureDim = widthGridNum * heightGridNum * orientationNum
        
        leftHandDepthHogFeatureVec, leftHandDepthHogImg = extractHoGFeatuer(leftHandDepthRegion, widthGridNum, heightGridNum, orientationNum)
        rightHandDepthHogFeatureVec, rightHandDepthHogImg = extractHoGFeatuer(rightHandDepthRegion, widthGridNum, heightGridNum, orientationNum)
        
        leftHandDepthHogFeatureVec = leftHandDepthHogFeatureVec.reshape((featureDim, 1))
        rightHandDepthHogFeatureVec = rightHandDepthHogFeatureVec.reshape((featureDim, 1))
    
    return leftHandDepthHogFeatureVec, rightHandDepthHogFeatureVec


def extractHandGrayHogFeature(gestureSample, frameNum):
    ''' extract hand depth hog feature in each frame '''
    skeleton = gestureSample.getSkeleton(frameNum)
    pixelCoordinates = skeleton.getPixelCoordinates()
    
    handRegionWidth = 40
    handRegionHeight = 40
    
    handLeftPixelCoor = pixelCoordinates.get('HandLeft')
    handRightPixelCoor = pixelCoordinates.get('HandRight')
    rgb = gestureSample.getRGB(frameNum)
    gray = cv2.cvtColor(rgb, cv2.cv.CV_RGB2GRAY)
    
    leftHandGrayRegion = gray[handLeftPixelCoor[1]-handRegionHeight/2 : handLeftPixelCoor[1]+handRegionHeight/2, 
                            handLeftPixelCoor[0]-handRegionWidth/2 : handLeftPixelCoor[0]+handRegionWidth/2]
    
    rightHandGrayRegion = gray[handRightPixelCoor[1]-handRegionHeight/2 : handRightPixelCoor[1]+handRegionHeight/2, 
                            handRightPixelCoor[0]-handRegionWidth/2 : handRightPixelCoor[0]+handRegionWidth/2]
    
    leftHandGrayHogFeatureVec = np.array([])
    rightHandGrayHogFeatureVec = np.array([])
    
    if leftHandGrayRegion.size !=0 and rightHandGrayRegion.size != 0:
        # if hand depth region can be segmented
        widthGridNum = 3
        heightGridNum = 3
        orientationNum = 9
        featureDim = widthGridNum * heightGridNum * orientationNum
        
        leftHandGrayHogFeatureVec, leftHandGrayHogImg = extractHoGFeatuer(leftHandGrayRegion, widthGridNum, heightGridNum, orientationNum)
        rightHandGrayHogFeatureVec, rightHandGrayHogImg = extractHoGFeatuer(rightHandGrayRegion, widthGridNum, heightGridNum, orientationNum)
        
        leftHandGrayHogFeatureVec = leftHandGrayHogFeatureVec.reshape((featureDim, 1))
        rightHandGrayHogFeatureVec = rightHandGrayHogFeatureVec.reshape((featureDim, 1))
    
    return leftHandGrayHogFeatureVec, rightHandGrayHogFeatureVec