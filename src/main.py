# -*- coding:utf-8 -*-  

'''
Created on 24/03/2014

@author: Bin Liang
'''

import os.path, zipfile
import numpy as np

from utils.ChalearnLAPEvaluation import evalGesture, exportGT_Gesture
from utils.ChalearnLAPSample import GestureSample
from extraction.feature_extraction import getGestureType,\
    getGestureTypeForGesture, extractGestureFeature
import matplotlib.pyplot as plt
from utils.public_utils import groupConsecutives, removeall
from domain.gesture_domain import GestureModel, Gesture
from common import SEGMENT_WINDOW_SIZE, GestureType
import itertools
from numpy import mean
from sklearn.cross_validation import KFold
from shutil import copyfile


def main():
    """ Main script """
    ###########################################################################
    # Step.0 Preparation    
    # Training data folder
    trainDataDir='D:/Research/Projects/Dataset/ChalearnLAP/Training data/';
    # Test data folder
    testDataDir='D:/Research/Projects/Dataset/ChalearnLAP/Validation data2/'
    # Predictions folder (output)
    outPred='./pred/'
    # Submision folder (output)
    outSubmision='./submision/'
    
    ###########################################################################    
    # Step.1 Cross validation
    #bestNumStates = crossValidate(trainDataDir)
    bestNumStates = 10
    
    ###########################################################################
    # Step.2 Split unlabeled test gesture samples
    labelsDict = getTestGestureLabels(testDataDir)
    
    ###########################################################################
    # Step.3 Load training data 
    trainGestureList = loadTrainData(trainDataDir)
    
    ###########################################################################    
    # Step.4 Learn gesture model
    gestureModelList = learnModel(trainGestureList, bestNumStates)

    ###########################################################################
    # Step.5 Predict over test data 
    predictTestData(gestureModelList, testDataDir, labelsDict, outPred)
    
    # Prepare submision file (only for validation and final evaluation data sets)
    createSubmisionFile(outPred, outSubmision);


### Step.2 Test gesture segmentation 
def getTestGestureLabels(testDataDir):
    ''' Split unlabeled gesture samples and save the files '''
    
    # label dictionary
    labelsDict = {}
    
    if os.path.exists("pred_labels.npy"):
        labelsDict = np.load("pred_labels.npy")
        return labelsDict.item()    # convert to dictionary
    
    # Get the list of training samples    
    samples = os.listdir(testDataDir)
    
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
        for i in range(1, numFrames + 1):
            
            # if out of the sequence, break the loop
            if i + SEGMENT_WINDOW_SIZE - 1 > numFrames:
                break
            
            skel = smp.getSkeleton(i)            
            hipCenterCors = skel.getWorldCoordinates().get('HipCenter')
            
            # decide the first frame number
            if firstFrameNum == 0:
                if np.sum(hipCenterCors) == 0:
                    continue
                else:
                    firstFrameNum = i
                          
            gestureType = getGestureType(i, SEGMENT_WINDOW_SIZE, smp)
            gestureTypeList.append(gestureType)
        
        # insert blank frames into type list
        for f in range(1, firstFrameNum):
            gestureTypeList.insert(0, 0)
            
        # get labels
        labels = smp.getLabels()
        labelList = []
        
        # if the ground truth labels exist
        if len(labels) > 0:
            labelArray = np.array(labels)
            
            for row in labelArray:
                labelList.append(row[1])
                labelList.append(row[2])
        
        processedGestureTypeList = processGestureTypeList(gestureTypeList)
        predLabelList = getLabelListFromProcessedGestureTypeList(processedGestureTypeList)
                
#         # show predict label result 
#         dsipLabelList = list(itertools.chain.from_iterable(predLabelList))    
#         
#         # show processed gesture type list
#         dsipGestureTypeList = list(itertools.chain.from_iterable(processedGestureTypeList))
#               
#         plotScoreList(dsipGestureTypeList, labelList)
            
        # save pred_labels.npy
        labelsDict[smp.seqID] = predLabelList
    
    np.save("pred_labels", labelsDict)
    return labelsDict


### Step.3 Load training data 
def loadTrainData(trainDataDir):
    ''' load training data from train dir and construct a training data list '''
    trainGestureList = []
    if os.path.exists("train_features.npy"):
        trainGestureList = np.load("train_features.npy")
    else:
        trainGestureList = generateGestureListWithFeatures(trainDataDir) # construct training gesture list using extracted features
        np.save("train_features", trainGestureList)
    
    return trainGestureList


# Step.1 Cross validation
def crossValidate(trainDataDir):
    ''' get best parameters through cross validation '''
    
    cvTrainDir = './cv/train/'
    cvTestDir = './cv/test/'
    cvPredDir = './cv/pred/'
    cvGtDir = './cv/gt/'
    
    # Get the data files
    fileList = os.listdir(trainDataDir);

    # Filter input files (only ZIP files)
    sampleList = []
    for file in fileList:
        if file.endswith(".zip"):
            sampleList.append(file)
    
    numStatesList = range(3, 11)    # number of states: 3-10
    n_folds = 5
    kf = KFold(len(sampleList), n_folds, indices = False)
    
    cvScoreArr = np.zeros((n_folds, len(numStatesList)))
    sampleArr = np.array(sampleList)
    
    runningFold = 0
    for cvTrain, cvTest in kf:
        cvResetDir(cvTrainDir)
        cvResetDir(cvTestDir)
        
        cvTrainSampleList = sampleArr[cvTrain].tolist()
        cvTestSampleList = sampleArr[cvTest].tolist()
        createCvDataSets(trainDataDir, cvTrainDir, cvTestDir, cvTrainSampleList, cvTestSampleList)
        
        cvTrainGestureList = generateGestureListWithFeatures(cvTrainDir)
        
        for s in range(len(numStatesList)):
            numStates = numStatesList[s]
            cvResetDir(cvPredDir)
            cvResetDir(cvGtDir)
            
            cvModelList = cvLearnModel(cvTrainGestureList, numStates)
            cvPredict(cvModelList, cvTestDir, cvPredDir)
            exportGT_Gesture(cvTestDir, cvGtDir)
            score = evalGesture(cvPredDir, cvGtDir)
            
            cvScoreArr[runningFold, s] = score
        
        runningFold += 1
        
    meanScoreArr = mean(cvScoreArr, axis = 0)
    
    maxIdx = meanScoreArr.argmax()
    bestNumStates = numStates[maxIdx]
    
    return bestNumStates 


def cvResetDir(dirName):
    ''' if exist, clear all files in the dir 
        if not, create the dir
    '''
    if os.path.exists(dirName):
        removeall(dirName)
    else:
        os.makedirs(dirName)
    

def createCvDataSets(trainDataDir, cvTrainDir, cvTestDir, cvTrainSampleList, cvTestSampleList):
    """ Divide input samples into Train and Test sets using cross validation"""
            
    # Create the output cv train dir
    if os.path.exists(cvTrainDir):
        removeall(cvTrainDir)
    else:
        os.makedirs(cvTrainDir)

    # Create the output cv test dir 
    if os.path.exists(cvTestDir):
        removeall(cvTrainDir)
    else:
        os.makedirs(cvTestDir)

    # Copy the cv train files
    for file in cvTrainSampleList:
        copyfile(os.path.join(trainDataDir, file), os.path.join(cvTrainDir, file))
    
    # Copy the cv test files
    for file in cvTestSampleList:
        copyfile(os.path.join(trainDataDir, file), os.path.join(cvTestDir, file))

        
# Step.1.1 Learn model in cross validation
def cvLearnModel(cvTrainGestureList, numStates):
    gestureModelList = []
    gestureIDList = getGestureIDListFromTrain(cvTrainGestureList)
    
    for gestureID in gestureIDList:
        gestureModel = GestureModel(gestureID)
        gestureModelList.append(gestureModel)
    
    # organize data into the same gesture id
    for gestureModel in gestureModelList:
        
        sameIDGestureList = []  # gesture list with the same gesture id
        
        for gesture in cvTrainGestureList:
            if gestureModel.gestureID == gesture.gestureID:
                sameIDGestureList.append(gesture)
            
        gestureModel.getTrainData(sameIDGestureList)
    
    # train model
    for gestureModel in gestureModelList:
        gestureModel.numStates = numStates
        gestureModel.trainHMMModel()

    return gestureModelList
    
    
# Step.1.2 Predict in cross validation
def cvPredict(cvModelList, cvTestDir, cvPredDir):
    
    # Get the list of training samples    
    samples = os.listdir(cvTestDir)
    
    for samplFile in samples:
        if not samplFile.endswith(".zip"):
            continue
        
        # Create the object to access the sample
        smp = GestureSample(os.path.join(cvTestDir, samplFile))
        pred = []
        
        gestureList = smp.getGestures()
    
        # Iterate for each gesture in this sample
        for gesture in gestureList:
            # Extract features from one gesture
            gestureType = getGestureTypeForGesture(gesture, smp)
            gesture.gestureType = gestureType
            
            leftGestureFeatures, rightGestureFeatures = extractGestureFeature(gesture, smp)
            gesture.setLeftGestureFeatures(leftGestureFeatures)
            gesture.setRightGestureFeatures(rightGestureFeatures)
            
            probs = []  # probability with models
            
            typeGestureModelList = []
            
            if gesture.gestureType == GestureType.left_hand:
                gestureType = GestureType.single_hand
                testData = gesture.leftGestureFeatures
            elif gesture.gestureType == GestureType.right_hand:
                gestureType = GestureType.single_hand
                testData = gesture.rightGestureFeatures
            elif gesture.gestureType == GestureType.both_hands:
                gestureType = GestureType.both_hands
                testData = np.hstack((gesture.leftGestureFeatures, gesture.rightGestureFeaturess))
            
            for gesureModel in cvModelList:
                # select gesture models with the same gesture type
                if gesureModel.gestureType == gestureType:
                    typeGestureModelList.append(gesureModel)
                
            for gestureModel in typeGestureModelList:
                prob = gestureModel.hmmModel.score(testData)
                probs.append(prob)
            
            idx = probs.index(max(probs))
            predGestureId = typeGestureModelList[idx].gestureID
            gesture.gestureID = predGestureId
            
            pred.append([gesture.gestureID, gesture.startFrame, gesture.endFrame])
        
        smp.exportPredictions(pred, cvPredDir)
        
        # Remove the sample object
        del smp
    

### Step.5 Learn model
def learnModel(trainGestureList, bestNumStates):
    gestureModelList = []
    gestureIDList = getGestureIDListFromTrain(trainGestureList)
    
    if os.path.exists("model.npy"):
        gestureModelList = np.load("model.npy")
    else:
        
        for gestureID in gestureIDList:
            gestureModel = GestureModel(gestureID)
            gestureModelList.append(gestureModel) 
        
        # organize data into the same gesture id
        for gestureModel in gestureModelList:
            
            sameIDGestureList = []  # gesture list with the same gesture id
            
            for gesture in trainGestureList:
                if gestureModel.gestureID == gesture.gestureID:
                    sameIDGestureList.append(gesture)
                
            gestureModel.getTrainData(sameIDGestureList)
            
        for gestureModel in gestureModelList:
            gestureModel.initModelParam(nStates = bestNumStates, nMix = 2, \
                                        covarianceType = 'diag', n_iter = 10, bakisLevel = 2)
            gestureModel.trainHMMModel()
            
        np.save("model", gestureModelList);
    
    return gestureModelList


### Step.6 Prediction
def predictTestData(gestureModelList, testDataDir, labelsDict, outPred):
    ''' prediction over test data '''
    
    # Get the list of training samples    
    samples = os.listdir(testDataDir)
    
    for samplFile in samples:
        if not samplFile.endswith(".zip"):
            continue
        
        # Create the object to access the sample
        smp = GestureSample(os.path.join(testDataDir, samplFile))
        pred = []
        
        gestureList = []
        labels = labelsDict[smp.seqID] 
                 
        for label in labels:
            startFrame = label[0]
            endFrame = label[1]
            
            #
            gestureLen = endFrame - startFrame + 1
            if gestureLen < 20:
                continue
            #
            
            gestureID = ''
            fileName = os.path.split(samplFile)[1]
            gesture = Gesture(gestureID, startFrame, endFrame, fileName)             
            gestureList.append(gesture)
            
        
        # Iterate for each gesture in this sample
        for gesture in gestureList:
            # Extract features from one gesture
            gestureType = getGestureTypeForGesture(gesture, smp)
            gesture.gestureType = gestureType
            
            leftGestureFeatures, rightGestureFeatures = extractGestureFeature(gesture, smp)
            gesture.setLeftGestureFeatures(leftGestureFeatures)
            gesture.setRightGestureFeatures(rightGestureFeatures)
            
            probs = []  # probability with models
            
            typeGestureModelList = []
            
            if gesture.gestureType == GestureType.left_hand:
                gestureType = GestureType.single_hand
                testData = gesture.leftGestureFeatures
                
            elif gesture.gestureType == GestureType.right_hand:
                gestureType = GestureType.single_hand
                testData = gesture.rightGestureFeatures
                
            else:
                gestureType = GestureType.both_hands
                testData = np.hstack((gesture.leftGestureFeatures, gesture.rightGestureFeatures))
            
            for gesureModel in gestureModelList:
                # select gesture models with the same gesture type
                if gesureModel.gestureType == gestureType:
                    typeGestureModelList.append(gesureModel)
                
            for gestureModel in typeGestureModelList:
                prob = gestureModel.hmmModel.score(testData)
                probs.append(prob)
            
            probsArr = np.array(probs)
            idx = probsArr.argmax()
            predGestureId = typeGestureModelList[idx].gestureID
            gesture.gestureID = predGestureId
            
            pred.append([gesture.gestureID, gesture.startFrame, gesture.endFrame])
        
        smp.exportPredictions(pred, outPred)
        
        # Remove the sample object
        del smp
        

# Self added
def generateGestureListWithFeatures(gestureDataDir):
    """ Generate gesture list with extracted features """
    print "Extracting features from training data"
    
    # Get the list of data samples    
    samples = os.listdir(gestureDataDir)
    
    # Initialize the gestures
    gestureList = []
    
    # Access to each sample
    for samplFile in samples:
        if not samplFile.endswith(".zip"):
            continue
        
        print "\t Extracting features from file", samplFile
        
        # Create the object to access the sample
        smp = GestureSample(os.path.join(gestureDataDir, samplFile))
        
        # Get the list of gestures for this sample
        smpGestureList = smp.getGestures()
        
        # Iterate for each gesture in this sample
        for gesture in smpGestureList:
            # extract features from skeleton
            
            gestureType = getGestureTypeForGesture(gesture, smp)
            gesture.gestureType = gestureType
            
            leftGestureFeatures, rightGestureFeatures = extractGestureFeature(gesture, smp)
            
            gesture.setLeftGestureFeatures(leftGestureFeatures)
            gesture.setRightGestureFeatures(rightGestureFeatures)
            
            gestureList.append(gesture)
            
    return gestureList


# Self added
def getGestureIDListFromTrain(trainGestureList):
    ''' get gesture id list from training gesture list '''
    # Initialize gesture_id list
    gestureIDList = []
    
    for gesture in trainGestureList:
        gestureIDList.append(gesture.getGestureID())

    return list(set(gestureIDList))


# Self added
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


# Self added
def processGestureTypeList(gestureTypeList):
    ''' process gesture type list, remove sudden pulse '''
    
    thresh_interval = 10
    
    groupedGestureTypeList = groupConsecutives(gestureTypeList, 0)
    
    # remove extreme short interval
    for i in range(len(groupedGestureTypeList)):
        gestureInterval = groupedGestureTypeList[i]
        if len(gestureInterval) < thresh_interval:
            if i - 1 > 0:
                for j in range(len(gestureInterval)):
                    groupedGestureTypeList[i][j] = groupedGestureTypeList[i - 1][0]
            else:
                for j in range(len(gestureInterval)):
                    groupedGestureTypeList[i][j] = groupedGestureTypeList[i + 1][0]
    
    groupedGestureTypeList = list(itertools.chain.from_iterable(groupedGestureTypeList))
    processedGroupedGestureTypeList = groupConsecutives(groupedGestureTypeList, 0)
    
    return processedGroupedGestureTypeList
        
        
def getLabelListFromProcessedGestureTypeList(groupedGestureTypeList):
    ''' get label list from  processed gesture type list '''
    
    labelList = []
    realIdx = 0
    for i in range(len(groupedGestureTypeList)):
        
        gestureInterval = groupedGestureTypeList[i]
        
        if i != 0: 
            realIdx += len(groupedGestureTypeList[i-1])
        
        if (mean(gestureInterval) != GestureType.rest) and (len(gestureInterval) > 10):
            # if gesture type is not rest and the gesture interval is larger than 10
            startIdx = realIdx + 1  # frame number is larger than index number
            endIdx = startIdx + len(gestureInterval) - 1
            labelList.append([startIdx, endIdx])
            
    return labelList

def createSubmisionFile(predictionsPath,submisionPath):
    """ Create the submission file, ready to be submited to Codalab. """

    # Create the output path and remove any old file
    if os.path.exists(submisionPath):
        oldFileList = os.listdir(submisionPath);
        for file in oldFileList:
            os.remove(os.path.join(submisionPath,file));
    else:
        os.makedirs(submisionPath);

    # Create a ZIP with all files in the predictions path
    zipf = zipfile.ZipFile(os.path.join(submisionPath,'Submission.zip'), 'w');
    for root, dirs, files in os.walk(predictionsPath):
        for file in files:
            zipf.write(os.path.join(root, file), file, zipfile.ZIP_DEFLATED);
    zipf.close()


if __name__ == '__main__':
    main()
