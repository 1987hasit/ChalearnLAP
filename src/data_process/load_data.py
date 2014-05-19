'''
Created on 08/05/2014

@author:  Bin Liang

'''
import os
import numpy as np
import gc
from extraction.feature_extraction import extractStaFeature, extractMtmFeature, extractTdFeature
from representation.gesture_rep import get2DMTM
from domain.gesture_domain import Gesture
from utils.ChalearnLAPSample import GestureSample


def loadTrainData(trainDataDir):
    ''' load training data from train dir and construct a training data list '''
    trainGestureList = []
    if os.path.exists("train_features.npy"):
        trainGestureList = np.load("train_features.npy")
    else:
        trainFeatureDir = './train_features/'
        trainGestureList = generateTrainGestureListWithFeatures(trainDataDir, trainFeatureDir) # construct training gesture list using extracted features
        np.save("train_features", trainGestureList)
    
    return trainGestureList


def generateTrainGestureListWithFeatures(gestureDataDir, trainFeatureDir):
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
        
        featureFileName = os.path.splitext(samplFile)[0] + '_features'
        if os.path.exists(os.path.join(trainFeatureDir, featureFileName + '.npy')):
            smpGestureList = np.load(os.path.join(trainFeatureDir, featureFileName + '.npy')).tolist()
            gestureList += smpGestureList
            
            # free memory
            del smpGestureList
            gc.collect()
            
            continue
        
        print "\t Extracting features from file", samplFile
        
        # Create the object to access the sample
        smp = GestureSample(os.path.join(gestureDataDir, samplFile))
        
        # Get the list of gestures for this sample
        smpGestureList = smp.getGestures()
        noDataGestureIdxList = []
        
        # Iterate for each gesture in this sample
        for gesture in smpGestureList:
            # extract features from skeleton
            
            # time domain features
            timeDomainFeatures = extractTdFeature(gesture, smp)
            
            # some gesture sample contains no skeleton data
            if timeDomainFeatures.size == 0:
                # record index
                noDataGestureIdxList.append(smpGestureList.index(gesture))
                continue
            
            # statistic skeleton features
            staFeatures = extractStaFeature(timeDomainFeatures)
            
            # D-MHI
            two_d_mtm = get2DMTM(gesture, smp)
#             gesture.set2DMTM(two_d_mtm)
            
            # HOG feature from 2DMTM
            mtmFeatures = extractMtmFeature(two_d_mtm)
            
            gesture.setTimeDomainFeatures(timeDomainFeatures)
            gesture.setStaFeatures(staFeatures)
            gesture.setMtmFeatures(mtmFeatures)
        
        
        # delete gesture with no skeleton data    
        if len(noDataGestureIdxList) != 0:
            tmpSmpGestureList = []
            for gesture in smpGestureList:
                if not smpGestureList.index(gesture) in noDataGestureIdxList:
                    tmpSmpGestureList.append(gesture)
            smpGestureList = tmpSmpGestureList
                 
        # save sample gesture
        if not os.path.exists(trainFeatureDir):
            os.makedirs(trainFeatureDir)
        np.save(os.path.join(trainFeatureDir, featureFileName), smpGestureList)
        gestureList += smpGestureList
        
        # free memory
        del smpGestureList, smp
        gc.collect()
            
    return gestureList


def loadTestData(testDataDir, labelsDict):
    ''' load test data from test dir and construct a test data list '''
    testGestureList = []
    if os.path.exists("test_features.npy"):
        testGestureList = np.load("test_features.npy")
    else:
        testDir = './test_features/'
        testGestureList = generateTestGestureListWithFeatures(testDataDir, testDir, labelsDict) # construct training gesture list using extracted features
        np.save("test_features", testGestureList)
    
    return testGestureList


def generateTestGestureListWithFeatures(testDataDir, testFeatureDir, labelsDict):
    """ Generate test gesture list with extracted features and predicted labels """
    print "Extracting features from test data"
    
    # Get the list of data samples    
    samples = os.listdir(testDataDir)
    
    # Initialize the gestures
    gestureList = []
    
    # Access to each sample
    for samplFile in samples:
        if not samplFile.endswith(".zip"):
            continue
        
        featureFileName = os.path.splitext(samplFile)[0] + '_features'
        if os.path.exists(os.path.join(testFeatureDir, featureFileName + '.npy')):
            smpGestureList = np.load(os.path.join(testFeatureDir, featureFileName + '.npy')).tolist()
            gestureList += smpGestureList
            continue
        
        print "\t Extracting features from file", samplFile
        
        # Create the object to access the sample
        smp = GestureSample(os.path.join(testDataDir, samplFile))
        
        # Get the list of gestures for this sample
        smpGestureList = []
        labels = labelsDict[smp.seqID] 
                 
        for label in labels:
            startFrame = label[0]
            endFrame = label[1]
            
            gestureID = 0   # test label for test data
            fileName = os.path.split(samplFile)[1]
            gesture = Gesture(gestureID, startFrame, endFrame, fileName)             
            smpGestureList.append(gesture)
        
        noDataGestureIdxList = []         
        # Iterate for each gesture in this sample
        for gesture in smpGestureList:
            # time domain features
            timeDomainFeatures = extractTdFeature(gesture, smp)
            
            # some gesture sample contains no skeleton data
            if timeDomainFeatures.size == 0:
                # record index
                noDataGestureIdxList.append(smpGestureList.index(gesture))
                continue
            
            # statistic skeleton features
            staFeatures = extractStaFeature(timeDomainFeatures)
            
            if np.isnan(staFeatures).any():
                print gesture.fileName, gesture.startFrame, gesture.endFrame
            
            # D-MHI
            two_d_mtm = get2DMTM(gesture, smp)
            
            # HOG feature from 2DMTM
            mtmFeatures = extractMtmFeature(two_d_mtm)
            
            gesture.setTimeDomainFeatures(timeDomainFeatures)
            gesture.setStaFeatures(staFeatures)
            gesture.setMtmFeatures(mtmFeatures)
            
        # delete gesture with no skeleton data    
        if len(noDataGestureIdxList) != 0:
            tmpSmpGestureList = []
            for gesture in smpGestureList:
                if not smpGestureList.index(gesture) in noDataGestureIdxList:
                    tmpSmpGestureList.append(gesture)
            smpGestureList = tmpSmpGestureList
                
        
        # save sample gesture
        if not os.path.exists(testFeatureDir):
            os.makedirs(testFeatureDir)
        np.save(os.path.join(testFeatureDir, featureFileName), smpGestureList)
        gestureList += smpGestureList
            
    return gestureList