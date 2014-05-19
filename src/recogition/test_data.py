'''
Created on 08/05/2014

@author:  Bin Liang

'''
import os
import numpy as np
from utils.ChalearnLAPSample import GestureSample
from svmutil import svm_predict
import config


def processTestGestureList(testGestureList, hmmModelList, modelScalers):
    ''' Process (scale and PCA) test data '''
    
    print "Processing test gesture data..."
    
    procTestGestureList = []
    if os.path.exists("pro_test_features.npy"):
        procTestGestureList = np.load("pro_test_features.npy")
    else:
    
        staSvmScaler = modelScalers[0]
        mtmSvmScaler = modelScalers[1]
        hmmSvmScaler = modelScalers[2]
        
        # format test features from test gesture list
        test_sta_X, test_mtm_X = formatTestFeatures(testGestureList)
        
        # process sta_svm test data
        test_sta_X_scaledArr = staSvmScaler.transform(test_sta_X)
        proc_test_sta_X = test_sta_X_scaledArr.tolist()
        
        # process mtm_svm test data
        test_mtm_X_scaledArr = mtmSvmScaler.transform(test_mtm_X)
        proc_test_mtm_X = test_mtm_X_scaledArr.tolist()
        
        # extract hmm score features
        test_hmm_X = []
        for testGesture in testGestureList:
            testData = testGesture.timeDomainFeatures
            probs = []
            
            for gestureModel in hmmModelList:
                prob = gestureModel.hmmModel.score(testData)
                probs.append(prob)
            
            hmmOutputFeatures = probs
            testGesture.setHmmOutputFeatures(hmmOutputFeatures)
            
            test_hmm_X.append(testGesture.hmmOutputFeatures)
        
        test_hmm_X_scaledArr = hmmSvmScaler.transform(test_hmm_X)
        proc_test_hmm_X = test_hmm_X_scaledArr.tolist()
        
        # assign processed features to test gesture list 
        for i in xrange(len(testGestureList)):
            testGesture = testGestureList[i]
            procStaFeatures = proc_test_sta_X[i]
            procMtmFeatures = proc_test_mtm_X[i]
            procHmmOutputFeatures = proc_test_hmm_X[i]
            
            # set processed features
            testGesture.setStaFeatures(procStaFeatures) 
            testGesture.setMtmFeatures(procMtmFeatures)
            testGesture.setHmmOutputFeatures(procHmmOutputFeatures)
        
        procTestGestureList = testGestureList
        np.save("pro_test_features", procTestGestureList)
    
    return procTestGestureList
    

def formatTestFeatures(testGestureList):
    ''' format test features from test gesture list '''
    test_sta_X = []
    test_mtm_X = []
    
    for gesture in testGestureList:
        staFeatures = gesture.staFeatures.ravel()
        mtmFeatures = gesture.mtmFeatures.ravel()
        
        if np.isnan(staFeatures).any():
            print gesture.fileName, gesture.startFrame, gesture.endFrame
            
        if np.isnan(mtmFeatures).any():
            print gesture.fileName, gesture.startFrame, gesture.endFrame
        
        test_sta_X.append(staFeatures)
        test_mtm_X.append(mtmFeatures)
        
    return test_sta_X, test_mtm_X


def predictTestData(staSvmModel, mtmSvmModel, hmmSvmModel, \
                    procTestGestureList, testDataDir, outPred):
    ''' prediction over test data '''
    print "Predicting test gesture data..."
    
    # Get the list of training samples    
    samples = os.listdir(testDataDir)
    
    for samplFile in samples:
        if not samplFile.endswith(".zip"):
            continue
        
        print "Predicting", samplFile, "..."
        # Create the object to access the sample
        smp = GestureSample(os.path.join(testDataDir, samplFile))
        pred = []
        
        # find gesture belonging to the sample
        for gesture in procTestGestureList:
            if gesture.fileName == samplFile:
                staFeatures = gesture.staFeatures   # statistic skeleton features
                mtmFeatures = gesture.mtmFeatures   # mtm features
                hmmOutputFeatures = gesture.hmmOutputFeatures   # hmm output features
            
                
                # construct results table
                resultTab = np.zeros((20, 5))
                resultTab[:, 0] = range(1, 21)
            
                # recognition
                # HMM  SVM recognition
                x_hmm = [hmmOutputFeatures]
                y = [0] * len(x_hmm)
                p_label, p_acc, p_val = svm_predict(y, x_hmm, hmmSvmModel, '-b 1 -q')
                labels = hmmSvmModel.get_labels()
                
                for i in xrange(20):
                    labelId = labels[i]
                    for j in xrange(20):
                        if labelId == resultTab[j, 0]:
                            resultTab[j, 1] = p_val[0][i]
                
            
                # sta SVM recognition
                x_sta = [staFeatures]
                y = [0] * len(x_sta)
                p_label, p_acc, p_val = svm_predict(y, x_sta, staSvmModel, '-b 1 -q')
                labels = staSvmModel.get_labels()
                  
                for i in xrange(20):
                    labelId = labels[i]
                    for j in xrange(20):
                        if labelId == resultTab[j, 0]:
                            resultTab[j, 2] = p_val[0][i]
                            
                # mtm SVM recognition
                x_mtm = [mtmFeatures]
                y = [0] * len(x_mtm)
                p_label, p_acc, p_val = svm_predict(y, x_mtm, mtmSvmModel, '-b 1 -q')
                labels = mtmSvmModel.get_labels()
                  
                for i in xrange(20):
                    labelId = labels[i]
                    for j in xrange(20):
                        if labelId == resultTab[j, 0]:
                            resultTab[j, 3] = p_val[0][i]
                        
                # models fusion
                resultTab[:, 4] = config.hmmSvmModelFactor * resultTab[:, 1] + \
                                  config.staModelFactor * resultTab[:, 2] + \
                                  config.mtmModelFactor * resultTab[:, 3]
                rowNum = resultTab[:, 4].argmax()
            
                # ignore low probability
                finalProb = resultTab[rowNum, 4]
                if finalProb >= config.accuThresh:
                    predGestureId = int(resultTab[rowNum, 0])
                    gesture.gestureID = predGestureId
                    pred.append([gesture.gestureID, gesture.startFrame, gesture.endFrame])
        
        smp.exportPredictions(pred, outPred)
        
        # Remove the sample object
        del smp