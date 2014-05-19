'''
Created on 08/05/2014

@author:  Bin Liang

'''
from sklearn import preprocessing
from utils.public_utils import perfromPca, getGestureIDListFromTrain
import os
from sklearn.cross_validation import KFold
import numpy as np
from numpy import mean
from domain.gesture_domain import GestureModel
from utils.svm_tool import SvmTool
import config

def corssValidate(trainGestureList):
    ''' cross validate on training data '''
#     svm_sta_c, svm_sta_g = svmStaCrossValidate(trainGestureList)
#     numStates = hmmCrossValidate(trainGestureList)
#     svm_mtm_c, svm_mtm_g = svmMtmCrossValidate(trainGestureList)
    svmHmmCrossValidate(trainGestureList)
    

def svmMtmCrossValidate(trainGestureList):
    ''' libsvm cross validation for mtm features'''
    y = []
    X = []
    for gesture in trainGestureList:
        gestureID = gesture.gestureID
        mtmFeatures = gesture.mtmFeatures.ravel()   # transpose to row vector
        
        y.append(gestureID)
        X.append(mtmFeatures)
    
    # scale data
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_scaledArr = min_max_scaler.fit_transform(X)
    
    # pca
    X_pca = perfromPca(X_scaledArr, 0.99)
    X_scaled = X_pca.tolist() # convert array to list
    
    # write to svm format file
    svmTool = SvmTool()
    outputPath = config.cvMtmOutputPath
    fileName = config.cvMtmFileName
    svmTool.write2SVMFormat(outputPath, fileName, X_scaled, y)
    
    # cross validation using grid.py
    kernelType = 'RBF'   # 'RBF', 'linear'
    c, g, r = svmTool.gridsearch(os.path.join(outputPath, fileName), kernelType)
    
    return c, g


def svmStaCrossValidate(trainGestureList):
    ''' libsvm cross validation for statistic features '''
    
    y = []
    X = []
    for gesture in trainGestureList:
        gestureID = gesture.gestureID
        staFeatures = gesture.staFeatures.ravel()   # transpose to row vector
        
        y.append(gestureID)
        X.append(staFeatures)
         
    # scale data
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_scaledArr = min_max_scaler.fit_transform(X)
    
#     # pca
#     X_pca = perfromPca(X_scaledArr, 0.99)
#     X_scaled = X_pca.tolist() # convert array to list
    
    X_scaled = X_scaledArr.tolist()
    # write to svm format file
    svmTool = SvmTool()
    outputPath = config.cvStaOutputPath
    fileName = config.cvStaFileName
    svmTool.write2SVMFormat(outputPath, fileName, X_scaled, y)
    
    # cross validation using grid.py
    kernelType = 'RBF'   # 'RBF', 'linear'
    c, g, r = svmTool.gridsearch(os.path.join(outputPath, fileName), kernelType)
    
    return c, g


def hmmCrossValidate(trainGestureList):
    ''' get best parameters through cross validation '''
    
#     numStatesList = range(3, 20)    # number of states: 3-20
    numStatesList = [15]
    n_folds = 5
    kf = KFold(len(trainGestureList), n_folds, indices = False)
    
    cvScoreArr = np.zeros((n_folds, len(numStatesList)))
    
    runningFold = 0
    for cvTrain, cvTest in kf:
        
        cvTrainGestureList = trainGestureList[cvTrain]
        cvTestGestureList = trainGestureList[cvTest]
        
        for s in xrange(len(numStatesList)):
            numStates = numStatesList[s]
            
            cvModelList = cvLearnModel(cvTrainGestureList, numStates)
            accuracy = cvPredict(cvModelList, cvTestGestureList)
            
            cvScoreArr[runningFold, s] = accuracy
        
        runningFold += 1
        
    meanScoreArr = mean(cvScoreArr, axis = 0)
    
    maxIdx = meanScoreArr.argmax()
    bestNumStates = numStatesList[maxIdx]
    
    return bestNumStates 


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
            gestureModel.initModelParam(nStates = numStates, nMix = config.hmm_numMix, \
                                        covarianceType = 'diag', n_iter = 10)
            gestureModel.trainHMMModel()

    return gestureModelList
    
    
def cvPredict(cvModelList, cvTestGestureList):
    
        numTestGestures = len(cvTestGestureList)
        score = 0
        
        # Iterate for each gesture in this sample
        for gesture in cvTestGestureList:
            
            probs = []  # probability with models
            
            for gestureModel in cvModelList:
                testData = gesture.timeDomainFeatures
                prob = gestureModel.hmmModel.score(testData)
                probs.append(prob)
                
            idx = probs.index(max(probs))
            predGestureId = cvModelList[idx].gestureID
            
            if gesture.gestureID == predGestureId:
                score += 1
        
        accuracy = float(score) / float(numTestGestures)
        
        return accuracy
    
def svmHmmCrossValidate(trainGestureList):
    
    hmmModelList = cvLearnModel(trainGestureList, config.hmm_numStates)
    y = []
    X = []
    
    for gesture in trainGestureList:
        testData = gesture.timeDomainFeatures
        probs = []
        
        for gestureModel in hmmModelList:
                prob = gestureModel.hmmModel.score(testData)
                probs.append(prob)
        
        hmmOutputFeatures = probs
        gesture.setHmmOutputFeatures(hmmOutputFeatures)
        
        y.append(gesture.gestureID)
        X.append(gesture.hmmOutputFeatures)
        
    # scale data
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X_scaledArr = min_max_scaler.fit_transform(X)
    X_scaled = X_scaledArr.tolist()
    
    # write to svm format file
    svmTool = SvmTool()
    outputPath = config.cvHmmSvmOutputPath
    fileName = config.cvHmmSvmFileName
    svmTool.write2SVMFormat(outputPath, fileName, X_scaled, y)