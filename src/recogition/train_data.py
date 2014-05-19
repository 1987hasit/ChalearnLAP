'''
Created on 08/05/2014

@author:  Bin Liang

'''
import os
from utils.public_utils import getGestureIDListFromTrain, perfromPca
import numpy as np
from domain.gesture_domain import GestureModel
from sklearn import preprocessing
from svmutil import svm_load_model, svm_train, svm_save_model
from svm import svm_problem, svm_parameter
import config

def learnModels(trainGestureList, modelDir):
    
    # learn statistic SVM model
    staSvmModel, staSvmScaler = learnStaSvmModel(trainGestureList, config.svm_sta_c, config.svm_sta_g, modelDir)
      
    # learn mtm SVM model
    mtmSvmModel, mtmSvmScaler = learnMtmSvmModel(trainGestureList, config.svm_mtm_c, config.svm_mtm_g, modelDir)
    
    # learn HMM model
    hmmModelList = learnHmmModel(trainGestureList, config.hmm_numStates, config.hmm_numMix, modelDir)
    
    # learn hmm-svm model
    hmmSvmModel, hmmSvmScaler = learnHmmSvmModel(trainGestureList, hmmModelList, config.hmm_svm_c, config.hmm_svm_g, modelDir)
    
    # put into list
    models = []
    models.append(staSvmModel)
    models.append(mtmSvmModel)
    models.append(hmmModelList)
    models.append(hmmSvmModel)
    
    modelScalers = []
    modelScalers.append(staSvmScaler)
    modelScalers.append(mtmSvmScaler)
    modelScalers.append(hmmSvmScaler)
    
    return models, modelScalers
    

def learnStaSvmModel(trainGestureList, svm_c, svm_g, modelDir):
    ''' learn statistics svm model '''
    print "Learning sta_svm model..."
    
    svmModel = None
    modelFileName = 'sta_svm_model.model'
    scalerFileName = 'sta_svm_scaler'
    
    if os.path.exists(os.path.join(modelDir, modelFileName)) and \
       os.path.exists(os.path.join(modelDir, scalerFileName + '.npy')):
        
        svmModel = svm_load_model(os.path.join(modelDir, modelFileName))
        staSvmScaler = np.load(os.path.join(modelDir, scalerFileName + '.npy'))[0]
    else:
        train_y = []
        train_X = []
        for gesture in trainGestureList:
            gestureID = gesture.gestureID
            staFeatures = gesture.staFeatures.ravel()
            
            train_y.append(gestureID)
            train_X.append(staFeatures)
        
        # scale train data
        staSvmScaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        train_X_scaledArr = staSvmScaler.fit_transform(train_X)
        
        # learn and save svm model
        X = train_X_scaledArr.tolist()   
        problem = svm_problem(train_y, X)
        paramStr = '-c ' + str(svm_c) + ' -g ' + str(svm_g) + ' -b 1'
        param = svm_parameter(paramStr)
        svmModel = svm_train(problem, param)
         
        # save model and scaler
        svm_save_model(os.path.join(modelDir, modelFileName), svmModel)
        np.save(os.path.join(modelDir, scalerFileName), [staSvmScaler])
        
    return svmModel, staSvmScaler


def learnMtmSvmModel(trainGestureList, svm_c, svm_g, modelDir):
    ''' learn mtm svm model '''
    
    print "Learning mtm_svm model"
    
    svmModel = None
    modelFileName = 'mtm_svm_model.model'
    scalerFileName = 'mtm_svm_scaler'
    
    if os.path.exists(os.path.join(modelDir, modelFileName)) and \
       os.path.exists(os.path.join(modelDir, scalerFileName + '.npy')):
        svmModel = svm_load_model(os.path.join(modelDir, modelFileName))
        mtmSvmScaler= np.load(os.path.join(modelDir, scalerFileName + '.npy'))[0]
    else:
        train_y = []
        train_X = []
        for gesture in trainGestureList:
            gestureID = gesture.gestureID
            mtmFeatures = gesture.mtmFeatures.ravel()
            
            train_y.append(gestureID)
            train_X.append(mtmFeatures)
        
        # scale train data
        mtmSvmScaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        train_X_scaledArr = mtmSvmScaler.fit_transform(train_X)
        
#         # PCA
#         numTrainData, _ = train_X_scaledArr.shape
#         X = np.vstack((train_X_scaledArr, test_X_scaledArr))
#           
#         X_pca = perfromPca(X, 0.99)
#         train_X_pca_arr = X_pca[:numTrainData, :]
#         test_X_pca_arr = X_pca[numTrainData:, :]
#         
#         # learn rbf-svm model           
#         problem = svm_problem(train_y, train_X_pca_arr.tolist())
#         paramStr = '-c ' + str(svm_c) + ' -g ' + str(svm_g) + ' -b 1'
#         param = svm_parameter(paramStr)
#         svmModel = svm_train(problem, param)
#         svm_save_model(mtmModelFileName, svmModel)

        # learn linear-svm model           
        problem = svm_problem(train_y, train_X_scaledArr.tolist())
        paramStr = '-c ' + str(svm_c) + ' -g ' + str(svm_g) + ' -t 0 ' + ' -b 1'
        param = svm_parameter(paramStr)
        svmModel = svm_train(problem, param)
        
        # save svm model and scaler
        svm_save_model(os.path.join(modelDir, modelFileName), svmModel)
        np.save(os.path.join(modelDir, scalerFileName), [mtmSvmScaler])
        
        
    return svmModel, mtmSvmScaler


def learnHmmModel(trainGestureList, numStates, numMix, modelDir):
    
    print "Learning hmm model..."
    gestureModelList = []
    gestureIDList = getGestureIDListFromTrain(trainGestureList)
    
    modelFileName = 'hmm_model'
    
    if os.path.exists(os.path.join(modelDir, modelFileName + '.npy')):
        gestureModelList = np.load(os.path.join(modelDir, modelFileName + '.npy'))
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
            gestureModel.initModelParam(nStates = numStates, nMix = numMix, \
                                        covarianceType = 'diag', n_iter = 10)
            gestureModel.trainHMMModel()
        
        # save model
        np.save(os.path.join(modelDir, modelFileName), gestureModelList)
    
    return gestureModelList


def learnHmmSvmModel(trainGestureList, hmmModelList, svm_c, svm_g, modelDir):
    
    print "Learning hmm_svm model..."
    
    svmModel = None
    
    modelFileName = 'hmm_svm_model.model'
    scalerFileName = 'hmm_svm_scaler'
    
    if os.path.exists(os.path.join(modelDir, modelFileName)) and \
       os.path.exists(os.path.join(modelDir, scalerFileName + '.npy')):
        svmModel = svm_load_model(os.path.join(modelDir, modelFileName))
        hmmSvmScaler = np.load(os.path.join(modelDir, scalerFileName + '.npy'))[0]
    else:
    
        train_y = []
        train_X = []
        
        for trainGesture in trainGestureList:
            testData = trainGesture.timeDomainFeatures
            probs = []
            
            for gestureModel in hmmModelList:
                prob = gestureModel.hmmModel.score(testData)
                probs.append(prob)
            
            hmmOutputFeatures = probs
            trainGesture.setHmmOutputFeatures(hmmOutputFeatures)
            
            train_y.append(trainGesture.gestureID)
            train_X.append(trainGesture.hmmOutputFeatures)
            
        
        # scale train data
        hmmSvmScaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        train_X_scaledArr = hmmSvmScaler.fit_transform(train_X)
        
        # learn and save svm model
        X = train_X_scaledArr.tolist()   
        problem = svm_problem(train_y, X)
        paramStr = '-c ' + str(svm_c) + ' -g ' + str(svm_g) + ' -b 1'
        param = svm_parameter(paramStr)
        svmModel = svm_train(problem, param)
        
        # save model and scaler
        # save model and scaler
        svm_save_model(os.path.join(modelDir, modelFileName), svmModel)
        np.save(os.path.join(modelDir, scalerFileName), [hmmSvmScaler])
        
    return svmModel, hmmSvmScaler


def loadModels(modelDir):
    ''' load models for recognition '''
    print "Loading models..."
    
    staSvmModel = svm_load_model(os.path.join(modelDir, 'sta_svm_model.model'))
    staSvmScaler = np.load(os.path.join(modelDir, 'sta_svm_scaler.npy'))[0]
    
    mtmSvmModel = svm_load_model(os.path.join(modelDir, 'mtm_svm_model.model'))
    mtmSvmScaler = np.load(os.path.join(modelDir, 'mtm_svm_scaler.npy'))[0]
    
    hmmModelList = np.load(os.path.join(modelDir,  'hmm_model.npy'))
    
    hmmSvmModel = svm_load_model(os.path.join(modelDir, 'hmm_svm_model.model'))
    hmmSvmScaler = np.load(os.path.join(modelDir, 'hmm_svm_scaler.npy'))[0]
    
    # put into list
    models = []
    models.append(staSvmModel)
    models.append(mtmSvmModel)
    models.append(hmmModelList)
    models.append(hmmSvmModel)
    
    modelScalers = []
    modelScalers.append(staSvmScaler)
    modelScalers.append(mtmSvmScaler)
    modelScalers.append(hmmSvmScaler)
    
    return models, modelScalers