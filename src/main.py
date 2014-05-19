# -*- coding:utf-8 -*-  

'''
Created on 24/03/2014

@author: Bin Liang
'''

import os.path, zipfile

from utils.ChalearnLAPEvaluation import evalGesture, exportGT_Gesture
import gc
from validation.cross_validation import corssValidate
from recogition.train_data import learnModels
from data_process.load_data import loadTrainData, loadTestData
from recogition.test_data import predictTestData, processTestGestureList
import config
from data_process.segment_sample import getTestGestureLabels


def main():
    """ Main script """
    
    ## Dirs claiming
    ##-------------------------------------------------------------------------
    # Training data folder
    trainDataDir = config.trainDataDir
    
    # out model folder
    modelDir = config.modelDir
    
    # Test data folder
    testDataDir = config.testDataDir

    # Ground truth folder
    outGT = config.outGT
    
    # Predictions folder (output)
    outPred = config.outPred
    
    # Submission folder (output)
    outSubmision = config.outSubmision
    
    
    ## Data loading and cross validation
    ##-------------------------------------------------------------------------
    # Load training data 
    trainGestureList = loadTrainData(trainDataDir)
     
    # Cross validation
    if config.cv:
        corssValidate(trainGestureList)
     
     
    ## Models learning
    ##-------------------------------------------------------------------------
    models, modelScalers = learnModels(trainGestureList, modelDir)
    # release memory
    del trainGestureList
    gc.collect()
     
    staSvmModel = models[0]
    mtmSvmModel = models[1]
    hmmModelList = models[2]
    hmmSvmModel = models[3]
     
    # Segment unlabeled test gesture samples
    labelsDict = getTestGestureLabels(testDataDir)
        
    # Load test data
    testGestureList = loadTestData(testDataDir, labelsDict)
        
    ## Recognition
    ##--------------------------------------------------------------------------
    # Process (scale and PCA) test data  
    procTestGestureList = processTestGestureList(testGestureList, hmmModelList, modelScalers)
       
    # release memory
    del testGestureList
    gc.collect()
        
    # Prediction
    predictTestData(staSvmModel, mtmSvmModel, hmmSvmModel, \
                    procTestGestureList, testDataDir, outPred)
      
      
    ## Evaluation prediction
    ##--------------------------------------------------------------------------
    # Create evaluation gt from labeled data
#     exportGT_Gesture(testDataDir, outGT)

    # Evaluate your predictions
    score = evalGesture(outPred, outGT)
    print("The score for this prediction is " + "{:.12f}".format(score))
    
    # Prepare submision file (only for validation and final evaluation data sets)
    createSubmisionFile(outPred, outSubmision)


def createSubmisionFile(predictionsPath,submisionPath):
    """ Create the submission file, ready to be submitted to Codalab. """

    # Create the output path and remove any old file
    if os.path.exists(submisionPath):
        oldFileList = os.listdir(submisionPath)
        for ifile in oldFileList:
            os.remove(os.path.join(submisionPath,ifile))
    else:
        os.makedirs(submisionPath)

    # Create a ZIP with all files in the predictions path
    zipf = zipfile.ZipFile(os.path.join(submisionPath,'Submission.zip'), 'w')
    for root, dirs, files in os.walk(predictionsPath):
        for ifile in files:
            zipf.write(os.path.join(root, ifile), ifile, zipfile.ZIP_DEFLATED)
    zipf.close()


if __name__ == '__main__':
    main()
