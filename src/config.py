'''
Created on 13/05/2014

@author: bliang03
'''

## Directories
## -------------------------------------------------------
# Training data folder
trainDataDir = 'C:\\ChaLAP_dataset\\Training data\\'

# out model folder
modelDir = './model_files/'

# Test data folder
testDataDir = 'C:\\ChaLAP_dataset\\Labeled validation data\\'

# Ground truth folder
outGT='./gt/'

# Predictions folder (output)
outPred='./pred/'

# Submision folder (output)
outSubmision='./submission/'

## -------------------------------------------------------


## Segmentation
## -------------------------------------------------------
SEGMENT_WINDOW_SIZE = 5
## -------------------------------------------------------


# Cross Validation
## -------------------------------------------------------
# True or false 
cv = False

# cross validation for mtm features
cvMtmOutputPath = './cv/mtm_svm/'
cvMtmFileName = 'mtm_data_file'

# cross validation for sta features
cvStaOutputPath = './cv/sta_svm/'
cvStaFileName = 'sta_data_file'

# cross validation for hmmoutput features
cvHmmSvmOutputPath = './cv/hmm_svm/'
cvHmmSvmFileName = 'hmm_data_file'
## -------------------------------------------------------


## Model parameters
## -------------------------------------------------------
# sta rbf-svm
svm_sta_c = 32
svm_sta_g = 0.0078125

# mtm linear-svm
svm_mtm_c = 0.5
svm_mtm_g = 2.0

# hmm
hmm_numStates = 15
hmm_numMix = 2

# hmm-svm
hmm_svm_c = 8192
hmm_svm_g = 8

## -------------------------------------------------------


## Model fusion parameters
## -------------------------------------------------------
# hmmModelFactor = 0.2
hmmSvmModelFactor = 0.25
staModelFactor = 0.5
mtmModelFactor = 0.25

accuThresh = 0.3
## -------------------------------------------------------