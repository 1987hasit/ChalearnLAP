# -*- coding:utf-8 -*-  

'''
Created on 24/03/2014

@author: Bin Liang
'''


from common import GestureType
import numpy as np
from sklearn import hmm
# from hmmlearn import hmm

class Gesture:
    """ Class for gesture features """
    
    def __init__(self, gestureID, startFrame, endFrame, fileName):
        """ Initialize Gesture using basic information """
        # label format (gestureID,startFrame,endFrame)
        self.gestureID = gestureID
        self.startFrame = startFrame
        self.endFrame = endFrame
        
        self.fileName = fileName
        self.gestureType = GestureType.unknown
        
        ## Features
        # statistic skeleton features
        self.staFeatures = None
        
        # MTM features
        self.mtmFeatures = None
        
        # time domain features
        self.timeDomainFeatures = None
        
        # hmm output features
        self.hmmOutputFeatures = None
        
        
    def getGestureID(self):
        """ Get start frame of the gesture """
        return self.gestureID    
    
    def getStartFrame(self):
        """ Get start frame of the gesture """
        return self.startFrame
        
    def getEndFrame(self):
        """ Get end frame of the gesture """
        return self.endFrame
    
    def setStaFeatures(self, staFeatures):
        self.staFeatures = staFeatures
        
    def setTimeDomainFeatures(self, timeDomainFeatures):
        self.timeDomainFeatures = timeDomainFeatures
    
    def setMtmFeatures(self, mtmFeatures):
        self.mtmFeatures = mtmFeatures
        
    def setHmmOutputFeatures(self, hmmOutputFeatures):
        self.hmmOutputFeatures = hmmOutputFeatures    


class GestureModel:
    ''' gesture model class '''
    
    def __init__(self, gestureID):
        self.gestureID = gestureID
        self.trainData = []
        self.hmmModel = None
        self.gestureType = GestureType.unknown
        
        self.nStates = 5  # number of states
        self.nMix = 2   # number of mixtures
        self.covarianceType = 'diag'    # covariance type
        self.n_iter = 10    # number of iterations
        self.startprobPrior = None
        self.transmatPrior = None
        self.bakisLevel = 2
       
    def initModelParam(self, nStates, nMix, covarianceType, n_iter):
        ''' init params for hmm model '''
        
        self.nStates = nStates  # number of states
        self.nMix = nMix   # number of mixtures
        self.covarianceType = covarianceType    # covariance type
        self.n_iter = n_iter    # number of iterations
        
        startprobPrior, transmatPrior = self.initByBakis(nStates, self.bakisLevel)
        self.startprobPrior = startprobPrior
        self.transmatPrior = transmatPrior
        
    def initByBakis(self, nStates, bakisLevel):
        ''' init start_prob and transmat_prob by Bakis model ''' 
        startprobPrior = np.zeros(nStates)
        startprobPrior[0 : bakisLevel - 1] = 1./ (bakisLevel - 1)
         
        transmatPrior = self.getTransmatPrior(nStates, bakisLevel)
         
        return startprobPrior, transmatPrior
    
    def getTransmatPrior(self, nStates, bakisLevel):
        ''' get transmat prior '''
        transmatPrior = (1. / bakisLevel) * np.eye(nStates)
         
        for i in range(nStates - (bakisLevel - 1)):
            for j in range(bakisLevel - 1):
                transmatPrior[i, i + j + 1] = 1. /  bakisLevel
                 
        for i in range(nStates - bakisLevel + 1, nStates):
            for j in range(nStates - i -j):
                transmatPrior[i, i + j] = 1. / (nStates - i)
         
        return transmatPrior
    
    def getTrainData(self, sameIDGestureList):
        ''' get training data from gesture list with the same id '''
#         gestureTypeList = []
        
        for gesture in sameIDGestureList:
            data = gesture.timeDomainFeatures
            if data.size == 0:
                print gesture.gestureID, gesture.fileName
            self.trainData.append(data)
#             gestureTypeList.append(gesture.gestureType)
            
#         gestureTypeArr = np.array(gestureTypeList)
#         numLeftHand = gestureTypeArr[np.where(gestureTypeArr == GestureType.left_hand)].size
#         numRightHand = gestureTypeArr[np.where(gestureTypeArr == GestureType.right_hand)].size
#         numBothHands = gestureTypeArr[np.where(gestureTypeArr == GestureType.both_hands)].size
#         
#         predTypeDict = {numLeftHand : GestureType.left_hand, numRightHand : GestureType.right_hand, numBothHands : GestureType.both_hands}
#         maxNum = max(numLeftHand, numRightHand, numBothHands)
#         gestureType = predTypeDict.get(maxNum)
#         
#         if gestureType == GestureType.both_hands:
#             # both hands gesture type
#             self.gestureType = gestureType
#         else:
#             # single hand gesture type
#             self.gestureType = GestureType.single_hand
#         
#         if self.gestureType == GestureType.single_hand:
#             
#             for geture in sameIDGestureList:
#                 if gesture.gestureType == GestureType.left_hand:
#                     data = geture.leftGestureFeatures
#                 elif gesture.gestureType == GestureType.right_hand:
#                     data = geture.rightGestureFeatures
#                     
#                 self.trainData.append(data)
#                 
#         elif self.gestureType == GestureType.both_hands:
#             # both hands 
#             for geture in sameIDGestureList:
#                 data = np.hstack((geture.leftGestureFeatures, geture.rightGestureFeatures))
#                 self.trainData.append(data)
        
        
    def trainHMMModel(self):
        ''' train hmm model from training data '''
        
        # GaussianHMM
#         model = hmm.GaussianHMM(self.nStates, "full") # initialize hmm model
#         model = hmm.GaussianHMM(n_components = self.nStates, transmat_prior = self.transmatPrior, \
#                                 startprob_prior = self.startprobPrior, covariance_type = self.covarianceType, \
#                                 n_iter = self.n_iter) # initialize hmm model
        
        # Gaussian Mixture HMM
#         model = hmm.GMMHMM(n_components = self.nStates, n_mix = self.nMix, \
#                            transmat_prior = self.transmatPrior, startprob_prior = self.startprobPrior, \
#                            covariance_type = self.covarianceType, n_iter = self.n_iter)
        
        model = hmm.GMMHMM(n_components = self.nStates, n_mix = self.nMix, \
                   covariance_type = self.covarianceType, n_iter = self.n_iter)
        
        
        obs = self.trainData
        model.fit(obs)   # get optimal parameters
        self.hmmModel = model
