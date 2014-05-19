# -*- coding:utf-8 -*-  

'''
Created on 01/04/2014

@author: bliang03
'''

import numpy as np
import cv2
from scipy.misc import bytescale


def depthFrameDiff(currImg, preImg, motion_thresh, static_thresh):
    ''' frame difference '''
    diff = cv2.absdiff(currImg, preImg).astype(np.int32)
    motionImg = diff.copy()
    motionImg[np.where(diff >= motion_thresh)] = 1
    motionImg[np.where(diff < motion_thresh)] = 0
    
    staticRegion = currImg - diff
    staticImg = staticRegion.copy()
    staticImg[np.where(staticRegion > static_thresh)] = 1
    staticImg[np.where(staticRegion <= static_thresh)] = 0
    
    return motionImg, staticImg


# Self added
def get2DMTM(gesture, gestureSample):
    """ Get 2DMTM representation from one gesture of a gesture sequence (gestureSample) """
    
    startFrame = gesture.getStartFrame()
    endFrame = gesture.getEndFrame()
    duration = endFrame - startFrame + 1
            
    xoyFirstImg, _ = gestureSample.getGestureRegion(startFrame)
    xoyHeight, xoyWidth = xoyFirstImg.shape
    
    # a set of DMHIs
    D_MHIs = []
    D_MHIs.append(np.zeros((xoyHeight, xoyWidth)).astype(np.int32))
    
    # a set of SHIs
    SHIs = []
    SHIs.append(np.zeros((xoyHeight, xoyWidth)).astype(np.int32))
    
    # AMI and ASI initialization
    AMI = np.zeros((xoyHeight, xoyWidth))
    ASI = np.zeros((xoyHeight, xoyWidth))
    
    # Motion threshold
    xoy_motion_thresh = 10
    xoy_static_thresh= 50
    
    for i in range(startFrame + 1, endFrame + 1):
        # current image
        currImg, _ = gestureSample.getGestureRegion(i)
        
        # previous image
        prevImg, _ = gestureSample.getGestureRegion(i - 1)
        
        # frame difference
        motionImg, staticImg = depthFrameDiff(currImg, prevImg, xoy_motion_thresh, xoy_static_thresh)
        
        # AMI and ASI summation
        AMI = AMI + motionImg
        ASI = ASI + staticImg
        
        ## XOY
        # DMHI
        # if D == 1
        DMHI = motionImg.copy()
        DMHI[np.where(motionImg == 1)] = duration
        
        # otherwise
        tmp = np.maximum(0, D_MHIs[-1] - 1)
        idx = np.where(motionImg != 1)
        DMHI[idx] = tmp[idx]
        
        D_MHIs.append(DMHI)
        
        # SHI
        # if D == 1
        SHI = staticImg.copy()
        SHI[np.where(staticImg)] = duration
        
        # otherwise
        tmp = np.maximum(0, SHIs[-1] - 1)
        idx = np.where(staticImg != 1)
        SHI[idx] = tmp[idx]
        
        SHIs.append(SHI)
        
    # get the result    
    finalDMHI = D_MHIs[-1]
    finalSHI = SHIs[-1]
    AMI = AMI / duration
    ASI = ASI / duration
    
    # convert to image
    dmhiImg = bytescale(finalDMHI)
    shiImg = bytescale(finalSHI)
    amiImg = bytescale(AMI)
    asiImg = bytescale(ASI)
    
    # return values
    two_d_mtm = []
    two_d_mtm.append(dmhiImg)
    two_d_mtm.append(shiImg)
    two_d_mtm.append(amiImg)
    two_d_mtm.append(asiImg)
        
    return two_d_mtm


# Self added
def get3DMTM(gesture, gestureSample):
    """ Extract 3DMTM features from one gesture of a gesture sequence (gestureSample) """
    
    startFrame = gesture.getStartFrame()
    endFrame = gesture.getEndFrame()
    duration = endFrame - startFrame + 1
            
    xoyFirstImg, _ = gestureSample.getGestureRegion(startFrame)
    xozFirstImg, yozFirstImg = gestureSample.getProjection(startFrame)
    xoyHeight, xoyWidth = xoyFirstImg.shape
    xozHeight, xozWidth = xozFirstImg.shape
    yozHeight, yozWidth = yozFirstImg.shape
    
    # a set of DMHIs
    D_MHIs = []
#     D_MHIs.append(xoyFirstImg.astype(np.int32)) 
    D_MHIs.append(np.zeros((xoyHeight, xoyWidth)).astype(np.int32))
    
    # a set of SHIs
    SHIs = []
#     SHIs.append(xoyFirstImg.astype(np.int32))
    SHIs.append(np.zeros((xoyHeight, xoyWidth)).astype(np.int32))
    
    # a set of XOZ_MHIs
    XOZ_MHIs = []
#     XOZ_MHIs.append(xozFirstImg.astype(np.int32))
    XOZ_MHIs.append(np.zeros((xozHeight, xozWidth)).astype(np.int32))
    
    # a set of YOZ_MHIs
    YOZ_MHIs = []
#     YOZ_MHIs.append(yozFirstImg.astype(np.int32))
    YOZ_MHIs.append(np.zeros((yozHeight, yozWidth)).astype(np.int32))
    
    # AMI and ASI initialization
    AMI = np.zeros((xoyHeight, xoyWidth))
    ASI = np.zeros((xoyHeight, xoyWidth))
    
    # Motion threshold
    xoy_motion_thresh = 10
    xoy_static_thresh= 50
    
    other_motion_thresh = 5
    other_static_thresh= 20
    
    for i in range(startFrame + 1, endFrame + 1):
        # current image
        currImg, _ = gestureSample.getGestureRegion(i)
        currXozImg, currYozImg = gestureSample.getProjection(i)
        
        # previous image
        prevImg, _ = gestureSample.getGestureRegion(i - 1)
        preXozImg, preYozImg = gestureSample.getProjection(i - 1)
        
        # frame difference
        motionImg, staticImg = depthFrameDiff(currImg, prevImg, xoy_motion_thresh, xoy_static_thresh)
        motionXozImg, staticXozImg = depthFrameDiff(currXozImg, preXozImg, other_motion_thresh, other_static_thresh)
        motionYozImg, staticYozImg = depthFrameDiff(currYozImg, preYozImg, other_motion_thresh, other_static_thresh)
        
        # AMI and ASI summation
        AMI = AMI + motionImg
        ASI = ASI + staticImg
        
        ## XOY
        # DMHI
        # if D == 1
        DMHI = motionImg.copy()
        DMHI[np.where(motionImg == 1)] = duration
        
        # otherwise
        tmp = np.maximum(0, D_MHIs[-1] - 1)
        idx = np.where(motionImg != 1)
        DMHI[idx] = tmp[idx]
        
        D_MHIs.append(DMHI)
        
        # SHI
        # if D == 1
        SHI = staticImg.copy()
        SHI[np.where(staticImg)] = duration
        
        # otherwise
        tmp = np.maximum(0, SHIs[-1] - 1)
        idx = np.where(staticImg != 1)
        SHI[idx] = tmp[idx]
        
        SHIs.append(SHI)
        
        ## XOZ
        # MHI
        # if D == 1
        XOZ_MHI = motionXozImg.copy()
        XOZ_MHI[np.where(motionXozImg == 1)] = duration
        
        # otherwise
        tmp = np.maximum(0, XOZ_MHIs[-1] - 1)
        idx = np.where(motionXozImg != 1)
        XOZ_MHI[idx] = tmp[idx]
        
        XOZ_MHIs.append(XOZ_MHI)
        
        ## YOZ
        # MHI
        # if D == 1
        YOZ_MHI = motionYozImg.copy()
        YOZ_MHI[np.where(motionYozImg == 1)] = duration
        
        # otherwise
        tmp = np.maximum(0, YOZ_MHIs[-1] - 1)
        idx = np.where(motionYozImg != 1)
        YOZ_MHI[idx] = tmp[idx]
        
        YOZ_MHIs.append(YOZ_MHI)
        
    # get the result    
    finalDMHI = D_MHIs[-1]
    finalSHI = SHIs[-1]
    finalXozMHI = XOZ_MHIs[-1]
    finalYozMHI = YOZ_MHIs[-1]
    AMI = AMI / duration
    ASI = ASI / duration
    
    # convert to image
    dmhiImg = bytescale(finalDMHI)
    shiImg = bytescale(finalSHI)
    xozMhiImg = bytescale(finalXozMHI)
    yozMhiImg = bytescale(finalYozMHI)
    amiImg = bytescale(AMI)
    asiImg = bytescale(ASI)
    
    # return values
    three_d_mtm = []
    three_d_mtm.append(dmhiImg)
    three_d_mtm.append(shiImg)
    three_d_mtm.append(xozMhiImg)
    three_d_mtm.append(yozMhiImg)
    three_d_mtm.append(amiImg)
    three_d_mtm.append(asiImg)
        
    return three_d_mtm
        