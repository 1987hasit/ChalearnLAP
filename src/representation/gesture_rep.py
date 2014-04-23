# -*- coding:utf-8 -*-  

'''
Created on 01/04/2014

@author: bliang03
'''


import numpy as np
import cv2
from scipy.misc import bytescale


def computeMHI(video, thresh):
    ''' Get the MHI image from a given video 
        frame shape is [h x w], frame number is n
        video shape is [h x w x n]
    '''
    [h, w, n] = video.shape
    
    firstFrame = video[:, :, 0]
    prev_frame = firstFrame.copy()
    motion_history = np.zeros((h, w), np.float32)
    
    MHI_DURATION = n
    
    for i in range(1, n):
        frame = video[:, :, i]
        frame_diff = cv2.absdiff(frame, prev_frame)
        #gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        gray_diff = bytescale(frame_diff)
        _, motion_mask = cv2.threshold(gray_diff, thresh, 1, cv2.THRESH_BINARY)
        timestamp = i
        # calculate motion history image
        cv2.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)        
            
    motionHistoryImage = bytescale(motion_history)  # convert motion history data to [0-255]
    return motionHistoryImage
        