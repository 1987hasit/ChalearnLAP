# -*- coding:utf-8 -*-  

'''
Created on 15/04/2014

@author:  Bin Liang

'''

# Gesture type
class GestureType():
    unknown = -1
    rest = 0
    left_hand = 1
    right_hand = 2
    both_hands = 3
    single_hand = 4
    
    
# Sliding window size for segmentation
SEGMENT_WINDOW_SIZE = 1