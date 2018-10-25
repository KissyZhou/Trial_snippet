#!/usr/bin/python2.7.13
# -*- coding: utf-8 -*-
"""
Created on  2018/10/25 17:03

@author: KissyZhou
"""

import keras.backend as K

def ce_l2(y_true, y_pred):
    '''
    :param y_true:
    :param y_pred:
    :return:
     usage: same as 'binary_crossentropy'
    '''
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) + K.mean(K.square((1.0 - y_true) * y_pred))