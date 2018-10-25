#!/usr/bin/python2.7.13
# -*- coding: utf-8 -*-
"""
Created on  2018/10/25 16:56

@author: KissyZhou
"""

from keras.layers import BatchNormalization, Activation, Lambda
from keras.layers import Conv1D
import keras.backend as K

def dice_keras(input, name='dice'):
    '''
    :param input: input features
    :param name: activation layer name
    :return: keras layer
    '''
    x_normed = BatchNormalization(center=False, scale=False, name='{}/bn'.format(name))(input)
    x_p = Activation('sigmoid', name='{}/activation'.format(name))(x_normed)
    x_p1 = Lambda(lambda x: K.expand_dims(1.0 - x, 2), name='{}/1_minus'.format(name))(x_p)
    x_p1 = Conv1D(1, int(input.get_shape()[-1]), activation=None, name='{}/conv_alpha'.format(name))(x_p1)
    x_p1 = Lambda(lambda x: K.squeeze(x, -1), name='{}/squeeze'.format(name))(x_p1)
    return Lambda(lambda x: x[0] * input + x[1] * input, name='{}/weighted_x'.format(name))([x_p1, x_p])

def swish(x):
    '''
    work better with 'he_uniform' initialization
    :param x:
    :return:
    usage:
    add custom activation function
    get_custom_objects().update({'swish': Activation(swish)})
    then use swish same as 'sigmoid'
    '''
    return K.sigmoid(x) * x