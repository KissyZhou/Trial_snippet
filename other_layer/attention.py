#!/usr/bin/python2.7.13
# -*- coding: utf-8 -*-
"""
Created on  2018/10/25 17:13

@author: KissyZhou
"""

from keras.layers import Lambda, Dense, multiply
import keras.backend as K

# attention layer used in our LSTM-based model
def attention_tartget_layer(inputs, target_vec=None, timesteps=8):
    #exp_target = Lambda(lambda x: K.expand_dims(x,axis=1), name='exp_target')(target_vec)
    #exp_target = target_vec
    target_vec = Lambda(lambda x: K.repeat_elements(x, axis=1, rep=timesteps), name='expand_tar')(target_vec)
    att_weight = Lambda(lambda x: K.concatenate([x[0], x[1], x[0]-x[1], x[0]*x[1]], axis=-1),
                        name='att_concat')([inputs, target_vec])
    att_weight = Dense(128, activation='sigmoid', name='att_fc1')(att_weight)
    att_weight = Dense(64, activation='sigmoid', name='att_fc2')(att_weight)
    att_weight = Dense(1, activation=None, name='att_fc3')(att_weight)
    # att_weight=Lambda(lambda x : K.batch_dot(x[0],x[1],  axes=2) /
    #                              K.maximum(K.expand_dims(
    #                                  K.sqrt(K.sum(K.square(x[0]),axis=2) *
    #                                         K.sum(K.square(x[1]),axis=2)),axis=-1), K.epsilon()) ,
    #                   name="calc_att_weight")([inputs, target_vec])
    att_weight = Lambda(lambda x: K.squeeze(x,axis=-1), name='dim_squeeze')(att_weight)
    att_weight = Lambda(lambda x: K.softmax(x), name='attention_norm')(att_weight)
    att_weight = Lambda(lambda x: K.expand_dims(x, axis=-1), name='dim_exp')(att_weight)
    weighted_vec = multiply(inputs=[att_weight, inputs])
    weighted_vec = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), name='dim_sum')(weighted_vec)
    return weighted_vec