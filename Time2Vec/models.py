# import os
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from tensorflow.keras import layers, Model, backend as K
from Time2Vec.layers import Time2Vec


def time2vec_lstm(dim, t2v_dim):
    '''
    
    :param dim:
    :param t2v_dim:
    :return:
    '''
    term = layers.Input(shape=(dim, 1))
    time = layers.Input(shape=(dim, 1))
    xti = Time2Vec(t2v_dim)(time)
    xte = layers.LSTM(32)(term)
    x = layers.Dense(1)(layers.concatenate([xte, layers.Flatten()(xti)]))
    m = Model([time, term], x)
    return m


def general_lstm(dim):
    inp = layers.Input(shape=(dim, 1))
    x = layers.LSTM(32)(inp)
    x = layers.Dense(1)(x)
    m = Model(inp, x)
    return m