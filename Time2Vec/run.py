import os
import numpy as np
import pandas as pd
import tensorflow as tf

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpu) == 1
tf.config.experimental.set_memory_growth(gpu[0], True)

from collections import deque
from tensorflow import keras

from Time2Vec.models import general_lstm, time2vec_lstm


SEQ_LEN = 32


dt = pd.read_csv(r'Time2Vec/data/788_2356_bundle_archive/venezia.csv')
dt['datetime']  = pd.to_datetime(dt['datetime'])
dt['timedelta'] = dt['datetime'] - dt['datetime'].shift()
dt['timedelta'] = dt.timedelta.dt.total_seconds()/100
dt['lvl_change']= dt.level.diff().fillna(0)


def raw2sample(seq_x, seq_y, seq_len=SEQ_LEN):
    X = []
    x, y = deque(maxlen=seq_len), []
    x.extend(seq_x[:SEQ_LEN].to_list())
    
    for val_x, val_y in zip(seq_x[seq_len:].to_list(), seq_y[seq_len:].to_list()):
        X.append(np.array(x.copy()))
        y.append(val_y)
        
        x.append(val_x)
    
    return X, y

X1, y  = raw2sample(dt.timedelta, dt.lvl_change, SEQ_LEN)
X2, _  = raw2sample(dt.lvl_change, dt.lvl_change, SEQ_LEN)
X1, X2, y  = np.array(X1), np.array(X2), np.array(y)

tr_X1, tr_X2,tr_y = X1[1:108000], X2[1:108000], y[1:108000]
# vl_X, vl_y = X[72000:108000],  y[72000:108000]
ts_X1, ts_X2, ts_y = X1[108000:], X2[108000:], y[108000:]

class ModelCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] <= .5:
            print('Retched loss <= .5, stop training...')
            self.model.stop_training=True

# ----------

model = time2vec_lstm(SEQ_LEN, 16)
model.summary()

model.compile(
    loss=keras.losses.mean_squared_error,
    optimizer=keras.optimizers.Adam(learning_rate=1e-3)
)
history = model.fit(
    x=[tr_X1.reshape((-1, SEQ_LEN, 1)), tr_X2.reshape((-1, SEQ_LEN, 1))],
    y=tr_y,
    batch_size=64, epochs=10, verbose=2,
    callbacks=[ModelCallback()]
)

model.evaluate([ts_X1, ts_X2], ts_y)

# ----------

model = general_lstm(SEQ_LEN)
model.summary()

model.compile(
    loss=keras.losses.mean_squared_error,
    optimizer=keras.optimizers.Adam(learning_rate=1e-3)
)
model.fit(
    x=tr_X1.reshape((-1, SEQ_LEN, 1)),
    y=tr_y,
    batch_size=64, epochs=50, verbose=2,
    callbacks=[ModelCallback()]
)

history = model.evaluate(ts_X1, ts_y)