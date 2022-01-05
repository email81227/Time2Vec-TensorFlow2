from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class Time2Vec(Layer):
    def __init__(self, kernel_size, periodic_activation='sin', **kwargs):
        '''
        
        :param kernel_size:         The length of time vector representation.
        :param periodic_activation: The periodic activation, sine or cosine, or any future function.
        '''
        super(Time2Vec, self).__init__(
            trainable=True,
            name='Time2VecLayer_'+periodic_activation.upper(),
            **kwargs
        )
        
        self.k = kernel_size
        self.p_activation = periodic_activation
    
    def build(self, input_shape):
        # While i = 0
        self.wb = self.add_weight(
            shape=(1, 1),
            initializer='uniform',
            trainable=True,
            name='wb_weight'
        )
        
        self.bb = self.add_weight(
            shape=(1, 1),
            initializer='uniform',
            trainable=True,
            name='bb_weight'
        )
        
        # Else needs to pass the periodic activation
        self.wa = self.add_weight(
            shape=(1, self.k),
            initializer='uniform',
            trainable=True,
            name='wa_weight'
        )
        
        self.ba = self.add_weight(
            shape=(1, self.k),
            initializer='uniform',
            trainable=True,
            name='ba_weight'
        )
        
        super(Time2Vec, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        '''
        
        :param inputs: A Tensor with shape (batch_size, feature_size, 1)
        :param kwargs:
        :return: A Tensor with shape (batch_size, feature_size, length of time vector representation + 1)
        '''
        bias = self.wb * inputs + self.bb
        if self.p_activation.startswith('sin') :
            wgts = K.sin(K.dot(inputs, self.wa) + self.ba)
        elif self.p_activation.startswith('cos') :
            wgts = K.cos(K.dot(inputs, self.wa) + self.ba)
        else:
            raise NotImplementedError('Neither sine or cosine periodic activation be selected.')
        return K.concatenate([bias, wgts], -1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.k + 1)

    def get_config(self):
        config = super(Time2Vec, self).get_config()
        config.update({"kernel_size": self.k})
        config.update({"periodic_activation": self.p_activation})
        return config
