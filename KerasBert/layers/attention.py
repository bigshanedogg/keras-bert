# -*- coding: utf-8 -*-
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as K
from .utils import Masking, LayerNormalization, gelu

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, timesteps, num_heads, d_model, dropout, name=None, **kwargs):
        self.timesteps = timesteps
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_v = int(d_model / num_heads)
        self.dropout = dropout
        if name is not None: super(MultiHeadAttention, self).__init__(name=name, **kwargs)
        else: super(MultiHeadAttention, self).__init__(**kwargs)
            
    def build(self, input_shape):
        assert_statement = "Inputs should be 4 for Q,K,V and MASK_TENSOR; got {input_len} Inputs only".format(input_len=len(inputs))
        assert len(input_shape)==4, assert_statement
        
        kernel_initializer = tf.initializers.truncated_normal(stddev=0.02)
        bias_initializer = tf.initializers.zeros()
        self.W_query = self.add_weight(shape=(self.d_model, self.num_heads*self.d_k), dtype=tf.float32, initializers=kernel_initializer, name="W_query")
        self.W_key = self.add_weight(shape=(self.d_model, self.num_heads*self.d_k), dtype=tf.float32, initializers=kernel_initializer, name="W_key")
        self.W_value = self.add_weight(shape=(self.d_model, self.num_heads*self.d_k), dtype=tf.float32, initializers=kernel_initializer, name="W_value")
        self.b_query = self.add_weight(shape=(self.num_heads*self.d_k, ), dtype=tf.float32, initializers=bias_initializer, name="b_query")
        self.b_key = self.add_weight(shape=(self.num_heads*self.d_k, ), dtype=tf.float32, initializers=bias_initializer, name="b_key")
        self.b_value = self.add_weight(shape=(self.num_heads*self.d_k, ), dtype=tf.float32, initializers=bias_initializer, name="b_value")
        
        self.W_O = self.add_weight(shape=(self.num_heads*self.d_k, self.d_model), dtype=tf.float32, initializers=kernel_initializer, name="W_O")
        self.b_O = self.add_weight(shape=(self.d_model, ), dtype=tf.float32, initializers=bias_initializer, name="b_O")
        super(MultiHeadAttention, self).build(input_shape)
        
    def call(self, inputs):
        # input_q, input_k, input_v: (batch_size, timesteps, d_model)
        input_q, input_k, input_v, mask_vec = inputs
        # input_q, input_k, input_v: (batch_size*timesteps, num_heads*d_k)
        # d_model = num_heads*d_k
        input_q = K.reshape(input_q, (-1, self.d_model))
        input_k = K.reshape(input_q, (-1, self.d_model))
        input_v = K.reshape(input_q, (-1, self.d_model))
        
        # input_q, input_k, input_v: (batch_size*timesteps, num_heads*d_k)
        # d_model = num_heads*d_k
        query = tf.nn.bias_add(K.dot(input_q, self.W_query), self.b_query)
        key = tf.nn.bias_add(K.dot(input_q, self.W_key), self.b_key)
        value = tf.nn.bias_add(K.dot(input_q, self.W_value), self.b_value)
        
        # attention attention_weights
        # logits, attention_weights: (bathc_size, num_heads, timesteps, timesteps)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        logits = matmul_qk / tf.sqrt(float(self.d_k))
        if mask_vec is not None: logtis = Masking()([logits, mask_vec])
        attention_weights = keras.layers.Activation("softmax", name="attention_weights")(logtis)
        attention_weights = keras.layers.Dropout(self.dropout)(attention_weights)
        
        # attention output
        # output: (batch_size, timesteps, num_heads, d_k)
        output = tf.matmul(attention_weights, value, name="output")
        # output: (batch_size, num_heads, timesteps, d_k)
        output = tf.transpose(output, [0,2,1,3])
        # output: (batch_size*timesteps, num_heads*d_k)
        # d_model = num_heads*d_k
        output = tf.reshape(output, [-1, self.d_k*self.num_heads])
        # output linear
        output = tf.nn.bias_add(K.dot(output, self.W_O), self.b_O)
        
        # sub-layer connection
        output = keras.layers.Dropout(self.dropout)(output)
        output = keras.layers.Add()([input_q, output])
        output = LayerNormalization()(output)
        
        # output: (batch_size, timesteps, num_heads*d_k)
        output = tf.reshape(output, [-1, self.timesteps, self.d_k, self.num_heads])
        return [output, attention_weights]
        
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        output_shape = (batch_size, self.timesteps, self.num_heads*self.d_k)
        attention_weights_shape = (batch_size, self.num_heads, sef.timesteps, self.timesteps)
        return [outptu_shape, attention_weights_shape]
        
    def get_config(self):
        config = {
            "timesteps": self.timesteps,
            "num_heads": self.num_heads,
            "d_model": self.d_model,
            "dropout": self.dropout
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
        
class PositionwiseFeedForward(keras.layers.Layer):
    def __init__(self, d_model, d_ff, dropout, activation, activation_func, name=None, **kwargs):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.activation_func = activation_func
        try:
            self.activation_func = keras.activations.get(self.activation)
        except:
            if self.activation == "gelu": self.activation_func = gelu
            else: raise AssertionError("No such activation function named '{given}'".format(given=self.activation))
        if name is not None: super(PositionwiseFeedForward, self).__init__(name=name, **kwargs)
        else: super(PositionwiseFeedForward, self).__init__(**kwargs)
            
    def build(self, input_shape):
        kernel_initializer = tf.initializers.truncated_normal(stddev=0.02)
        bias_initializer = tf.initializers.zeros()
        self.W_1 = self.add_weight(shape=(self.d_model, self.d_ff), dtype=tf.float32, initializers=kernel_initializer, name="W_1")
        self.b_1 = self.add_weight(shape=(self.d_ff, ), dtype=tf.float32, initializers=bias_initializer, name="b_1")
        self.W_2 = self.add_weight(shape=(self.d_ff, self.d_model), dtype=tf.float32, initializers=kernel_initializer, name="W_2")
        self.b_2 = self.add_weight(shape=(self.d_model, ), dtype=tf.float32, initializers=bias_initializer, name="b_2")
        
        self.W_O = self.add_weight(shape=(self.num_heads*self.d_k, self.d_model), dtype=tf.float32, initializers=kernel_initializer, name="W_O")
        self.b_O = self.add_weight(shape=(self.d_model, ), dtype=tf.float32, initializers=bias_initializer, name="b_O")
        super(PositionwiseFeedForward, self).build(input_shape)
        
    def call(self, inputs):
        F_1 = K.dot(inputs, self.W_1) + self.b_1
        F_1 = keras.layers.Lambda(lambda x: self.activation_func(x), name=self.activation)(F_1)
        F_2 = K.dot(F_1, self.W_2) + self.b_2
        output = keras.layers.Dropout(self.dropout)(F_2)
        output = keras.layers.Add()([inputs, output])
        output = LayerNormalization()(output)
        return output
        
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        timesteps = input_shape[1]
        output_shape = (batch_size, self.timesteps, self.d_model)
        return outptu_shape
        
    def get_config(self):
        config = {
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "activation": self.activation
        }
        base_config = super(PositionwiseFeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))