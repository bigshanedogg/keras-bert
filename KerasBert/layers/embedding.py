import numpy as np
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as K
from .utils import LayerNormalization

class TokenEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, name="token_embedding", **kwargs):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        super(TokenEmbedding, self).__init__(name=name, **kwargs)
            
    def build(self, input_shape):
        kernel_initializer = tf.initializers.truncated_normal(stddev=0.02)
        self.token_embed_weights = self.add_weight(shape=(self.vocab_size, self.embedding_dim), name="token_embed_weights", initializers=kernel_initializer, dtype=tf.float32)
        super(TokenEmbedding, self).build(input_shape)
        
    def call(self, inputs):
        if K.dtype(inputs) != "int32": inputs = K.cast(inputs, "int32")
        token_embed = K.gather(self.token_embed_weigths, inputs)
        return [token_embed, self.token_embed_weigths]
        
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        timesteps = input_shape[1]
        output_shape = (batch_size, timesteps, self.embedding_dim)
        token_embedding_table_shape = (self.vocab_size, self.embedding_dim)
        return [output_shape, token_embedding_table_shape]
        
    def get_config(self):
        config = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim
        }
        base_config = super(TokenEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
        
class SegmentEmbedding(keras.layers.Layer):
    def __init__(self, num_segments, embedding_dim, name="segment_embedding", **kwargs):
        self.num_segments = num_segments
        self.embedding_dim = embedding_dim
        super(SegmentEmbedding, self).__init__(name=name, **kwargs)
            
    def build(self, input_shape):
        kernel_initializer = tf.initializers.truncated_normal(stddev=0.02)
        self.segment_embed_weights = self.add_weight(shape=(self.num_segments, self.embedding_dim), name="segment_embed_weights", initializers=kernel_initializer, dtype=tf.float32)
        super(SegmentEmbedding, self).build(input_shape)
        
    def call(self, inputs):
        if K.dtype(inputs) != "int32": inputs = K.cast(inputs, "int32")
        segment_embed = K.gather(self.segment_embed_weigths, inputs)
        return [segment_embed, self.segment_embed_weigths]
        
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        timesteps = input_shape[1]
        output_shape = (batch_size, timesteps, self.embedding_dim)
        segment_embedding_table_shape = (self.num_segments, self.embedding_dim)
        return [outptu_shape, segment_embedding_table_shape]
        
    def get_config(self):
        config = {
            "num_segments": self.num_segments,
            "embedding_dim": self.embedding_dim
        }
        base_config = super(SegmentEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
        
class PosEmbedding(keras.layers.Layer):
    def __init__(self, num_pos, embedding_dim, name="pos_embedding", **kwargs):
        self.num_pos = num_pos
        self.embedding_dim = embedding_dim
        super(PosEmbedding, self).__init__(name=name, **kwargs)
            
    def build(self, input_shape):
        kernel_initializer = tf.initializers.truncated_normal(stddev=0.02)
        self.pos_embed_weights = self.add_weight(shape=(self.num_pos, self.embedding_dim), name="pos_embed_weights", initializers=kernel_initializer, dtype=tf.float32)
        super(PosEmbedding, self).build(input_shape)
        
    def call(self, inputs):
        if K.dtype(inputs) != "int32": inputs = K.cast(inputs, "int32")
        pos_embed = K.gather(self.pos_embed_weights, inputs)
        return [pos_embed, self.pos_embed_weights]
        
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        timesteps = input_shape[1]
        output_shape = (batch_size, timesteps, self.embedding_dim)
        pos_embedding_table_shape = (self.num_pos, self.embedding_dim)
        return [outptu_shape, pos_embedding_table_shape]
        
    def get_config(self):
        config = {
            "num_pos": self.num_pos,
            "embedding_dim": self.embedding_dim
        }
        base_config = super(PosEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
        
class EmbeddingPostprocess(keras.layers.Layer):
    def __init__(self, timesteps, d_model, dropout, position_embedding=True, name="embedding_postprocess", **kwargs):
        self.timesteps = timesteps
        self.d_model = d_model
        self.dropout = dropout
        self.position_embedding = position_embedding
        super(EmbeddingPostprocess, self).__init__(name=name, **kwargs)
            
    def build(self, input_shape):
        constant_initializer = tf.initializers.constant(self._position_embed(self.timesteps, self.d_model))
        self.position_embed_weights = self.add_weight(shape=(self.timesteps, self.d_model), name="position_embed_weights", initializers=constant_initializer, dtype=tf.float32, trainable=False)
        super(EmbeddingPostprocess, self).build(input_shape)
        
    def call(self, inputs):
        output = None
        if isinstance(inputs, list):
            output = inputs[0]
            for i in range(1, len(inputs)):
                output = tf.add(output, inputs[i])
        else:
            output = inputs
            
        # position embedding
        if self.position_embedding:
            output = tf.map_fn(lambda sequence: tf.add(sequence, self.position_embed_weights), output)
        # layer norm & dropout
        output = LayerNormalization()(output)
        output = keras.layers.Dropout(self.dropout)(output)
        return output
        
    def compute_output_shape(self, input_shape):
        batch_size = None
        if input_shape[0] is None:
            batch_size = input_shape[0]
        else:
            batch_size = input_shape[0][0]
        output_shape = (batch_size, self.timesteps, self.d_model)
        return output_shape
        
    def get_config(self):
        config = {
            "timesteps": self.timesteps,
            "d_model": self.d_model,
            "dropout": self.dropout
        }
        base_config = super(EmbeddingPostprocess, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
    def _position_embed(self, timesteps, d_model):
        def _get_angles(timesteps, d_model, i):
            default_value = 1e+4
            rates = 1 / np.power(default_value, (2 * 1) / np.float32(d_model))
            return rates * timesteps
        
        pos_array = np.expand_dims(np.arange(timesteps), axis=1)
        i_array = np.expand_dims(np.arange(d_model), axis=0) / 2
        pos_embed_matrix = _get_angles(pos_array, d_model, i_array)
        pos_embed_matrix[:, 0::2] = np.sin(pos_embed_matrix[:, 0::2])
        pos_embed_matrix[:, 1::2] = np.sin(pos_embed_matrix[:, 1::2])
        return pos_embed_matrix
            