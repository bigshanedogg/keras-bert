import numpy as np
import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as K

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * tf.pow(x, 3))))
    return x * cdf


class GetMaskTensor(keras.layers.Layer):
    def __init__(self, mask_type, pad_token_id=0, name="get_mask_tensor", **kwargs):
        self.mask_type_list = ["pad", "sub"]
        self.mask_type = mask_type
        self.pad_token_id = pad_token_id
        
        assert_statement_template = "No such mask type named '{given}'; should be one of {func_list}"
        mask_type_assert = assert_statement_template.format(given=self.mask_type, func_list=str(self.mask_type_list))
        assert self.mask_type in self.mask_type_list, mask_type_assert
        super(GetMaskTensor, self).__init__(name=name, **kwargs)
            
    def build(self, input_shape):
        super(GetMaskTensor, self).build(input_shape)
        
    def call(self, inputs):
        # inputs: (batch_size, timesteps)
        assert self.mask_type in self.mask_type_list, AssertionError("Invalid option, available mask types:", self.mask_type_list)
        # mask_tensor (batch_size, timesteps)
        mask_tensor = tf.equal(inputs, self.pad_token_id)
        
        if self.mask_type == "sub":
            timesteps = inputs.shape[-1]
            mask_tensor = K.repeat_elements(tf.expand_dims(mask_tensor, axis=-1), rep=timesteps, axis=-1)
            sub_mask_tensor = tf.linalg.LinearOperatorLowerTriangular(tf.ones_like(mask_tensor, dtype=tf.float32)).to_dense()
            sub_mask_tensor = tf.equal(sub_mask_tensor, 0)
            mask_tensor = tf.logical_or(mask_tensor, sub_mask_tensor)
            
        mask_tensor.trainable = False
        return mask_tensor
        
    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape
        
    def get_config(self):
        config = {
            "mask_type": self.mask_type,
            "pad_token_id": self.pad_token_id
        }
        base_config = super(GetMaskTensor, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

class Masking(keras.layers.Layer):
    def __init__(self, padding_value=-2 ** 32 + 1, **kwargs):
        self.padding_value = padding_value
        super(Masking, self).__init__(**kwargs)
            
    def build(self, input_shape):
        super(Masking, self).build(input_shape)
        
    def call(self, inputs):
        '''
        input_tensor: 
            BERT pad mask: (batch_size, num_heads, timesteps, timesteps)
            BERT sub mask: (batch_size, num_heads, timesteps, timesteps)
            Sentence BERT embedding mask: (batch_size, timesteps, d_model)
        mask_tensor: 
            BERT pad mask: (batch_size, timesteps)
            BERT sub mask: (batch_size, timesteps, timesteps)
            Sentence BERT embedding mask: (batch_size, timesteps, d_model)
        '''
        input_tensor, mask_tensor = inputs
        
        output = None
        if len(input_tensor.shape) == 4:
            # BERT pad mask, BERT sub mask_tensor
            num_heads = input_tensor.shape[1]
            timesteps = input_tensor.shape[-1]
            if len(mask_tensor.shape) == 2:
                # mask_tensor: (batch_size, timesteps, timesteps)
                mask_tensor = K.repeat_elements(tf.expand_dims(mask_tensor, axis=-1), rep=timesteps, axis=-1)
            if len(mask_tensor.shape) == 3:
                # mask_tensor: (batch_size, num_heads, timesteps, timesteps)
                mask_tensor = K.repeat_elements(tf.expand_dims(mask_tensor, axis=[1]), rep=num_heads, axis=-1)
            # input_tensor, mask_tensor, padding_tensor, output: (batch_size, num_heads, timesteps, timesteps)
            padding_tensor = tf.multiply(tf.ones_like(input_tensor), self.padding_value)
            output = tf.where(mask_tensor, padding_tensor, input_tensor)
            
        elif len(input_tensor.shape) == 3:
            # S-BERT embedding mask_tensor
            timesteps = input_tensor.shape[1]
            d_model = input_tensor.shape[-1]
            if len(mask_tensor.shape) == 2:
                # mask_tensor: (batch_size, timesteps, d_model)
                mask_tensor = K.repeat_elements(tf.expand_dims(mask_tensor, axis=-1), rep=d_model, axis=-1)
            # input_tensor, mask_tensor, padding_tensor, output: (batch_size, timesteps, d_model)
            padding_tensor = tf.multiply(tf.ones_like(input_tensor), self.padding_value)
            output = tf.where(mask_tensor, padding_tensor, input_tensor)
        return output 
        
    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape
        
    def get_config(self):
        config = {
            "padding_value": self.padding_value
        }
        base_config = super(Masking, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

class LayerNormalization(keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        self.epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)
            
    def build(self, input_shape):
        layer_dim = input_shape[-1]
        gamma_initializer =tf.initializers.ones()
        beta_initializer =tf.initializers.zeros()
        self.gamma = self.add_weight(shape=(layer_dim,), dtype=tf.float32, initializer=gamma_initializer, name="gamma")
        self.beta = self.add_weight(shape=(layer_dim,), dtype=tf.float32, initializer=beta_initializer, name="beta")
        super(LayerNormalization, self).build(input_shape)
        
    def call(self, inputs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        std = K.std(inputs, axis=-1, keepdims=True)
        output = (self.gamma * ((inputs - mean) / (std + self.epsilon))) + self.beta
        return output
        
    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        return output_shape
        
    def get_config(self):
        config = {
            "epsilon": self.epsilon
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BertPooling(keras.layers.Layer):
    def __init__(self, pooling, underflow=1e-7, dropout=0.1, name="bert_pooling", **kwargs):
        self.pooling_type_list = ["cls", "token_mean", "token_max", "tokens", "lstm", "bi_lstm"]
        self.pooling = pooling
        self.underflow = underflow
        self.dropout = dropout
        self.assert_statement = ""

        assert_statement_template = "No such pooling type named '{given}'; should be one of {func_list}"
        pooling_type_assert = assert_statement_template.format(given=self.pooling, func_list=str(self.pooling_type_list))
        assert self.pooling in self.pooling_type_list, pooling_type_assert
        super(BertPooling, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        super(BertPooling, self).build(input_shape)

    def call(self, inputs):
        # embed_output: (batch_size, timesteps, d_model)
        # mask_tensor: (batch_size, timesteps)
        embed_output, mask_vec = inputs
        d_model = int(embed_output.shapde[-1])

        output = None
        if self.pooling == "cls":
            output = embed_output[:, 0, :]
        elif self.pooling == "token_mean":
            # masking
            embed_output = Masking(padding_value=0)([embed_output, mask_vec])
            nom = tf.reduce_sum(embed_output, axis=1)
            denom = tf.reduce_sum(tf.cast(~mask_vec, dtype=tf.float32), axis=-1)
            denom = K.repeat_elements(tf.expand_dims(denom, axis=-1), rep=d_model, axis=-1)
            output = nom / (denom + self.underflow)
        elif self.pooling == "token_max":
            # masking
            embed_output = Masking(padding_value=0)([embed_output, mask_vec])
            output = tf.reduce_max(embed_output, axis=1)
        elif self.pooling == "tokens":
            # masking
            output = Masking(padding_value=0)([embed_output, mask_vec])
        elif self.pooling == "lstm":
            # masking
            embed_output = Masking(padding_value=0)([embed_output, mask_vec])
            output = keras.layers.LSTM(d_model, return_sequences=False, dropout=self.dropout, recurrent_dropout=self.dropout)(embed_output)
        elif self.pooling =="bi_lstm":
            # masking
            embed_output = Masking(padding_value=0)([embed_output, mask_vec])
            output = keras.layers.Bidirectional(keras.layers.LSTM(d_model, return_sequences=False, dropout=self.dropout, recurrent_dropout=self.dropout))(embed_output)

        return output

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        d_model = input_shape[0][1]
        output_shape = (batch_size, d_model)
        return output_shape

    def get_config(self):
        config = {
            "pooling": self.pooling,
            "underflow": self.underflow,
            "dropout": self.dropout
        }
        base_config = super(BertPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CosineSimilarity(keras.layers.Layer):
    def __init__(self, name="cosine_similarity", **kwargs):
        super(CosineSimilarity, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        super(CosineSimilarity, self).build(input_shape)

    def call(self, inputs):
        # input_tensor: (batch_size, d_model), (batch_size, d_model)
        left, right = inputs
        normalized_left = tf.nn.l2_normalize(left, axis=-1)
        normalized_right = tf.nn.l2_normalize(right, axis=-1)
        output = tf.reduce_sum(tf.multiply(normalized_left, normalized_right), axis=-1, keepdims=True)
        return output

    def compute_output_shape(self, input_shape):
        batch_size, d_model = input_shape[0]
        output_shape = (batch_size, 1)
        return output_shape

    def get_config(self):
        config = {}
        base_config = super(CosineSimilarity, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CosineDistance(keras.layers.Layer):
    def __init__(self, name="cosine_similarity", **kwargs):
        super(CosineDistance, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        super(CosineDistance, self).build(input_shape)

    def call(self, inputs):
        # input_tensor: (batch_size, d_model), (batch_size, d_model)
        left, right = inputs
        normalized_left = tf.nn.l2_normalize(left, axis=-1)
        normalized_right = tf.nn.l2_normalize(right, axis=-1)
        cosine_similarity = tf.reduce_sum(tf.multiply(normalized_left, normalized_right), axis=-1, keepdims=True)
        output = tf.subtract(tf.ones_like(cosine_similarity), cosine_similarity)
        return output

    def compute_output_shape(self, input_shape):
        batch_size, d_model = input_shape[0]
        output_shape = (batch_size, 1)
        return output_shape

    def get_config(self):
        config = {}
        base_config = super(CosineDistance, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class EuclideanDistance(keras.layers.Layer):
    def __init__(self, name="euclidean_distance", **kwargs):
        super(EuclideanDistance, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        super(EuclideanDistance, self).build(input_shape)

    def call(self, inputs):
        # input_tensor: (batch_size, d_model), (batch_size, d_model)
        left, right = inputs
        output = tf.sqrt(tf.reduce_sum(tf.square(left - right), axis=-1))
        return output

    def compute_output_shape(self, input_shape):
        batch_size, d_model = input_shape[0]
        output_shape = (batch_size, )
        return output_shape

    def get_config(self):
        config = {}
        base_config = super(EuclideanDistance, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TripletPostprocess(keras.layers.Layer):
    def __init__(self, name="triplet_postprocess", **kwargs):
        super(TripletPostprocess, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        super(TripletPostprocess, self).build(input_shape)

    def call(self, inputs):
        # input_tensor: (batch_size, ), (batch_size, )
        anchor_pooled, positive_pooled, negative_pooled = inputs
        anchor_pooled = tf.expand_dims(anchor_pooled, axis=1)
        positive_pooled = tf.expand_dims(positive_pooled, axis=1)
        negative_pooled = tf.expand_dims(negative_pooled, axis=1)
        output = tf.concat([anchor_pooled, positive_pooled, negative_pooled], axis=1)
        return output

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        d_model = input_shape[0][-1]
        output_shape = (batch_size, 3, d_model)
        return output_shape

    def get_config(self):
        config = {}
        base_config = super(TripletPostprocess, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SiameseConcat(keras.layers.Layer):
    def __init__(self, name="siamese_concat", **kwargs):
        super(SiameseConcat, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        super(SiameseConcat, self).build(input_shape)

    def call(self, inputs):
        # input_tensor: (batch_size, d_model), (batch_size, d_model)
        left, right = inputs
        l1_dist = keras.layers.Lambda(lambda x : tf.abs(x[0] - x[1]), name="l1_dist")([left, right])
        output = keras.layers.Concatenate(axis=1)([left, right, l1_dist])
        return output

    def compute_output_shape(self, input_shape):
        batch_size, d_model = input_shape[0]
        output_shape = (batch_size, d_model * 3)
        return output_shape

    def get_config(self):
        config = {}
        base_config = super(SiameseConcat, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))