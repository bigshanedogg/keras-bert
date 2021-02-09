import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as K
from .utils import LayerNormalization, gelu


class MlmLinear(keras.layers.Layer):
    def __init__(self, units, name="mlm_linear", **kwargs):
        self.units = units
        super(MlmLinear, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        hidden_dim = input_shape[0][-1] # d_model
        kernel_initializer = tf.initializers.truncated_normal(stddev=0.02)
        bias_initializer = tf.initializers.zeros()
        self.linear_weights = self.add_weight(shape=(hidden_dim, hidden_dim), dtype=tf.float32, initializer=kernel_initializer, name="linear_weights")
        self.linear_bias = self.add_weight(shape=(hidden_dim,), dtype=tf.float32, initializer=bias_initializer, name="linear_bias")
        self.softmax_bias = self.add_weight(shape=(self.units,), dtype=tf.float32, initializer=bias_initializer, name="softmax_bias")
        super(MlmLinear, self).build(input_shape)

    def call(self, inputs):
        # logits: (batch_size, timesteps, d_model)
        # token_embedding_table: (units, d_model)
        logits = inputs[0]
        token_embedding_table = tf.transpose(inputs[1])

        logits = tf.nn.bias_add(K.dot(logits, self.linear_weights), self.linear_bias)
        logits = keras.layers.Lambda(lambda x : gelu(x))(logits)
        logits = LayerNormalization()(logits)
        logits = tf.nn.bias_add(K.dot(logits, token_embedding_table), self.softmax_bias)
        output = tf.nn.log_softmax(logits, axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        timesteps = input_shape[0][1]
        output_shape = (batch_size, timesteps, self.units)
        return output_shape

    def get_config(self):
        config = {
            "units": self.units,
        }
        base_config = super(MlmLinear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NspLinear(keras.layers.Layer):
    def __init__(self, units=2, name="nsp_linear", **kwargs):
        self.units = units
        super(NspLinear, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        hidden_dim = input_shape[-1] # d_model
        kernel_initializer = tf.initializers.truncated_normal(stddev=0.02)
        bias_initializer = tf.initializers.zeros()
        self.linear_weights = self.add_weight(shape=(hidden_dim, hidden_dim), dtype=tf.float32, initializer=kernel_initializer, name="linear_weights")
        self.linear_bias = self.add_weight(shape=(hidden_dim,), dtype=tf.float32, initializer=bias_initializer, name="linear_bias")
        self.softmax_weights = self.add_weight(shape=(hidden_dim, self.units), dtype=tf.float32, initializer=kernel_initializer, name="softmax_weights")
        self.softmax_bias = self.add_weight(shape=(self.units,), dtype=tf.float32, initializer=bias_initializer, name="softmax_bias")
        super(NspLinear, self).build(input_shape)

    def call(self, inputs):
        logits = tf.squeeze(inputs[:, 0:1, :], axis=1)
        logits = tf.nn.bias_add(K.dot(logits, self.linear_weights), self.linear_bias)
        logits = keras.layers.Activation("tanh")(logits)
        logits = tf.nn.bias_add(K.dot(logits, self.softmax_weights), self.softmax_bias)
        output = tf.nn.log.log_softmax(logits, axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        output_shape = (batch_size, self.units)
        return output_shape

    def get_config(self):
        config = {
            "units": self.units,
        }
        base_config = super(NspLinear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DecoderLinear(keras.layers.Layer):
    def __init__(self, units, shared_embedding=True, name="decoder_linear", **kwargs):
        self.units = units
        self.shared_embedding = shared_embedding
        super(DecoderLinear, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        hidden_dim = input_shape[0][-1] # d_model
        kernel_initializer = tf.initializers.truncated_normal(stddev=0.02)
        bias_initializer = tf.initializers.zeros()
        if self.shared_embedding:
            self.linear_weights = self.add_weight(shape=(hidden_dim, hidden_dim), dtype=tf.float32, initializer=kernel_initializer, name="linear_weights")
            self.linear_bias = self.add_weight(shape=(hidden_dim,), dtype=tf.float32, initializer=bias_initializer, name="linear_bias")
        else:
            self.softmax_weights = self.add_weight(shape=(hidden_dim, self.units,), dtype=tf.float32, initializer=kernel_initializer, name="softmax_weights")
        self.softmax_bias = self.add_weight(shape=(self.units,), dtype=tf.float32, initializer=bias_initializer, name="softmax_bias")
        super(DecoderLinear, self).build(input_shape)

    def call(self, inputs):
        # logits: (batch_size, timesteps, d_model)
        # token_embedding_table: (units, d_model)
        logtis = None
        token_embedding_table = None
        if self.shared_embedding:
            logits = inputs[0]
            token_embedding_table = tf.transpose(inputs[1])
        else:
            logtis = inputs

        if self.shared_embedding:
            logits = tf.nn.bias_add(K.dot(logits, self.linear_weights), self.linear_bias)
            logits = keras.layers.Lambda(lambda x : gelu(x))(logits)
            logits = LayerNormalization()(logits)
            logits = tf.nn.bias_add(K.dot(logits, token_embedding_table), self.softmax_bias)
        else:
            logits = tf.nn.bias_add(K.dot(logits, self.softmax_weights), self.softmax_bias)
        output = tf.nn.log_softmax(logits, axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]
        timesteps = input_shape[0][1]
        output_shape = (batch_size, timesteps, self.units)
        return output_shape

    def get_config(self):
        config = {
            "units": self.units,
            "shared_embedding": self.shared_embedding,
        }
        base_config = super(DecoderLinear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TokenClsLinear(keras.layers.Layer):
    def __init__(self, units, name="token_cls_linear", **kwargs):
        self.units = units
        super(TokenClsLinear, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        hidden_dim = input_shape[-1] # d_model
        kernel_initializer = tf.initializers.truncated_normal(stddev=0.02)
        bias_initializer = tf.initializers.zeros()
        self.linear_weights = self.add_weight(shape=(hidden_dim, hidden_dim), dtype=tf.float32, initializer=kernel_initializer, name="linear_weights")
        self.linear_bias = self.add_weight(shape=(hidden_dim,), dtype=tf.float32, initializer=bias_initializer, name="linear_bias")
        self.softmax_weights = self.add_weight(shape=(hidden_dim, self.units), dtype=tf.float32, initializer=kernel_initializer, name="softmax_weights")
        self.softmax_bias = self.add_weight(shape=(self.units,), dtype=tf.float32, initializer=bias_initializer, name="softmax_bias")
        super(TokenClsLinear, self).build(input_shape)

    def call(self, inputs):
        # logits: (batch_size, timesteps, d_model)
        logits = tf.nn.bias_add(K.dot(inputs, self.linear_weights), self.linear_bias)
        logits = keras.layers.Lambda(lambda x : gelu(x))(logits)
        logits = LayerNormalization()(logits)
        logits = tf.nn.bias_add(K.dot(logits, self.softmax_weights), self.softmax_bias)
        output = tf.nn.log_softmax(logits, axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        timesteps = input_shape[1]
        output_shape = (batch_size, timesteps, self.units)
        return output_shape

    def get_config(self):
        config = {
            "units": self.units,
        }
        base_config = super(TokenClsLinear, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))