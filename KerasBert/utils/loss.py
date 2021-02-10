import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as K

def mlm_loss(y_true, y_pred):
    label_weights = K.argmax(y_true, axis=-1)
    label_weights = tf.cast(K.not_equal(label_weights, 0), dtype=tf.float32)
    numerator = -1 * tf.reduce_sum(y_true, y_pred, axis=[-1])
    numerator = tf.reduce_sum(label_weights * numerator)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator
    return loss


def nsp_loss(y_true, y_pred):
    loss = -1 * tf.reduce_sum(y_true * y_pred, axis=-1)
    loss = tf.reduce_mean(loss)
    return loss


def triplet_loss(y_true, y_pred):
    # triplet loss is included in SentenceBertTriplet model with tensorflow implementation
    epsilon = 1.0
    anchor_pred = y_pred[:, 0, :]
    positive_pred = y_pred[:, 1, :]
    negative_pred = y_pred[:, 2, :]
    distance_a_p = tf.sqrt(tf.reduce_sum(tf.square(anchor_pred - positive_pred), axis=-1))
    distance_a_n = tf.sqrt(tf.reduce_sum(tf.square(anchor_pred - negative_pred), axis=-1))
    loss = tf.maximum(distance_a_p - distance_a_n, 0)
    return loss


def weighted_categorical_crossentropy(weights, name=None):
    weights = K.variable(weights)
    def func(y_true, y_pred):
        # scale predictions to make the sum of probabilities be equal to 1
        y_pred = y_pred / K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent Nan's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), (1 - K.epsilon()))
        # calcuation
        loss = y_true * K.log(y_pred) * weights
        loss = - K.sum(loss, axis=-1)
        return loss

    func.__name__ = "loss"
    if name is not None: func.__name__ = name
    return func

