import tensorflow as tf
import keras
import keras.backend.tensorflow_backend as K

def pearson_correlation(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x-mx, y-my
    r_numerator = K.sum(xm*ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_denominator = K.sqrt(x_square_sum * y_square_sum)
    correlation = K.mean(r_numerator / r_denominator)
    return correlation