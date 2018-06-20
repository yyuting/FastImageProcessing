import tensorflow as tf
from tensor_flow_simplex_matrix import simplex_noise_2arg
import numpy as np

select = lambda a, b, c: a*b + (1-a)*c

def new_mul(x, y):
    try:
        if x == 0.0 or y == 0.0:
            return 0.0
    except:
        pass
    return tf.multiply(x, y)

tf.Tensor.__mul__ = new_mul
tf.Tensor.__rmul__ = new_mul

def simplex_noise(a0, a1, a2, a3, a4, a5, x, y):
    return simplex_noise_2arg(x, y)

def tf_fract(x):
    return tf.floormod(x, 1.0)

def tf_np_wrapper(func):
    def f(x, y=None):
        if isinstance(x, tf.Tensor) or isinstance(y, tf.Tensor):
            if func == 'fmod':
                actual_func = tf.floormod
            else:
                actual_func = getattr(tf, func)
        else:
            actual_func = getattr(np, func)
        if y is None:
            return actual_func(x)
        else:
            return actual_func(x, y)
    return f
