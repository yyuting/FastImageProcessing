import tensorflow as tf
from tensor_flow_simplex_matrix import simplex_noise_2arg
import numpy as np

select = lambda a, b, c: a*b + (1-a)*c

def simplex_noise(a0, a1, a2, a3, a4, a5, x, y):
    return simplex_noise_2arg(x, y)
    
def tf_fract(x):
    return tf.floormod(x, 1.0)
        
def tf_np_wrapper(func):
    def f(x, y=None):
        if isinstance(x, tf.Tensor) or isinstance(y, tf.Tensor):
            actual_func = getattr(tf, func)
        else:
            actual_func = getattr(np, func)
        if y is None:
            return actual_func(x)
        else:
            return actual_func(x, y)
    return f