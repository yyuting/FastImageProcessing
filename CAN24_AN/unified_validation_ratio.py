import numpy
import numpy as np
import sys
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

all_shaders = ['mandelbrot', 'mandelbulb', 'trippy_heart', 'primitives_wheel_only']

def main():
    model1 = sys.argv[1]
    model2 = sys.argv[2]
    dst = sys.argv[3]
    
    _, model1_short = os.path.split(model1)
    _, model2_short = os.path.split(model2)
    
    for shader in all_shaders:
        val0 = np.load(os.path.join(model1, '%s_validation.npy' % shader))
        val1 = np.load(os.path.join(model2, '%s_validation.npy' % shader))
        
        assert val0.shape == val1.shape
        
        ratio_all = val0[:, 1] / val1[:, 1]
        ratio_l2 = val0[:, 2] / val1[:, 2]
        ratio_perceptual = val0[:, 3] / val1[:, 3]
        
        figure = plt.figure(figsize=(20,10))

        plt.subplot(3, 1, 1)
        plt.plot(val0[:, 0], ratio_all)
        plt.ylabel('all loss ratio')

        plt.subplot(3, 1, 2)
        plt.plot(val0[:, 0], ratio_l2)
        plt.ylabel('l2 loss ratio')

        plt.subplot(3, 1, 3)
        plt.plot(val0[:, 0], ratio_perceptual)
        plt.ylabel('perceptual loss ratio')
        
        figure.suptitle('%s\n vs \n%s' % (model1_short, model2_short))

        plt.savefig(os.path.join(dst, '%s_validation_ratio.png' % shader))
        plt.close(figure)
        
if __name__ == '__main__':
    main()
        