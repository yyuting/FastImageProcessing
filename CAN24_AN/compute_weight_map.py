import skimage.io
import skimage
import numpy
import sys
import os
from scipy.ndimage.filters import convolve
import skimage.feature
from scipy.ndimage.morphology import binary_dilation
import tensorflow as tf
import demo

sobel_x = numpy.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
sobel_y = numpy.transpose(sobel_x)

energy_ratio = 0.5

nchannels = 3

def main(dir):

    print(energy_ratio)

    output = tf.placeholder(tf.float32, shape=[1, None, None, None])
    gradient = demo.image_gradients(output)
    sess = tf.Session()

    for prefix in ['test_', 'train_']:

        img_dir = os.path.join(dir, prefix + 'img')
        map_dir = os.path.join(dir, prefix + 'map')
        gradient_dir = os.path.join(dir, prefix + 'grad')
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        if not os.path.exists(gradient_dir):
            os.makedirs(gradient_dir)

        assert os.path.exists(img_dir)
        files = sorted(os.listdir(img_dir))
        for file in files:
            name, ext = os.path.splitext(file)
            assert file.endswith('.png')
            img = skimage.img_as_float(skimage.io.imread(os.path.join(img_dir, file)))
            img_arr = skimage.color.rgb2gray(img)
            #gradient_x = convolve(img_arr, sobel_x)
            #gradient_y = convolve(img_arr, sobel_y)
            #mag = (gradient_x ** 2 + gradient_y ** 2) ** 0.5
            #is_edge = mag >= 0.5
            is_edge = skimage.feature.canny(img_arr)
            #is_edge = binary_dilation(is_edge)
            if nchannels == 1:
                dx, dy = sess.run(gradient, feed_dict={output: numpy.expand_dims(numpy.expand_dims(img_arr, axis=0), axis=3)})
            elif nchannels == 3:
                # change rgb to bgr to be in accordance with opencv read image format in demo.py
                dx, dy = sess.run(gradient, feed_dict={output: numpy.expand_dims(img[...,::-1], axis=0)})
            else:
                raise
            gradient_arr = numpy.concatenate((numpy.expand_dims(numpy.expand_dims(is_edge, axis=0), axis=3), dx, dy), axis=3)
            numpy.save(os.path.join(gradient_dir, name + '.npy'), gradient_arr)
            edge_count = numpy.sum(is_edge)
            all_pix = img_arr.shape[0] * img_arr.shape[1]
            edge_energy = energy_ratio * all_pix / edge_count
            flat_energy = (1 - energy_ratio) * all_pix / (all_pix - edge_count)
            weight_map = flat_energy * numpy.ones(img_arr.shape)
            weight_map[is_edge] = edge_energy
            assert abs(numpy.mean(weight_map) - 1.0) < 1e-8
            numpy.save(os.path.join(map_dir, name + '.npy'), weight_map)

if __name__ == '__main__':
    energy_ratio = float(sys.argv[2])
    nchannels = int(sys.argv[3])
    main(sys.argv[1])
