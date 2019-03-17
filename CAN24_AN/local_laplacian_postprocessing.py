import tensorflow as tf
import os
import sys
sys.path += ['../../differentiable_proxy']
from local_laplacian_tf import local_laplacian_tf
os.environ['CUDA_VISIBLE_DEVICES']='1'
import skimage.io
import skimage
import numpy

dir = 'trippy'
min_res = 128

def main():
    input = tf.placeholder(tf.float32, [None, None, 3])
    output = local_laplacian_tf(tf.expand_dims(input, axis=0))
    sess = tf.Session()
    files = os.listdir(dir)
    for file in files:
        if file.startswith('t') and file.endswith('.png'):
            in_img = skimage.img_as_float(skimage.io.imread(os.path.join(dir, file)))
            if in_img.shape[0] % min_res != 0:
                #dif = int((in_img.shape[0] % min_res) / 2)
                new_res = min_res * int(numpy.floor(in_img.shape[0] / min_res))
                #in_img = in_img[dif:dif+new_res, :, :]
                in_img = in_img[:new_res, :, :]
            if in_img.shape[1] % min_res != 0:
                #dif = int((in_img.shape[1] % min_res) / 2)
                new_res = min_res * int(numpy.floor(in_img.shape[1] / min_res))
                #in_img = in_img[:, dif:dif+new_res, :]
                in_img = in_img[:, :new_res, :]
            out_img = sess.run(output, feed_dict={input: in_img})
            out_img = numpy.clip(numpy.squeeze(out_img), 0.0, 1.0)
            skimage.io.imsave(os.path.join(dir, 'local_laplacian_' + file), out_img)

if __name__ == '__main__':
    main()