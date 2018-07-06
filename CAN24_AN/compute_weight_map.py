import skimage.io
import skimage
import numpy
import sys
import os
from scipy.ndimage.filters import convolve

sobel_x = numpy.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
sobel_y = numpy.transpose(sobel_x)

energy_ratio = 0.5

def main(dir):

    print(energy_ratio)

    for prefix in ['test_', 'train_']:

        img_dir = os.path.join(dir, prefix + 'img')
        map_dir = os.path.join(dir, prefix + 'map')
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)

        assert os.path.exists(img_dir)
        files = sorted(os.listdir(img_dir))
        for file in files:
            name, ext = os.path.splitext(file)
            assert file.endswith('.png')
            img_arr = skimage.color.rgb2gray(skimage.io.imread(os.path.join(img_dir, file)))
            gradient_x = convolve(img_arr, sobel_x)
            gradient_y = convolve(img_arr, sobel_y)
            mag = (gradient_x ** 2 + gradient_y ** 2) ** 0.5
            is_edge = mag >= 0.5
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
    main(sys.argv[1])
