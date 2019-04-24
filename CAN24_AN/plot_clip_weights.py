import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import sys
import skimage.measure
import numpy
import numpy as np
import skimage.io
from skimage.morphology import disk, dilation
import skimage.feature

#os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
#os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
#os.system('rm tmp')

MSE_ONLY = False

#TEST_GROUND = '/localtmp/yuting/out_1x_1_sample/train/zigzag_plane_normal_spheres/test_img'
#TRAIN_GROUND = '/localtmp/yuting/out_1x_1_sample/train/zigzag_plane_normal_spheres/train_img'
#TEST_GROUND = '/bigtemp/yy2bb/out_4_sample_features_random_camera_uniform_all_features/zigzag_plane_normal_spheres/datas/test_img'
#TRAIN_GROUND = '/bigtemp/yy2bb/out_4_sample_features_random_camera_uniform_all_features/zigzag_plane_normal_spheres/datas/train_img'
#TEST_GROUND = '/localtmp/yuting/datas_features_only/datas_feature_only/test_img'
#TRAIN_GROUND = '/localtmp/yuting/datas_features_only/datas_feature_only/train_img'

if not MSE_ONLY:
    sys.path += ['../../PerceptualSimilarity']

    model = None

    #cwd = os.getcwd()
    #os.chdir('../../PerceptualSimilarity')
    #model = dm.DistModel()
    #model.initialize(model='net-lin',net='alex',use_gpu=True, spatial=True)
    #print('Model [%s] initialized'%model.name())
    #os.chdir(cwd)

def compute_metric(dir1, dir2, mode, mask=None, thre=0):

    if mode == 'perceptual':
        from models import dist_model as dm
        import torch
        from util import util
        global model
        if model is None:
            cwd = os.getcwd()
            os.chdir('../../PerceptualSimilarity')
            model = dm.DistModel()
            #model.initialize(model='net-lin',net='alex',use_gpu=True, spatial=True)
            model.initialize(model='net-lin',net='alex',use_gpu=True)
            print('Model [%s] initialized'%model.name())
            os.chdir(cwd)

    if mode == 'perceptual_tf':
        sys.path += ['../../lpips-tensorflow']
        import lpips_tf
        import tensorflow as tf
        image0_ph = tf.placeholder(tf.float32, [1, None, None, 3])
        image1_ph = tf.placeholder(tf.float32, [1, None, None, 3])
        distance_t = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex')
        sess = tf.Session()

    if mode == 'l2_with_gradient':
        import demo
        import tensorflow as tf
        output = tf.placeholder(tf.float32, shape=[1, None, None, None])
        gradient = demo.image_gradients(output)
        sess = tf.Session()

    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)
    img_files1 = sorted([file for file in files1 if file.endswith('.png') or file.endswith('.jpg')])
    img_files2 = sorted([file for file in files2 if file.endswith('.png') or file.endswith('.jpg')])
    if mask is not None:
        mask_files = []
        for dir in sorted(mask):
            files3 = os.listdir(dir)
            add_mask_files = sorted([os.path.join(dir, file) for file in files3 if file.startswith('mask')])
            if len(add_mask_files) == 0:
                files_to_dilate = sorted([file for file in files3 if file.startswith('g_intermediates')])
                for file in files_to_dilate:
                    mask_arr = numpy.load(os.path.join(dir, file))
                    mask_arr = numpy.squeeze(mask_arr)
                    mask_arr = mask_arr >= thre
                    dilated_mask = dilation(mask_arr, disk(10))
                    dilated_filename = 'mask_' + file
                    numpy.save(os.path.join(dir, dilated_filename), dilated_mask.astype('f'))
                    add_mask_files.append(os.path.join(dir, dilated_filename))
            mask_files += add_mask_files
        print(mask_files)
        assert len(mask_files) == len(img_files1)

    assert len(img_files1) == len(img_files2)

    # locate GT gradient directory
    if mode == 'l2_with_gradient':
        head, tail = os.path.split(dir2)
        gradient_gt_dir = os.path.join(head, tail[:-3] + 'grad')
        if not os.path.exists(gradient_gt_dir):
            printf("dir not found,", gradient_gt_dir)
            raise
        gradient_gt_files = os.listdir(gradient_gt_dir)
        gradient_gt_files = sorted([file for file in gradient_gt_files if file.endswith('.npy')])
        assert len(img_files1) == len(gradient_gt_files)

    vals = numpy.empty(len(img_files1))
    #if mode == 'perceptual':
    #    global model

    for ind in range(len(img_files1)):
        if mode == 'ssim' or mode == 'l2' or mode == 'l2_with_gradient':
            img1 = skimage.img_as_float(skimage.io.imread(os.path.join(dir1, img_files1[ind])))
            img2 = skimage.img_as_float(skimage.io.imread(os.path.join(dir2, img_files2[ind])))
            if mode == 'ssim':
                #vals[ind] = skimage.measure.compare_ssim(img1, img2, datarange=img2.max()-img2.min(), multichannel=True)
                metric_val = skimage.measure.compare_ssim(img1, img2, datarange=img2.max()-img2.min(), multichannel=True)
            else:
                #vals[ind] = numpy.mean((img1 - img2) ** 2) * 255.0 * 255.0
                metric_val = ((img1 - img2) ** 2) * 255.0 * 255.0
            if mode == 'l2_with_gradient':
                metric_val = numpy.mean(metric_val, axis=2)
                gradient_gt = numpy.load(os.path.join(gradient_gt_dir, gradient_gt_files[ind]))
                dx, dy = sess.run(gradient, feed_dict={output: numpy.expand_dims(img1[...,::-1], axis=0)})
                #is_edge = skimage.feature.canny(skimage.color.rgb2gray(img1))
                dx_ground = gradient_gt[:, :, :, 1:4]
                dy_ground = gradient_gt[:, :, :, 4:]
                edge_ground = gradient_gt[:, :, :, 0]
                gradient_loss_term = numpy.mean((dx - dx_ground) ** 2.0 + (dy - dy_ground) ** 2.0, axis=3)
                metric_val += numpy.squeeze(0.2 * 255.0 * 255.0 * gradient_loss_term * edge_ground * edge_ground.size / numpy.sum(edge_ground))

            #if mode == 'l2' and mask is not None:
            #    img_diff = (img1 - img2) ** 2.0
            #    mask_img = numpy.load(mask_files[ind])
            #    img_diff *= numpy.expand_dims(mask_img, axis=2)
            #    vals[ind] = (numpy.sum(img_diff) / numpy.sum(mask_img * 3)) * 255.0 * 255.0
        elif mode == 'perceptual':
            img1 = util.im2tensor(util.load_image(os.path.join(dir1, img_files1[ind])))
            img2 = util.im2tensor(util.load_image(os.path.join(dir2, img_files2[ind])))
            #vals[ind] = numpy.mean(model.forward(img1, img2)[0])
            metric_val = numpy.expand_dims(model.forward(img1, img2), axis=2)
        elif mode == 'perceptual_tf':
            img1 = np.expand_dims(skimage.img_as_float(skimage.io.imread(os.path.join(dir1, img_files1[ind]))), axis=0)
            img2 = np.expand_dims(skimage.img_as_float(skimage.io.imread(os.path.join(dir2, img_files2[ind]))), axis=0)
            metric_val = sess.run(distance_t, feed_dict={image0_ph: img1, image1_ph: img2})
        else:
            raise

        if mask is not None:
            assert mode in ['l2', 'perceptual']
            mask_img = numpy.load(mask_files[ind])
            metric_val *= numpy.expand_dims(mask_img, axis=2)
            vals[ind] = numpy.sum(metric_val) / (numpy.sum(mask_img) * metric_val.shape[2])
        else:
            vals[ind] = numpy.mean(metric_val)

    mode = mode + ('_mask' if mask is not None else '')
    filename_all = mode + '_all.txt'
    filename_breakdown = mode + '_breakdown.txt'
    filename_single = mode + '.txt'
    numpy.savetxt(os.path.join(dir1, filename_all), vals, fmt="%f, ")
    target=open(os.path.join(dir1, filename_single),'w')
    target.write("%f"%numpy.mean(vals))
    target.close()
    if len(img_files1) == 30:
        target=open(os.path.join(dir1, filename_breakdown),'w')
        target.write("%f, %f, %f"%(numpy.mean(vals[:5]), numpy.mean(vals[5:10]), numpy.mean(vals[10:])))
        target.close()
    if mode in ['l2_with_gradient', 'perceptual_tf']:
        sess.close()
    return vals

def compute_ssim(dir1, dir2):
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)
    img_files1 = sorted([file for file in files1 if file.endswith('.png') or file.endswith('.jpg')])
    img_files2 = sorted([file for file in files2 if file.endswith('.png') or file.endswith('.jpg')])
    #img_files1 = sorted([file for file in files1 if file.endswith('.png') or file.endswith('synthesized_image.jpg')])
    #img_files2 = sorted([file for file in files1 if file.endswith('.png') or file.endswith('real_image.jpg')])
    assert len(img_files1) == len(img_files2)

    vals = numpy.empty(len(img_files1))

    for ind in range(len(img_files1)):
        img1 = skimage.img_as_float(skimage.io.imread(os.path.join(dir1, img_files1[ind])))
        img2 = skimage.img_as_float(skimage.io.imread(os.path.join(dir2, img_files2[ind])))
        vals[ind] = skimage.measure.compare_ssim(img1, img2, datarange=img2.max()-img2.min(), multichannel=True)

    numpy.savetxt(os.path.join(dir1, 'ssim_all.txt'), vals, fmt="%f, ")
    target=open(os.path.join(dir1, 'ssim.txt'),'w')
    target.write("%f"%numpy.mean(vals))
    target.close()
    if len(img_files1) == 30:
        target=open(os.path.join(dir1, 'ssim_breakdown.txt'),'w')
        target.write("%f, %f, %f"%(numpy.mean(vals[:5]), numpy.mean(vals[5:10]), numpy.mean(vals[10:])))
        target.close()
    return vals

def compute_perceptual(dir1, dir2):
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)
    img_files1 = sorted([file for file in files1 if file.endswith('.png') or file.endswith('.jpg')])
    img_files2 = sorted([file for file in files2 if file.endswith('.png') or file.endswith('.jpg')])
    #img_files1 = sorted([file for file in files1 if file.endswith('.png') or file.endswith('synthesized_image.jpg')])
    #img_files2 = sorted([file for file in files1 if file.endswith('.png') or file.endswith('real_image.jpg')])
    assert len(img_files1) == len(img_files2)

    vals = numpy.empty(len(img_files1))

    global model

    for ind in range(len(img_files1)):
        img1 = util.im2tensor(util.load_image(os.path.join(dir1, img_files1[ind])))
        img2 = util.im2tensor(util.load_image(os.path.join(dir2, img_files2[ind])))
        vals[ind] = model.forward(img1, img2)[0]

    numpy.savetxt(os.path.join(dir1, 'perceptual_all.txt'), vals, fmt="%f, ")
    target=open(os.path.join(dir1, 'perceptual.txt'),'w')
    target.write("%f"%numpy.mean(vals))
    target.close()
    if len(img_files1) == 30:
        target=open(os.path.join(dir1, 'perceptual_breakdown.txt'),'w')
        target.write("%f, %f, %f"%(numpy.mean(vals[:5]), numpy.mean(vals[5:10]), numpy.mean(vals[10:])))
        target.close()
    return vals

def get_score(name):
    train_x = []
    train_y = []
    test_all_y = []
    test_close_y = []
    test_far_y = []
    test_middle_y = []
    test_x = []
    if not MSE_ONLY:
        train_ssim_y = []
        test_all_ssim_y = []
        test_close_ssim_y = []
        test_far_ssim_y = []
        test_middle_ssim_y = []
        train_perceptual_y = []
        test_all_perceptual_y = []
        test_close_perceptual_y = []
        test_far_perceptual_y = []
        test_middle_perceptual_y = []
    dirs = sorted(os.listdir(name))
    for dir in dirs:
        if dir.startswith('test_pct_norm'):
            test_x.append(float(dir.replace('test_pct_norm', '')) / 10)
            score_file = os.path.join(name, dir, 'score.txt')
            test_all_y.append(float(open(score_file).read()))
            score_breakdown_file = os.path.join(name, dir, 'score_breakdown.txt')
            vals = open(score_breakdown_file).read().split(',')
            test_close_y.append(float(vals[0]))
            test_far_y.append(float(vals[1]))
            test_middle_y.append(float(vals[2]))

            if not MSE_ONLY:
                ssim_file = os.path.join(name, dir, 'ssim.txt')
                ssim_breakdown_file = os.path.join(name, dir, 'ssim_breakdown.txt')
                if not (os.path.exists(ssim_file) and os.path.exists(ssim_breakdown_file)):
                    compute_ssim(os.path.join(name, dir), TEST_GROUND)
                test_all_ssim_y.append(float(open(ssim_file).read()))

                perceptual_file = os.path.join(name, dir, 'perceptual.txt')
                perceptual_breakdown_file = os.path.join(name, dir, 'perceptual_breakdown.txt')
                if not os.path.exists(perceptual_file):
                    compute_perceptual(os.path.join(name, dir), TEST_GROUND)
                test_all_perceptual_y.append(float(open(perceptual_file).read()))

                ssim_vals = open(ssim_breakdown_file).read().split(', ')
                test_close_ssim_y.append(float(ssim_vals[0]))
                test_far_ssim_y.append(float(ssim_vals[1]))
                test_middle_ssim_y.append(float(ssim_vals[2]))

                perceptual_vals = open(perceptual_breakdown_file).read().split(', ')
                test_close_perceptual_y.append(float(perceptual_vals[0]))
                test_far_perceptual_y.append(float(perceptual_vals[1]))
                test_middle_perceptual_y.append(float(perceptual_vals[2]))

        elif dir.startswith('train_pct_norm'):
            train_x.append(float(dir.replace('train_pct_norm', '')) / 10)
            score_file = os.path.join(name, dir, 'score.txt')
            train_y.append(float(open(score_file).read()))

            if not MSE_ONLY:
                ssim_file = os.path.join(name, dir, 'ssim.txt')
                if not os.path.exists(ssim_file):
                    compute_ssim(os.path.join(name, dir), TRAIN_GROUND)
                train_ssim_y.append(float(open(ssim_file).read()))

                perceptual_file = os.path.join(name, dir, 'perceptual.txt')
                if not os.path.exists(perceptual_file):
                    compute_perceptual(os.path.join(name, dir), TRAIN_GROUND)
                train_perceptual_y.append(float(open(perceptual_file).read()))

    if len(train_x) == 0:
        #compute_ssim(name, sys.argv[2])
        #compute_perceptual(name, sys.argv[2])
        if len(sys.argv) > 3:
            mode = sys.argv[3]
        else:
            mode = None
        if mode in ['l2', 'ssim', 'perceptual', 'l2_with_gradient', 'perceptual_tf']:
            print('running mode', mode)
            compute_metric(name, sys.argv[2], mode)
        else:
            print('running all mode')
            compute_metric(name, sys.argv[2], 'ssim')
            compute_metric(name, sys.argv[2], 'perceptual')
            compute_metric(name, sys.argv[2], 'l2')
        #if len(sys.argv) > 4:
        #    compute_metric(name, sys.argv[2], 'l2', mask=sys.argv[3], thre=sys.argv[4])
        return

    figure = pyplot.figure()
    pyplot.plot(train_x, train_y, label='train_loss')
    pyplot.plot(test_x, test_all_y, label='test_all_loss')
    pyplot.plot(test_x, test_close_y, label='test_close_loss')
    pyplot.plot(test_x, test_far_y, label='test_far_loss')
    pyplot.plot(test_x, test_middle_y, label='test_middle_loss')
    pyplot.legend()
    #pyplot.ylim((50, 1000))
    figure.savefig(os.path.join(name, 'clip_weights_mse.png'))
    pyplot.ylim((50, 200))
    figure.savefig(os.path.join(name, 'clip_weights_mse_200.png'))
    pyplot.ylim((50, 1000))
    figure.savefig(os.path.join(name, 'clip_weights_mse_1000.png'))
    pyplot.close(figure)

    if not MSE_ONLY:
        figure = pyplot.figure()
        pyplot.plot(train_x, train_ssim_y, label='train_ssim')
        pyplot.plot(test_x, test_all_ssim_y, label='test_all_ssim')
        pyplot.plot(test_x, test_close_ssim_y, label='test_close_ssim')
        pyplot.plot(test_x, test_far_ssim_y, label='test_far_ssim')
        pyplot.plot(test_x, test_middle_ssim_y, label='test_middle_ssim')
        pyplot.legend()
        #pyplot.ylim((0.5, 1.0))
        figure.savefig(os.path.join(name, 'clip_weights_ssim.png'))
        pyplot.close(figure)

        figure = pyplot.figure()
        pyplot.plot(train_x, train_perceptual_y, label='train_perceptual')
        pyplot.plot(test_x, test_all_perceptual_y, label='test_all_perceptual')
        pyplot.plot(test_x, test_close_perceptual_y, label='test_close_perceptual')
        pyplot.plot(test_x, test_far_perceptual_y, label='test_far_perceptual')
        pyplot.plot(test_x, test_middle_perceptual_y, label='test_middle_perceptual')
        pyplot.legend()
        figure.savefig(os.path.join(name, 'clip_weights_perceptual.png'))
        pyplot.close(figure)

if __name__ == '__main__':
    get_score(sys.argv[1])
