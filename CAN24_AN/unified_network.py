"""
testing command:
python unified_network.py --name test_unified --add_initial_layers --initial_layer_channels 48 --add_final_layers --final_layer_channels 48 --conv_channel_multiplier 0 --conv_channel_no 48 --dilation_remove_layer --identity_initialize --no_identity_output_layer --efficient_trace --collect_loop_statistic --lpips_loss --lpips_loss_scale 0.04 --tile_only --tiled_h 320 --tiled_w 320 --use_batch --batch_size 6 --render_sigma 0.3 --epoch 1 --save_frequency 1 --feature_normalize_lo_pct 5 --relax_clipping --is_train --dataroot_parent /shaderml/playground/ --no_preload

current progress:
need to sort out the logic on which checkpoints to read from
"""

from __future__ import division

import gpu_util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_util.pick_gpu_lowest_memory())

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
plt = pyplot

import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import numpy
import numpy.random
import random
import argparse_util
import pickle
from tensorflow.python.client import timeline
import copy
from shaders import *
import sys; sys.path += ['../../global_opt/proj/apps']
import importlib
import importlib.util
import subprocess
import shutil
from tf_util import *
import json
import read_timeline
import glob

import warnings
import skimage
import skimage.io
import skimage.transform
import scipy.ndimage

import copy
import gc

allowed_dtypes = ['float64', 'float32', 'uint8']
no_L1_reg_other_layers = True

width = 500
height = 400

allow_nonzero = False

identity_output_layer = True

less_aggresive_ini = False

conv_padding = "SAME"
padding_offset = 32

shaders_pool = [
    [
        ('mandelbrot', 'plane', 'datas_mandelbrot', {'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True}), 
        ('mandelbulb', 'none', 'datas_mandelbulb', {'fov': 'small_seperable'}), 
        ('trippy_heart', 'plane', 'datas_trippy_subsample_2', {'every_nth': 2, 'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True}), 
        ('primitives_wheel_only', 'none', 'datas_primitives', {'fov': 'small'})
    ],
    [
        ('mandelbrot', 'plane', 'datas_mandelbrot', {'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True}), 
        ('mandelbulb_slim', 'none', 'datas_mandelbulb', {'fov': 'small_seperable'}), 
        ('trippy_heart', 'plane', 'datas_trippy_subsample_2', {'every_nth': 2, 'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True}), 
        ('primitives_wheel_only', 'none', 'datas_primitives', {'fov': 'small'})
    ],
    [
        ('mandelbrot', 'plane', 'datas_mandelbrot', {'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True, 'specified_ind': 'subsample_taylor_exp_vals.npy2.npy'}), 
        ('mandelbulb_slim', 'none', 'datas_mandelbulb', {'fov': 'small_seperable'}), 
        ('trippy_heart', 'plane', 'datas_trippy_subsample_2', {'every_nth': 2, 'fov': 'small', 'additional_features': False, 'ignore_last_n_scale': 7, 'include_noise_feature': True, 'specified_ind': 'subsample_taylor_exp_vals.npy4.npy'}), 
        ('primitives_wheel_only', 'none', 'datas_primitives', {'fov': 'small'})
    ]
]

all_shaders = shaders_pool[0]

all_shaders_aux = [
    ('mandelbrot', 'plane', 'datas_mandelbrot', {'fov': 'small'}), 
    ('mandelbulb', 'none', 'datas_mandelbulb', {'fov': 'small_seperable'}), 
    ('trippy_heart', 'plane', 'datas_trippy_subsample_2', {'every_nth': 2, 'fov': 'small'}), 
    ('primitives_wheel_only', 'none', 'datas_primitives', {'fov': 'small'})]

def smart_mkdir(dir):
    if os.path.isdir(dir):
        print('dir already exists', dir)
        return
    
    parent, _ = os.path.split(dir)
    parent_stack = []
    
    while not os.path.isdir(parent):
        parent_stack.append(parent)
        parent, _ = os.path.split(parent)
        
    for pa in parent_stack[::-1]:
        os.mkdir(pa)
   
    os.mkdir(dir)


def get_tensors(dataroot, name, camera_pos, shader_time, output_type='remove_constant', nsamples=1, shader_name='zigzag', geometry='plane', color_inds=[], manual_features_only=False, efficient_trace=False, collect_loop_statistic=False, h_start=0, h_offset=height, w_start=0, w_offset=width, samples=None, fov='regular', t_sigma=1/60.0, first_last_only=False, last_only=False, subsample_loops=-1, last_n=-1, first_n=-1, first_n_no_last=-1, mean_var_only=False, zero_samples=False, render_fix_spatial_sample=False, render_zero_spatial_sample=False, spatial_samples=None, every_nth=-1, every_nth_stratified=False, additional_features=True, ignore_last_n_scale=0, include_noise_feature=False, no_noise_feature=False, relax_clipping=False, render_sigma=None, same_sample_all_pix=False, texture_maps=[], use_dataroot=True, automatic_subsample=False, automate_raymarching_def=False, log_only_return_def_raymarching=True, debug=[], SELECT_FEATURE_THRE=200, compiler_problem_idx=-1, feature_normalize_lo_pct=20, get_col_aux_inds=False, specified_ind=None, write_file=True, alt_dir=''):
    # 2x_1sample on margo
    #camera_pos = np.load('/localtmp/yuting/out_2x1_manual_carft/train.npy')[0, :]

    #feature_scale = np.load('/localtmp/yuting/out_2x1_manual_carft/train/zigzag_plane_normal_spheres/datas_rescaled_25_75_2_153/feature_scale.npy')
    #feature_bias = np.load('/localtmp/yuting/out_2x1_manual_carft/train/zigzag_plane_normal_spheres/datas_rescaled_25_75_2_153/feature_bias.npy')

    manual_features_only = manual_features_only

    if output_type not in ['rgb', 'bgr']:
        if use_dataroot:
            
            hi_pct = 100 - feature_normalize_lo_pct
            feature_scale_file = os.path.join(dataroot, 'feature_scale_%d_%d.npy' % (feature_normalize_lo_pct, hi_pct))
            feature_bias_file = os.path.join(dataroot, 'feature_bias_%d_%d.npy' % (feature_normalize_lo_pct, hi_pct))
            
            if os.path.exists(feature_scale_file) and os.path.exists(feature_bias_file):
                feature_scale = np.load(feature_scale_file)
                feature_bias = np.load(feature_bias_file)
            else:
                print('-----------------------------------------------')
                print('WARNING: featue_scale and feature_bias with corresponding lo/hi pct label not found, using default files instead')
                feature_scale = np.load(os.path.join(dataroot, 'feature_scale.npy'))
                feature_bias = np.load(os.path.join(dataroot, 'feature_bias.npy'))
        else:
            feature_scale = 1.0
            feature_bias = 0.0

        #Q1 = np.load(os.path.join(dataroot, 'Q1.npy'))
        #Q3 = np.load(os.path.join(dataroot, 'Q3.npy'))
        #IQR = np.load(os.path.join(dataroot, 'IQR.npy'))
        tolerance = 2.0

    if compiler_problem_idx < 0:
        compiler_problem_full_name = os.path.abspath(os.path.join(name, 'compiler_problem.py'))
    else:
        compiler_problem_full_name = os.path.abspath(os.path.join(name, 'compiler_problem%d.py' % compiler_problem_idx))
    if not os.path.exists(compiler_problem_full_name):
        if shader_name == 'zigzag':
            shader_args = ' render_zigzag ' + geometry + ' spheres '
        elif shader_name == 'sin_quadratic':
            shader_args = ' render_sin_quadratic ' + geometry + ' ripples '
        elif shader_name == 'bricks':
            shader_args = ' render_bricks ' + geometry + ' none '
        elif shader_name in ['mandelbrot', 'mandelbrot_tile_radius']:
            shader_args = ' render_mandelbrot_tile_radius ' + geometry + ' none '
        elif shader_name == 'mandelbrot_simplified_proxy':
            shader_args = ' render_mandelbrot_tile_radius_short_05 ' + geometry + ' none '
        elif shader_name == 'fire':
            shader_args = ' render_fire ' + geometry + ' spheres '
        elif shader_name == 'marble':
            shader_args = ' render_marble ' + geometry + ' ripples '
        elif shader_name == 'marble_manual_pick':
            shader_args = ' render_marble_manual_pick ' + geometry + ' ripples '
        elif shader_name == 'marble_def_color':
            shader_args = ' render_marble_def_color ' + geometry + ' ripples '
        elif shader_name == 'mandelbulb':
            shader_args = ' render_mandelbulb ' + geometry + ' none'
        elif shader_name == 'mandelbulb_slim':
            shader_args = ' render_mandelbulb_slim ' + geometry + ' none'
        elif shader_name == 'mandelbulb_slim_simplified_proxy':
            shader_args = ' render_mandelbulb_slim_simplified_proxy ' + geometry + ' none'
        elif shader_name == 'mandelbulb_simplified_proxy':
            shader_args = ' render_mandelbulb_simplified_proxy ' + geometry + ' none'
        elif shader_name == 'wood':
            shader_args = ' render_wood_real ' + geometry + ' none'
        elif shader_name == 'wood_staggered':
            shader_args = ' render_wood_staggered ' + geometry + ' none'
        elif shader_name == 'primitives_aliasing':
            shader_args = ' render_primitives_aliasing ' + geometry + ' none'
        elif shader_name == 'primitives_wheel_only':
            shader_args = ' render_primitives_wheel_only ' + geometry + ' none'
        elif shader_name == 'trippy_heart':
            shader_args = ' render_trippy_heart ' + geometry + ' none'
        elif shader_name == 'trippy_heart_simplified_proxy':
            shader_args = ' render_trippy_heart_simplified_proxy ' + geometry + ' none'
        elif shader_name == 'oceanic':
            shader_args = ' render_oceanic_simple ' + geometry + ' none'
        elif shader_name == 'oceanic_simplified_proxy':
            shader_args = ' render_oceanic_simplified_proxy ' + geometry + ' none'
        elif shader_name == 'texture_map':
            shader_args = ' render_texture_map ' + geometry + ' none'
        elif shader_name == 'oceanic_simple_generate_def':
            shader_args = ' render_oceanic_simple_generate_def ' + geometry + ' none'
        elif shader_name == 'oceanic_simple_all_raymarching':
            shader_args = ' render_oceanic_simple_all_raymarching ' + geometry + ' none'
        elif shader_name == 'venice_simplified_proxy':
            shader_args = ' render_venice_simplified_proxy ' + geometry + ' none'
        elif shader_name == 'venice_simplified_proxy_30_70':
            shader_args = ' render_venice_simplified_proxy_30_70 ' + geometry + ' none'
        elif shader_name == 'venice':
            shader_args = ' render_venice ' + geometry + ' none'
        elif shader_name == 'plane_basics':
            shader_args = ' render_plane_basics ' + geometry + ' none'
        else:
            shader_args = ' render_' + shader_name + ' ' + geometry + ' none' 

        render_util_dir = os.path.abspath('../../global_opt/proj/apps')
        render_single_full_name = os.path.abspath(os.path.join(render_util_dir, 'render_single.py'))
        cwd = os.getcwd()
        os.chdir(render_util_dir)
        render_single_cmd = 'python ' + render_single_full_name + ' ' + os.path.join(cwd, name) + shader_args + ' --is-tf --code-only --log-intermediates --no_compute_g --SELECT_FEATURE_THRE %d ' % SELECT_FEATURE_THRE

        render_single_cmd = render_single_cmd + ' --log_intermediates_subset_level 1'
        if collect_loop_statistic:
            render_single_cmd = render_single_cmd + ' --collect_loop_statistic'
        if first_last_only:
            render_single_cmd = render_single_cmd + ' --first_last_only'
        if last_only:
            assert not first_last_only
            render_single_cmd = render_single_cmd + ' --last_only'
        if subsample_loops > 0:
            assert not first_last_only
            assert not last_only
            render_single_cmd = render_single_cmd + ' --subsample_loops ' + str(subsample_loops)
        if last_n > 0:
            assert not first_last_only
            assert not last_only
            assert subsample_loops < 0
            render_single_cmd = render_single_cmd + ' --last_n ' + str(last_n)
        if first_n > 0:
            assert not first_last_only
            assert not last_only
            assert subsample_loops < 0
            assert last_n < 0
            render_single_cmd = render_single_cmd + ' --first_n ' + str(first_n)
        if first_n_no_last > 0:
            assert not first_last_only
            assert not last_only
            assert subsample_loops < 0
            assert last_n < 0
            assert first_n < 0
            render_single_cmd = render_single_cmd + ' --first_n_no_last ' + str(first_n_no_last)
        if mean_var_only:
            assert not first_last_only
            assert not last_only
            assert subsample_loops < 0
            assert last_n < 0
            assert first_n < 0
            assert first_n_no_last < 0
            render_single_cmd = render_single_cmd + ' --mean_var_only'
        if every_nth > 0:
            render_single_cmd = render_single_cmd + ' --every_nth ' + str(every_nth)
        if every_nth_stratified:
            stratified_random_file = os.path.join(dataroot, 'stratified_random_file.npy')
            render_single_cmd = render_single_cmd + ' --every_nth_stratified --stratified_random_file ' + stratified_random_file

        if automatic_subsample:
            render_single_cmd = render_single_cmd + ' --automatic_subsample'
        if automatic_subsample or automate_raymarching_def:
            render_single_cmd = render_single_cmd + ' --automate_raymarching_def'
        if log_only_return_def_raymarching:
            render_single_cmd = render_single_cmd + ' --log_only_return_def_raymarching'
        entire_cmd = 'cd ' + render_util_dir + ' && ' + render_single_cmd + ' && cd ' + cwd
        ans = os.system(entire_cmd)
        #ans = subprocess.call('cd ' + render_util_dir + ' && source activate py36 && python ' + render_single_full_name + ' out ' + shader_args + ' --is-tf --code-only --log-intermediates && source activate tensorflow35 && cd ' + cwd)

        print(ans)
        os.chdir(cwd)
        #compiler_problem_old = os.path.abspath('../../global_opt/proj/apps/compiler_problem.py')
        #os.rename(compiler_problem_old, compiler_problem_full_name)

        if compiler_problem_idx >= 0:
            os.rename(os.path.join(name, 'compiler_problem.py'), compiler_problem_full_name)
        
    spec = importlib.util.spec_from_file_location("module.name", compiler_problem_full_name)
    compiler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler_module)
    
    texture_map_size = []
    
    feature_pl = []

    features, vec_output, manual_features = get_render(camera_pos, shader_time, nsamples=nsamples, shader_name=shader_name, geometry=geometry, return_vec_output=True, compiler_module=compiler_module, manual_features_only=manual_features_only, h_start=h_start, h_offset=h_offset, w_start=w_start, w_offset=w_offset, samples=samples, fov=fov, t_sigma=t_sigma, zero_samples=zero_samples, render_fix_spatial_sample=render_fix_spatial_sample, render_zero_spatial_sample=render_zero_spatial_sample, spatial_samples=spatial_samples, additional_features=additional_features, include_noise_feature=include_noise_feature, no_noise_feature=no_noise_feature, render_sigma=render_sigma, same_sample_all_pix=same_sample_all_pix, texture_maps=texture_maps, texture_map_size=texture_map_size, debug=debug, phase=compiler_problem_idx)
    
    
    # workaround if for some feature sparsification setup, RGB channels are not logged
    # also prevent aux feature from not being logged
    if efficient_trace:
        features = features + vec_output + manual_features
        

    color_features = vec_output
    valid_features = []
    with tf.control_dependencies(color_features):
        with tf.variable_scope("auxiliary"):
            valid_inds = []
            feature_ind = 0
            if output_type in ['rgb', 'bgr']:
                valid_features = vec_output
                valid_inds = [0, 1, 2]
            else:
                for i in range(len(features)):
                    if isinstance(features[i], (float, int, numpy.bool_)):
                        continue
                    else:
                        if efficient_trace:
                            if features[i] in valid_features:
                                continue
                    feature_ind += 1
                    valid_inds.append(i)
                    valid_features.append(features[i])

            for i in range(len(valid_features)):
                if valid_features[i].dtype != dtype:
                    valid_features[i] = tf.cast(valid_features[i], dtype)
            if write_file:
                numpy.save('%s/valid_inds.npy' % name, valid_inds)
            #valid_features = [features[k] for k in valid_inds]

            if manual_features_only:
                manual_inds = []
                manual_features_valid = []
                additional_bias = []
                additional_scale = []
                for k in range(len(manual_features)):
                    feature = manual_features[k]
                    if not isinstance(feature, (float, int)):
                        try:
                            raw_ind = valid_features.index(feature)
                            manual_inds.append(raw_ind)
                            manual_features_valid.append(feature)
                        except ValueError:
                            if k == len(manual_features) - 1:
                                # t_ray case
                                assert geometry in ['hyperboloid1', 'paraboloid']
                                t_ray_bias = numpy.load(os.path.join(dataroot, 't_ray_bias.npy'))
                                t_ray_scale = numpy.load(os.path.join(dataroot, 't_ray_scale.npy'))
                                additional_bias.append(t_ray_bias[0])
                                additional_scale.append(t_ray_scale[0])
                                manual_features_valid.append(feature)
                            else:
                                raise
                        except:
                            raise
                out_features = manual_features_valid

                if use_dataroot:
                    feature_bias = feature_bias[manual_inds]
                    feature_scale = feature_scale[manual_inds]
                if len(additional_bias):
                    feature_bias = numpy.concatenate((feature_bias, numpy.array(additional_bias)))
                    feature_scale = numpy.concatenate((feature_scale, numpy.array(additional_scale)))
            else:
                out_features = valid_features
                
            

            if ignore_last_n_scale > 0 and output_type not in ['rgb', 'bgr'] and use_dataroot:
                new_inds = numpy.arange(feature_bias.shape[0] - ignore_last_n_scale)
                if include_noise_feature:
                    new_inds = numpy.concatenate((new_inds, [feature_bias.shape[0]-2, feature_bias.shape[0]-1]))
                feature_bias = feature_bias[new_inds]
                feature_scale = feature_scale[new_inds]
                #feature_bias = feature_bias[:-ignore_last_n_scale]
                #feature_scale = feature_scale[:-ignore_last_n_scale]

            if specified_ind is not None:
                specified_ind_vals = specified_ind
                
                new_features = []
                for ind in specified_ind_vals:
                    new_features.append(out_features[ind])
                out_features = new_features
                
                feature_bias = feature_bias[specified_ind_vals]
                feature_scale = feature_scale[specified_ind_vals]
            
            for vec in vec_output:
                #raw_ind = features.index(vec)
                #actual_ind = valid_inds.index(raw_ind)
                actual_ind = out_features.index(vec)
                color_inds.append(actual_ind)
                
            if get_col_aux_inds:
                for vec in manual_features:
                    if vec not in vec_output:
                        if isinstance(vec, tf.Tensor):
                            actual_ind = out_features.index(vec)
                            color_inds.append(actual_ind)
                if alt_dir == '':
                    np.save(os.path.join(name, 'col_aux_inds.npy'), color_inds)
                else:
                    np.save(os.path.join(alt_dir, 'col_aux_inds.npy'), color_inds)
                return

            if output_type not in ['rgb', 'bgr'] and use_dataroot:
                for ind in color_inds:
                    feature_bias[ind] = 0.0
                    feature_scale[ind] = 1.0
            
            if len(feature_pl) > 0:
                for var in feature_pl:
                    if var in out_features:
                        idx = out_features.index(var)

            
            if output_type == 'remove_constant':
                features = tf.parallel_stack(out_features)

                features = tf.transpose(features, [1, 2, 3, 0])


            elif output_type == 'all':
                features = tf.cast(tf.stack(features, axis=-1), tf.float32)
            elif output_type in ['rgb', 'bgr']:
                features = tf.cast(tf.stack(vec_output, axis=-1), tf.float32)
                if output_type == 'bgr':
                    features = features[..., ::-1]
            else:
                raise
            

            if (output_type not in ['rgb', 'bgr']):
                features += feature_bias
                features *= feature_scale

                # sanity check for manual features
                if manual_features_only and False:
                    manual_inds = []
                    manual_features_valid = []
                    for feature in manual_features:
                        if not isinstance(feature, (float, int)):
                            raw_ind = valid_features.index(feature)
                            manual_inds.append(raw_ind)
                            manual_features_valid.append(feature)
                    manual_features_valid = tf.parallel_stack(manual_features_valid)
                    manual_features_valid = tf.transpose(manual_features_valid, [1, 2, 3, 0])
                    manual_features_valid += feature_bias[manual_inds]
                    manual_features_valid *= feature_scale[manual_inds]

                    camera_pos_val = numpy.load(os.path.join(dataroot, 'train.npy'))
                    feed_dict = {camera_pos: camera_pos_val[0], shader_time:[0]}
                    sess = tf.Session()
                    manual_features_val, features_val = sess.run([manual_features_valid, features], feed_dict=feed_dict)
                    for k in range(len(manual_inds)):
                        print(numpy.max(numpy.abs(manual_features_val[:, :, :, k] - features_val[:, :, :, manual_inds[k]])))

            elif output_type in ['rgb', 'bgr']: # workaround because clip_by_value will make all nans to be the higher value
                features = tf.where(tf.is_nan(features), tf.zeros_like(features), features)

            if (not relax_clipping) or (output_type in ['rgb', 'bgr']):
                features_clipped = tf.clip_by_value(features, 0.0, 1.0)
                features = features_clipped
            else:
                features -= 0.5
                features *= 2
                features = tf.clip_by_value(features, -2.0, 2.0)
                #features = tf.clip_by_value(features, -1.0, 2.0)
            #features = tf.minimum(tf.maximum(features, 0.0), 1.0)

            features = tf.where(tf.is_nan(features), tf.zeros_like(features), features)

    
    return features

    #numpy.save('valid_inds.npy', valid_inds)
    #return features

def get_render(camera_pos, shader_time, samples=None, nsamples=1, shader_name='zigzag', color_inds=None, return_vec_output=False, render_size=None, render_sigma=None, compiler_module=None, geometry='plane', zero_samples=False, debug=[], extra_args=[None], render_g=False, manual_features_only=False, fov='regular', h_start=0, h_offset=height, w_start=0, w_offset=width, t_sigma=1/60.0, render_fix_spatial_sample=False, render_zero_spatial_sample=False, spatial_samples=None, additional_features=True, include_noise_feature=False, no_noise_feature=False, same_sample_all_pix=False, texture_maps=[], texture_map_size=[], phase=-1):

    assert compiler_module is not None

    if additional_features:
        if geometry not in ['none', 'texture', 'texture_approximate_10f']:
            features_len_add = 7
        else:
            features_len_add = 2
        if no_noise_feature:
            features_len_add -= 2
    else:
        if include_noise_feature:
            features_len_add = 2
        else:
            features_len_add = 0

    features_len = compiler_module.f_log_intermediate_len + features_len_add

    vec_output_len = compiler_module.vec_output_len

    manual_features_len = compiler_module.f_log_intermediate_subset_len
    manual_depth_offset = 0
    if geometry not in ['none', 'texture', 'texture_approximate_10f']:
        manual_features_len += 1
        manual_depth_offset = 1

    f_log_intermediate_subset = [None] * manual_features_len

        
    if render_size is not None:
        global width
        global height
        width = render_size[0]
        height = render_size[1]
        
    texture_map_size.append(compiler_module.vec_output_len)

    f_log_intermediate = [None] * features_len
    vec_output = [None] * vec_output_len

    xv, yv = tf.meshgrid(tf.range(w_offset, dtype=dtype), tf.range(h_offset, dtype=dtype), indexing='ij')
    xv = tf.transpose(xv)
    yv = tf.transpose(yv)
    xv = tf.expand_dims(xv, 0)
    yv = tf.expand_dims(yv, 0)
    xv = tf.tile(xv, [nsamples, 1, 1])
    yv = tf.tile(yv, [nsamples, 1, 1])
    xv_orig = xv
    yv_orig = yv
    xv += tf.expand_dims(tf.expand_dims(w_start, axis=1), axis=2)
    yv += tf.expand_dims(tf.expand_dims(h_start, axis=1), axis=2)
    tensor_x0 = xv
    tensor_x1 = yv
    tensor_x2 = tf.expand_dims(tf.expand_dims(shader_time, axis=1), axis=2) * tf.cast(tf.fill(tf.shape(xv), 1.0), dtype)

    if samples is None:
        if not same_sample_all_pix:
            sample1 = tf.random_normal(tf.shape(xv), dtype=dtype)
            sample2 = tf.random_normal(tf.shape(xv), dtype=dtype)
            sample3 = tf.random_normal(tf.shape(xv), dtype=dtype)
        else:
            sample1 = tf.fill(tf.shape(xv), tf.random_normal((), dtype=dtype))
            sample2 = tf.fill(tf.shape(xv), tf.random_normal((), dtype=dtype))
            sample3 = tf.fill(tf.shape(xv), tf.random_normal((), dtype=dtype))
    else:
        sample3 = 0.0
        # the assumption here is batch_size = 1
        if isinstance(samples[0], numpy.ndarray) and isinstance(samples[1], numpy.ndarray):
            sample1 = tf.constant(samples[0], dtype=dtype)
            sample2 = tf.constant(samples[1], dtype=dtype)
            if samples[0].shape[1] == height + padding_offset and samples[0].shape[2] == width + padding_offset:
                start_slice = [0, tf.cast(h_start[0], tf.int32) + padding_offset // 2, tf.cast(w_start[0], tf.int32) + padding_offset // 2]
                size_slice = [nsamples, int(h_offset), int(w_offset)]
                sample1 = tf.slice(sample1, start_slice, size_slice)
                sample2 = tf.slice(sample2, start_slice, size_slice)
            else:
                assert samples[0].shape[1] == h_offset and samples[1].shape[2] == w_offset
        else:
            assert isinstance(samples[0], tf.Tensor) and isinstance(samples[1], tf.Tensor)
            sample1 = samples[0]
            sample2 = samples[1]
            start_slice = [0, tf.cast(h_start[0], tf.int32) + padding_offset // 2, tf.cast(w_start[0], tf.int32) + padding_offset // 2]
            size_slice = [nsamples, int(h_offset), int(w_offset)]
            sample1 = tf.slice(sample1, start_slice, size_slice)
            sample2 = tf.slice(sample2, start_slice, size_slice)
        #if dtype == tf.float64:
        #    sample1 = samples[0].astype(np.float64)
        #    sample2 = samples[1].astype(np.float64)

    if render_sigma is None:
        render_sigma = [0.5, 0.5, t_sigma]
    print('render_sigma:', render_sigma)

    if not zero_samples:
        #print("using random samples")

        if (render_fix_spatial_sample or render_zero_spatial_sample) and spatial_samples is not None:
            #print("fix spatial samples")
            sample1 = tf.constant(spatial_samples[0], dtype=dtype)
            sample2 = tf.constant(spatial_samples[1], dtype=dtype)

        vector3 = [tensor_x0 + render_sigma[0] * sample1, tensor_x1 + render_sigma[1] * sample2, tensor_x2]

    else:
        vector3 = [tensor_x0, tensor_x1, tensor_x2]
        sample1 = tf.zeros_like(sample1)
        sample2 = tf.zeros_like(sample2)

    f_log_intermediate[0] = shader_time
    f_log_intermediate[1] = camera_pos

    get_shader(vector3, f_log_intermediate, f_log_intermediate_subset, camera_pos, features_len, manual_features_len, shader_name=shader_name, color_inds=color_inds, vec_output=vec_output, compiler_module=compiler_module, geometry=geometry, debug=debug, extra_args=extra_args, render_g=render_g, manual_features_only=manual_features_only, fov=fov, features_len_add=features_len_add, manual_depth_offset=manual_depth_offset, additional_features=additional_features, texture_maps=texture_maps, phase=phase)


    if (not no_noise_feature):
        if (additional_features or include_noise_feature):
            f_log_intermediate[features_len-2] = sample1
            f_log_intermediate[features_len-1] = sample2

    if return_vec_output:
        return f_log_intermediate, vec_output, f_log_intermediate_subset
    else:
        return f_log_intermediate

def get_shader(x, f_log_intermediate, f_log_intermediate_subset, camera_pos, features_len, manual_features_len, shader_name='zigzag', color_inds=None, vec_output=None, compiler_module=None, geometry='plane', debug=[], extra_args=[None], render_g=False, manual_features_only=False, fov='regular', features_len_add=7, manual_depth_offset=1, additional_features=True, texture_maps=[], phase=-1):
    assert compiler_module is not None
    features_dt = []
    
    input_pl_to_features = []
    
    features = get_features(x, camera_pos, geometry=geometry, debug=debug, extra_args=extra_args, fov=fov, features_dt=features_dt, phase=phase)
    
    if vec_output is None:
        vec_output = [None] * 3

    # adding depth
    if geometry == 'plane':
        f_log_intermediate_subset[-1] = features[7]
    elif geometry in ['hyperboloid1', 'paraboloid']:
        f_log_intermediate_subset[-1] = extra_args[0]
    elif geometry not in ['none', 'texture', 'texture_approximate_10f']:
        raise

    with tf.variable_scope("auxiliary"):

        if geometry not in ['none', 'texture', 'texture_approximate_10f'] and additional_features:
            if not render_g:
                h = 1e-4
            else:
                h = 1e-8
            if geometry == 'plane':
                u_ind = 1
                v_ind = 2
            elif geometry in ['hyperboloid1', 'sphere', 'paraboloid']:
                u_ind = 8
                v_ind = 9
            else:
                raise

            new_x = x[:]
            new_x[0] = x[0] - h
            features_neg_x = get_features(new_x, camera_pos, geometry=geometry, fov=fov)
            new_x[0] = x[0] + h
            features_pos_x = get_features(new_x, camera_pos, geometry=geometry, fov=fov)
            f_log_intermediate[features_len-features_len_add] = (features_pos_x[u_ind] - features_neg_x[u_ind]) / (2 * h)
            f_log_intermediate[features_len-features_len_add+1] = (features_pos_x[v_ind] - features_neg_x[v_ind]) / (2 * h)

            new_x = x[:]
            new_x[1] = x[1] - h
            features_neg_y = get_features(new_x, camera_pos, geometry=geometry, fov=fov)
            new_x[1] = x[1] + h
            features_pos_y = get_features(new_x, camera_pos, geometry=geometry, fov=fov)
            f_log_intermediate[features_len-features_len_add+2] = (features_pos_y[u_ind] - features_neg_y[u_ind]) / (2 * h)
            f_log_intermediate[features_len-features_len_add+3] = (features_pos_y[v_ind] - features_neg_y[v_ind]) / (2 * h)

            f_log_intermediate[features_len-features_len_add+4] = f_log_intermediate[features_len-features_len_add] * f_log_intermediate[features_len-features_len_add+3] - f_log_intermediate[features_len-features_len_add+1] * f_log_intermediate[features_len-features_len_add+2]

            
    if len(debug) > 0:
        vec_output[0] = debug[0]
    if not render_g:
        if texture_maps != []:
            compiler_module.f(features, f_log_intermediate, vec_output, f_log_intermediate_subset, texture_maps=texture_maps)
        else:
            compiler_module.f(features, f_log_intermediate, vec_output, f_log_intermediate_subset)
    else:
        assert geometry not in ['none', 'texture', 'texture_approximate_10f']
        sigma = [None] * len(features)
        for i in range(len(features)):
            variance = (tf.square((features_pos_x[i] - features[i]) / h) * 0.25 + tf.square(numpy.array(features_pos_y[i] - features[i]) / h) * 0.25)
            sigma[i] = tf.sqrt(variance)
            sigma[i] = tf.where(tf.is_nan(sigma[i]), tf.zeros_like(sigma[i]), sigma[i])
        # workaround for bug in shader compiler
        if geometry == 'plane':
            sigma[7] = 0.0
        global dtype
        dtype = tf.float32
        for i in range(len(features)):
            if isinstance(features[i], tf.Tensor):
                features[i] = tf.cast(features[i], dtype)
            if isinstance(sigma[i], tf.Tensor):
                sigma[i] = tf.cast(sigma[i], dtype)

        compiler_module.g(features, vec_output, sigma)

    return

def get_features(x, camera_pos, geometry='plane', debug=[], extra_args=[None], fov='regular', features_dt=[], phase=-1):
    
    if fov.startswith('regular'):
        ray_dir = [x[0] - width / 2, x[1] + 1, width / 2]
    elif fov.startswith('small'):
        ray_dir = [x[0] - width / 2, x[1] - height / 2, 1.73 * width / 2]
        #print("use small fov (60 degrees horizontally)")
    else:
        raise
        
    ray_origin = [camera_pos[0], camera_pos[1], camera_pos[2]]
    ang1 = camera_pos[3]
    ang2 = camera_pos[4]
    ang3 = camera_pos[5]


    for i in range(len(ray_origin)):
        ray_origin[i] = tf.expand_dims(tf.expand_dims(ray_origin[i], axis=1), axis=2)
    ang1 = tf.expand_dims(tf.expand_dims(ang1, axis=1), axis=2)
    ang2 = tf.expand_dims(tf.expand_dims(ang2, axis=1), axis=2)
    ang3 = tf.expand_dims(tf.expand_dims(ang3, axis=1), axis=2)

    ray_dir_norm = tf.sqrt(ray_dir[0] **2 + ray_dir[1] ** 2 + ray_dir[2] ** 2)
    ray_dir[0] /= ray_dir_norm
    ray_dir[1] /= ray_dir_norm
    ray_dir[2] /= ray_dir_norm

    sin1 = tf.sin(ang1)
    cos1 = tf.cos(ang1)
    sin2 = tf.sin(ang2)
    cos2 = tf.cos(ang2)
    sin3 = tf.sin(ang3)
    cos3 = tf.cos(ang3)

    if 'seperable' in fov:
        ray_dir_p = [(sin1 * sin3 + cos1 * sin2 * cos3) * ray_dir[0] + (-cos1 * sin3 + sin1 * sin2 * cos3) * ray_dir[1] + cos2 * cos3 * ray_dir[2],
                     (-sin1 * cos3 + cos1 * sin2 * sin3) * ray_dir[0] + (cos1 * cos3 + sin1 * sin2 * sin3) * ray_dir[1] + cos2 * sin3 * ray_dir[2],
                     cos1 * cos2 * ray_dir[0] + sin1 * cos2 * ray_dir[1] + -sin2 * ray_dir[2]]
    else:
        ray_dir_p = [cos2 * cos3 * ray_dir[0] + (-cos1 * sin3 + sin1 * sin2 * cos3) * ray_dir[1] + (sin1 * sin3 + cos1 * sin2 * cos3) * ray_dir[2],
                     cos2 * sin3 * ray_dir[0] + (cos1 * cos3 + sin1 * sin2 * sin3) * ray_dir[1] + (-sin1 * cos3 + cos1 * sin2 * sin3) * ray_dir[2],
                     -sin2 * ray_dir[0] + sin1 * cos2 * ray_dir[1] + cos1 * cos2 * ray_dir[2]]

    N = [0, 0, 1.0]

    if geometry == 'hyperboloid1':
        features = [None] * 19
        hyperboloid_center = numpy.zeros(3)
        hyperboloid_radius = 30.0
        hyperboloid_radius2 = hyperboloid_radius ** 2.0
        quadric = numpy.array([1.0, 0.0, -1.0,
                               0.0, 0.0, 0.0,
                               -2.0 * hyperboloid_center[0],
                               hyperboloid_radius2,
                               2.0 * hyperboloid_center[2],
                               hyperboloid_center[0] * hyperboloid_center[0] - hyperboloid_center[2] * hyperboloid_center[2] - hyperboloid_center[1] * hyperboloid_radius2])
        t_ray = solve_quadric(quadric, ray_origin, ray_dir_p, features, debug=debug)
        extra_args[0] = t_ray
        feature_scale = 10.0
        features[0] = x[2]
        features[8] = (features[1] - hyperboloid_center[0]) / feature_scale
        features[9] = (features[3] - hyperboloid_center[2]) / feature_scale

        features[13] = 1.0
        features[14] = -2.0 * feature_scale * features[8] / (hyperboloid_radius2)
        #features[14] = 0.0
        features[15] = 0.0
        features[16] = 0.0
        features[17] = 2.0 * feature_scale * features[9] / (hyperboloid_radius2)
        #features[17] = 0.0
        features[18] = 1.0
    elif geometry == 'sphere':
        features = [None] * 19
        sphere_center = numpy.zeros(3)
        sphere_radius = 175.0
        quadric = numpy.array([1.0, 1.0, 1.0,
                               0.0, 0.0, 0.0,
                               -2.0 * sphere_center[0],
                               -2.0 * sphere_center[1],
                               -2.0 * sphere_center[2],
                               sphere_center[0] * sphere_center[0] + sphere_center[1] * sphere_center[1] + sphere_center[2] * sphere_center[2] - sphere_radius * sphere_radius])
        solve_quadric(quadric, ray_origin, ray_dir_p, features)
        feature_scale = 80.0 / numpy.pi
        features[0] = x[2]
        u = tf.atan2(features[10], features[12])
        v = tf.acos(features[11])
        features[8] = u * feature_scale
        features[9] = v * feature_scale

        sin_u = tf.sin(u)
        cos_u = tf.cos(u)
        sin_v = tf.sin(v)
        cos_v = tf.cos(v)

        features[13] = cos_u
        features[14] = 0.0
        features[15] = -sin_u

        features[16] = -sin_u * cos_v
        features[17] = sin_v
        features[18] = -cos_u * cos_v
    elif geometry == 'paraboloid':
        features = [None] * 19
        paraboloid_center = numpy.zeros(3)
        paraboloid_radius = 30.0
        paraboloid_raiuds2 = paraboloid_radius ** 2.0
        paraboloid_xz_scale = 3.0
        quadric = numpy.array([1.0, 0.0, 1.0 / paraboloid_xz_scale ** 2.0,
                               0.0, 0.0, 0.0,
                               -2.0 * paraboloid_center[0],
                               -paraboloid_raiuds2,
                               -2.0 * paraboloid_center[2],
                               paraboloid_center[0] * paraboloid_center[0] + paraboloid_center[2] * paraboloid_center[2] / paraboloid_xz_scale ** 2.0 + paraboloid_center[1] * paraboloid_raiuds2])
        t_ray = solve_quadric(quadric, ray_origin, ray_dir_p, features, debug=debug)
        extra_args[0] = t_ray
        feature_scale = 10.0
        features[0] = x[2]
        features[8] = (features[1] - paraboloid_center[0]) / feature_scale
        features[9] = (features[3] - paraboloid_center[2]) / feature_scale

        features[13] = 1.0
        features[14] = 2.0 * feature_scale * features[8] / (paraboloid_raiuds2)
        #features[14] = 0.0
        features[15] = 0.0
        features[16] = 0.0
        features[17] = 2.0 * feature_scale * features[9] / (paraboloid_raiuds2 * paraboloid_xz_scale ** 2.0)
        #features[17] = 0.0
        features[18] = 1.0
    elif geometry == 'plane':
        features = [None] * 8
        
        t_ray = -ray_origin[2] / (ray_dir_p[2])
        features[0] = x[2]

        features[1] = ray_origin[0] + t_ray * ray_dir_p[0]
        features[2] = ray_origin[1] + t_ray * ray_dir_p[1]
        features[3] = ray_origin[2] + t_ray * ray_dir_p[2]
        features[4] = -ray_dir_p[0]
        features[5] = -ray_dir_p[1]
        features[6] = -ray_dir_p[2]
        features[7] = t_ray
    elif geometry == 'none':
        features = [None] * 7
        features[0] = x[2]
        features[1] = ray_dir_p[0]
        features[2] = ray_dir_p[1]
        features[3] = ray_dir_p[2]
        #features[4] = ray_dir[0]
        #features[5] = ray_dir[1]
        #features[6] = ray_dir[2]
        features[4] = ray_origin[0] * tf.constant(1.0, dtype=dtype, shape=x[0].shape)
        features[5] = ray_origin[1] * tf.constant(1.0, dtype=dtype, shape=x[0].shape)
        features[6] = ray_origin[2] * tf.constant(1.0, dtype=dtype, shape=x[0].shape)
    elif geometry == 'texture':
        features = [None] * 9
        features[0] = x[2]
        features[1] = x[0]
        features[2] = x[1]
        features[3] = camera_pos[0] * tf.constant(1.0, dtype=dtype, shape=x[0].shape)
        features[4] = camera_pos[1] * tf.constant(1.0, dtype=dtype, shape=x[0].shape)
        features[5] = camera_pos[2] * tf.constant(1.0, dtype=dtype, shape=x[0].shape)
        features[6] = camera_pos[3] * tf.constant(1.0, dtype=dtype, shape=x[0].shape)
        features[7] = camera_pos[4] * tf.constant(1.0, dtype=dtype, shape=x[0].shape)
        features[8] = camera_pos[5] * tf.constant(1.0, dtype=dtype, shape=x[0].shape)
    elif geometry == 'texture_approximate_10f':
        features = [None] * 36
        features[0] = x[2]
        features[1] = x[0]
        features[2] = x[1]
        for i in range(33):
            features[i+3] = tf.expand_dims(tf.expand_dims(camera_pos[i], axis=1), axis=2) * tf.constant(1.0, dtype=dtype, shape=x[0].shape)

    return features

def solve_quadric(quadric, ray_origin, ray_dir_p, features, debug=[]):
    Aq = quadric[0] * tf.square(ray_dir_p[0]) + \
         quadric[1] * tf.square(ray_dir_p[1]) + \
         quadric[2] * tf.square(ray_dir_p[2]) + \
         quadric[3] * ray_dir_p[0] * ray_dir_p[1] + \
         quadric[4] * ray_dir_p[0] * ray_dir_p[2] + \
         quadric[5] * ray_dir_p[1] * ray_dir_p[2]

    Bq = 2.0 * (quadric[0] * ray_origin[0] * ray_dir_p[0] + \
                quadric[1] * ray_origin[1] * ray_dir_p[1] + \
                quadric[2] * ray_origin[2] * ray_dir_p[2]) + \
         quadric[3] * (ray_origin[0] * ray_dir_p[1] + ray_origin[1] * ray_dir_p[0]) + \
         quadric[4] * (ray_origin[0] * ray_dir_p[2] + ray_origin[2] * ray_dir_p[0]) + \
         quadric[5] * (ray_origin[1] * ray_dir_p[2] + ray_origin[2] * ray_dir_p[1]) + \
         quadric[6] * ray_dir_p[0] + \
         quadric[7] * ray_dir_p[1] + \
         quadric[8] * ray_dir_p[2];

    Cq = quadric[0] * tf.square(ray_origin[0]) + \
         quadric[1] * tf.square(ray_origin[1]) +\
         quadric[2] * tf.square(ray_origin[2]) + \
         quadric[3] * ray_origin[0] * ray_origin[1] + \
         quadric[4] * ray_origin[0] * ray_origin[2] + \
         quadric[5] * ray_origin[1] * ray_origin[2] + \
         quadric[6] * ray_origin[0] + \
         quadric[7] * ray_origin[1] + \
         quadric[8] * ray_origin[2] + \
         quadric[9]

    root2 = Bq * Bq - 4.0 * Aq * Cq

    cond_reg = tf.abs(Aq) <= 1e-4

    sqrt_root2 = tf.sqrt(root2)
    t0 = (-Bq - sqrt_root2) / (2.0 * Aq)
    t1 = (-Bq + sqrt_root2) / (2.0 * Aq)

    t_ray = tf.where(cond_reg, -Cq / Bq,
                               tf.where(Aq > 0, tf.where(t0 >= 0, t0, t1),
                                                tf.where(t1 >= 0, t1, t0)))
    #t_ray = -Cq / Bq

    root2 = tf.where(t_ray > 0.0, root2, -1.0 * tf.ones_like(root2))

    #intersect_pos = ray_origin + t_ray * ray_dir_p
    intersect_pos = [None] * 3
    for i in range(3):
        intersect_pos[i] = ray_origin[i] + t_ray * ray_dir_p[i]

    normal = [None] * 3

    normal[0] = quadric[6] + 2.0 * quadric[0] * intersect_pos[0] + \
                quadric[3] * intersect_pos[1] + quadric[4] * intersect_pos[2]
    normal[1] = quadric[7] + 2.0 * quadric[1] * intersect_pos[1] + \
                quadric[3] * intersect_pos[0] + quadric[5] * intersect_pos[2]
    normal[2] = quadric[8] + 2.0 * quadric[2] * intersect_pos[2] + \
                quadric[4] * intersect_pos[0] + quadric[5] * intersect_pos[1]
    normal_len = tf.sqrt(normal[0] ** 2.0 + normal[1] ** 2.0 + normal[2] ** 2.0)
    normal_dot_ray_dir_p = normal[0] * ray_dir_p[0] + normal[1] * ray_dir_p[1] + normal[2] * ray_dir_p[2]

    actual_normal = [None] * 3
    for i in range(3):
        normal[i] /= normal_len
        actual_normal[i] = tf.where(normal_dot_ray_dir_p > 0.0, -normal[i], normal[i])
        #actual_normal[i] = normal[i]

    features[1:4] = intersect_pos[:]
    #features[4:7] = -ray_dir_p[:]
    for i in range(3):
        features[4+i] = -ray_dir_p[i]
    features[7] = root2
    features[10:13] = actual_normal[:]
    return t_ray

def lrelu(x):
    return tf.maximum(x*0.2,x)

def identity_initializer(in_channels=[], allow_map_to_less=False, ndims=2):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if not allow_nonzero:
            #print('initializing all zero')
            array = np.zeros(shape, dtype=float)
        else:
            x = np.sqrt(6.0 / (shape[ndims] + shape[ndims+1])) / 1.5
            array = numpy.random.uniform(-x, x, size=shape)
            #print('initializing xavier')
            #return tf.constant(array, dtype=dtype)
        cx = shape[0] // 2
        if ndims > 1:
            cy = shape[1] // 2
        if len(in_channels) > 0:
            input_inds = in_channels
            output_inds = range(len(in_channels))
        elif allow_map_to_less:
            input_inds = range(min(shape[ndims], shape[ndims+1]))
            output_inds = input_inds
        else:
            input_inds = range(shape[ndims])
            output_inds = input_inds
        for i in range(len(input_inds)):
            if ndims == 2:
                if less_aggresive_ini:
                    array[cx, cy, input_inds[i], output_inds[i]] *= 10.0
                else:
                    array[cx, cy, input_inds[i], output_inds[i]] = 1.0
            elif ndims == 1:
                if less_aggresive_ini:
                    array[cx, input_inds[i], output_inds[i]] *= 10.0
                else:
                    array[cx, input_inds[i], output_inds[i]] = 1.0
        return tf.constant(array, dtype=dtype)
    return _initializer

nm = None

conv_channel = 24
actual_conv_channel = conv_channel

dilation_remove_large = False
dilation_clamp_large = False
dilation_remove_layer = False
dilation_threshold = 8

def build_vgg(input, output_nc=3, channels=[]):
    # 2 modifications
    # 1. use avg_pool instead of max_pool (for smooth gradient)
    # 2. use lrelu instead of relu (consistent with other models)
    net = input
    if len(channels) == 0:
        out_channels = [64, 128, 256, 512, 512]
    else:
        out_channels = channels
    nconvs = [2, 2, 3, 3, 3]
    for i in range(5):
        if i > 0:
            net = slim.avg_pool2d(net, 2, scope='pool_%d' % i)
        for j in range(nconvs[i]):
            net = slim.conv2d(net, out_channels[i], [3, 3], activation_fn=lrelu, scope='lrelu_%d_%d' % (i, j), padding=conv_padding)
    
    net = slim.conv2d(net, 1, [1, 1], activation_fn=lrelu, scope='out')
    return net
        
def build(input, ini_id=True, regularizer_scale=0.0, final_layer_channels=-1, identity_initialize=False, output_nc=3):
    regularizer = None
    if not no_L1_reg_other_layers and regularizer_scale > 0.0:
        regularizer = slim.l1_regularizer(regularizer_scale)
    if ini_id or identity_initialize:
        net=slim.conv2d(input,actual_conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(allow_map_to_less=True),scope='g_conv1',weights_regularizer=regularizer, padding=conv_padding)
    else:
        net=slim.conv2d(input,actual_conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,scope='g_conv1',weights_regularizer=regularizer, padding=conv_padding)

    dilation_schedule = [2, 4, 8, 16, 32, 64]
    for ind in range(len(dilation_schedule)):
        dilation_rate = dilation_schedule[ind]
        conv_ind = ind + 2
        if dilation_rate > dilation_threshold:
            if dilation_remove_large:
                dilation_rate = 1
            elif dilation_clamp_large:
                dilation_rate = dilation_threshold
            elif dilation_remove_layer:
                continue
        #print('rate is', dilation_rate)
        net=slim.conv2d(net,actual_conv_channel,[3,3],rate=dilation_rate,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv'+str(conv_ind),weights_regularizer=regularizer, padding=conv_padding)


    net=slim.conv2d(net,actual_conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9',weights_regularizer=regularizer, padding=conv_padding)
    if final_layer_channels > 0:
        if actual_conv_channel > final_layer_channels and (not identity_initialize):
            net = slim.conv2d(net, final_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, scope='final_0', weights_regularizer=regularizer, padding=conv_padding)
            nlayers = [1, 2]
        else:
            nlayers = [0, 1, 2]
        for nlayer in nlayers:
            net = slim.conv2d(net, final_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(allow_map_to_less=True), scope='final_'+str(nlayer),weights_regularizer=regularizer, padding=conv_padding)

    #print('identity last layer?', identity_initialize and identity_output_layer)
    net=slim.conv2d(net,output_nc,[1,1],rate=1,activation_fn=None,scope='g_conv_last',weights_regularizer=regularizer, weights_initializer=identity_initializer(allow_map_to_less=True) if (identity_initialize and identity_output_layer) else tf.contrib.layers.xavier_initializer(), padding=conv_padding)
    return net

def prepare_data_root(dataroot, additional_input=False):
    output_names=[]
    val_img_names=[]
    map_names = []
    val_map_names = []
    grad_names = []
    val_grad_names = []
    add_names = []
    val_add_names = []
    
    validate_img_names = []

    train_output_dir = os.path.join(dataroot, 'train_img')
    test_output_dir = os.path.join(dataroot, 'test_img')
    
    validate_output_dir = os.path.join(dataroot, 'validate_img')

    for file in sorted(os.listdir(train_output_dir)):
        output_names.append(os.path.join(train_output_dir, file))
    for file in sorted(os.listdir(test_output_dir)):
        val_img_names.append(os.path.join(test_output_dir, file))
        
    if os.path.isdir(validate_output_dir):
        for file in sorted(os.listdir(validate_output_dir)):
            validate_img_names.append(os.path.join(validate_output_dir, file))

    if additional_input:
        train_add_dir = os.path.join(dataroot, 'train_add')
        test_add_dir = os.path.join(dataroot, 'test_add')
        for file in sorted(os.listdir(train_add_dir)):
            add_names.append(os.path.join(train_add_dir, file))
        for file in sorted(os.listdir(test_add_dir)):
            val_add_names.append(os.path.join(test_add_dir, file))

    return output_names, val_img_names, map_names, val_map_names, grad_names, val_grad_names, add_names, val_add_names, validate_img_names

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

def generate_parser():
    parser = argparse_util.ArgumentParser(description='FastImageProcessing')
    parser.add_argument('--name', dest='name', default='', help='name of task')
    parser.add_argument('--is_train', dest='is_train', action='store_true', help='state whether this is training or testing')
    parser.add_argument('--use_batch', dest='use_batch', action='store_true', help='whether to use batches in training')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='size of batches')
    parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='number of epochs to train, seperated by comma')
    parser.add_argument('--debug_mode', dest='debug_mode', action='store_true', help='debug mode')
    parser.add_argument('--no_preload', dest='preload', action='store_false', help='whether to preload data')
    parser.add_argument('--test_training', dest='test_training', action='store_true', help='use training data for testing purpose')
    parser.add_argument('--input_w', dest='input_w', type=int, default=960, help='supplemental information needed when using queue to read binary file')
    parser.add_argument('--input_h', dest='input_h', type=int, default=640, help='supplemental information needed when using queue to read binary file')
    parser.add_argument('--which_epoch', dest='which_epoch', type=int, default=0, help='decide which epoch to read the checkpoint')
    parser.add_argument('--generate_timeline', dest='generate_timeline', action='store_true', help='generate timeline files')
    parser.add_argument('--add_initial_layers', dest='add_initial_layers', action='store_true', help='add initial conv layers without dilation')
    parser.add_argument('--initial_layer_channels', dest='initial_layer_channels', type=int, default=-1, help='number of channels in initial layers')
    parser.add_argument('--conv_channel_multiplier', dest='conv_channel_multiplier', type=int, default=1, help='multiplier for conv channel')
    parser.add_argument('--add_final_layers', dest='add_final_layers', action='store_true', help='add final conv layers without dilation')
    parser.add_argument('--final_layer_channels', dest='final_layer_channels', type=int, default=-1, help='number of channels in final layers')
    parser.add_argument('--dilation_remove_large', dest='dilation_remove_large', action='store_true', help='when specified, use ordinary conv layer instead of dilated conv layer with large dilation rate')
    parser.add_argument('--dilation_clamp_large', dest='dilation_clamp_large', action='store_true', help='when specified, clamp large dilation rate to a give threshold')
    parser.add_argument('--dilation_threshold', dest='dilation_threshold', type=int, default=8, help='threshold used to remove or clamp dilation')
    parser.add_argument('--dilation_remove_layer', dest='dilation_remove_layer', action='store_true', help='when specified, use less dilated conv layers')
    parser.add_argument('--conv_channel_no', dest='conv_channel_no', type=int, default=-1, help='directly specify number of channels for dilated conv layers')
    parser.add_argument('--mean_estimator', dest='mean_estimator', action='store_true', help='if true, use mean estimator instead of neural network')
    parser.add_argument('--estimator_samples', dest='estimator_samples', type=int, default=1, help='number of samples used in mean estimator')
    parser.add_argument('--accurate_timing', dest='accurate_timing', action='store_true', help='if true, do not calculate loss for more accurate timing')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.0001, help='learning rate for adam optimizer')
    parser.add_argument('--identity_initialize', dest='identity_initialize', action='store_true', help='if specified, initialize weights such that output is 1 sample RGB')
    parser.add_argument('--nonzero_ini', dest='allow_nonzero', action='store_true', help='if specified, use xavier for all those supposed to be 0 entries in identity_initializer')
    parser.add_argument('--no_identity_output_layer', dest='identity_output_layer', action='store_false', help='if specified, do not use identity mapping for output layer')
    parser.add_argument('--less_aggresive_ini', dest='less_aggresive_ini', action='store_true', help='if specified, use a less aggresive way to initialize RGB weights (multiples of the original xavier weights)')
    parser.add_argument('--render_only', dest='render_only', action='store_true', help='if specified, render using given camera pos, does not calculate loss')
    parser.add_argument('--render_camera_pos', dest='render_camera_pos', default='camera_pos.npy', help='used to render result')
    parser.add_argument('--render_t', dest='render_t', default='render_t.npy', help='used to render output')
    parser.add_argument('--train_res', dest='train_res', action='store_true', help='if specified, out_img = in_noisy_img + out_network')
    parser.add_argument('--RGB_norm', dest='RGB_norm', type=int, default=2, help='specify which p-norm to use for RGB loss')
    parser.add_argument('--mean_estimator_memory_efficient', dest='mean_estimator_memory_efficient', action='store_true', help='if specified, use a memory efficient way to calculate mean estimator, but may not be accurate in time')
    parser.add_argument('--manual_features_only', dest='manual_features_only', action='store_true', help='if specified, use only manual features already specified in each shader program')
    parser.add_argument('--efficient_trace', dest='efficient_trace', action='store_true', help='if specified, use traces that are unique')
    parser.add_argument('--collect_loop_statistic', dest='collect_loop_statistic', action='store_true', help='if specified, use loop statistic only')
    parser.add_argument('--tiled_training', dest='tiled_training', action='store_true', help='if specified, use tiled training')
    parser.add_argument('--tiled_w', dest='tiled_w', type=int, default=240, help='default width for tiles if using tiled training')
    parser.add_argument('--tiled_h', dest='tiled_h', type=int, default=320, help='default height for tiles if using tiled training')
    parser.add_argument('--fov', dest='fov', default='regular', help='specified the camera field of view')
    parser.add_argument('--first_last_only', dest='first_last_only', action='store_true', help='if specified, log only 1st and last iteration, do not log mean and std')
    parser.add_argument('--last_only', dest='last_only', action='store_true', help='log last iteration only')
    parser.add_argument('--subsample_loops', dest='subsample_loops', type=int, default=-1, help='log every n iter')
    parser.add_argument('--last_n', dest='last_n', type=int, default=-1, help='log last n iterations')
    parser.add_argument('--first_n', dest='first_n', type=int, default=-1, help='log first n iterations and last one')
    parser.add_argument('--first_n_no_last', dest='first_n_no_last', type=int, default=-1, help='log first n iterations')
    parser.add_argument('--mean_var_only', dest='mean_var_only', action='store_true', help='if flagged, use only mean and variance as loop statistic')
    parser.add_argument('--render_fix_spatial_sample', dest='render_fix_spatial_sample', action='store_true', help='if specified, fix spatial sample at rendering')
    parser.add_argument('--render_zero_spatial_sample', dest='render_zero_spatial_sample', action='store_true', help='if specified, use zero spatial sample')
    parser.add_argument('--render_fov', dest='render_fov', default='', help='if specified, can overwrite fov at render time')
    parser.add_argument('--every_nth', dest='every_nth', type=int, default=-1, help='log every nth var')
    parser.add_argument('--every_nth_stratified', dest='every_nth_stratified', action='store_true', help='if specified, do stratified sampling for every nth traces')
    parser.add_argument('--aux_plus_manual_features', dest='aux_plus_manual_features', action='store_true', help='if specified, use RGB+aux+manual features')
    parser.add_argument('--no_additional_features', dest='additional_features', action='store_false', help='if specified, do not use additional features during training')
    parser.add_argument('--ignore_last_n_scale', dest='ignore_last_n_scale', type=int, default=0, help='if nonzero, ignore the last n entries of stored feature_bias and feature_scale')
    parser.add_argument('--include_noise_feature', dest='include_noise_feature', action='store_true', help='if specified, include noise as additional features during trianing')
    parser.add_argument('--no_noise_feature', dest='no_noise_feature', action='store_true', help='if specified, do not include noise as additional features during training, will override include_noise_feature')
    parser.add_argument('--perceptual_loss', dest='perceptual_loss', action='store_true',help='if specified, use perceptual loss as well as L2 loss')
    parser.add_argument('--perceptual_loss_term', dest='perceptual_loss_term', default='conv1_1', help='specify to use which layer in vgg16 as perceptual loss')
    parser.add_argument('--perceptual_loss_scale', dest='perceptual_loss_scale', type=float, default=0.0001, help='used to scale perceptual loss')
    parser.add_argument('--relax_clipping', dest='relax_clipping', action='store_true', help='if specified relax the condition of clipping features from 0-1 to -2-2')
    parser.add_argument('--train_with_zero_samples', dest='train_with_zero_samples', action='store_true', help='if specified, only use center of pixel for training')
    parser.add_argument('--tile_only', dest='tile_only', action='store_true', help='if specified, render only tiles (part of an entire image) according to tile_start')
    parser.add_argument('--no_summary', dest='write_summary', action='store_false', help='if specified, do not write train result to summary')
    parser.add_argument('--lpips_loss', dest='lpips_loss', action='store_true', help='if specified, use perceptual loss from Richard Zhang paepr')
    parser.add_argument('--lpips_loss_scale', dest='lpips_loss_scale', type=float, default=1.0, help='specifies the scale of lpips loss')
    parser.add_argument('--no_l2_loss', dest='l2_loss', action='store_false', help='if specified, do not use l2 loss')
    parser.add_argument('--lpips_net', dest='lpips_net', default='alex', help='specifies which network to use for lpips loss')
    parser.add_argument('--render_sigma', dest='render_sigma', type=float, default=0.5, help='specifies the sigma used for rendering')
    parser.add_argument('--same_sample_all_pix', dest='same_sample_all_pix', action='store_true', help='if specified, generate scalar random noise for all pixels instead of a matrix')
    parser.add_argument('--analyze_channel', dest='analyze_channel', action='store_true', help='in this mode, analyze and visualize contribution of each channel')
    parser.add_argument('--bad_example_base_dir', dest='bad_example_base_dir', default='', help='base dir for bad examples(RGB+Aux) in analyze_channel mode')
    parser.add_argument('--analyze_current_only', dest='analyze_current_only', action='store_true', help='in this mode, analyze only g_current (mostly because at lower res to resolve OOM)')
    parser.add_argument('--texture_maps', dest='texture_maps', default='', help='if not empty, retrieve texture map from the file')
    parser.add_argument('--additional_input', dest='additional_input', action='store_true', help='if true, find additional input features from train/test_add')
    parser.add_argument('--patch_gan_loss', dest='patch_gan_loss', action='store_true', help='if specified, use a patch gan loss together with existing loss')
    parser.add_argument('--no_spatial_GAN', dest='spatial_GAN', action='store_false', help='if sepcified, do not include spatial GAN. This option is only valid when patch_gan_loss is true, and train_temporal_seq is also true.')
    parser.add_argument('--ndf', dest='ndf', type=int, default=32, help='number of discriminator filters on first layer if using patch GAN loss')
    parser.add_argument('--ndf_temporal', dest='ndf_temporal', type=int, default=32, help='number of temporal discriminator filters on first layer')
    parser.add_argument('--gan_loss_scale', dest='gan_loss_scale', type=float, default=1.0, help='the scale multiplied to GAN loss before adding to regular loss')
    parser.add_argument('--discrim_nlayers', dest='discrim_nlayers', type=int, default=2, help='number of layers of discriminator')
    parser.add_argument('--save_frequency', dest='save_frequency', type=int, default=100, help='specifies the frequency to save a checkpoint')
    parser.add_argument('--discrim_train_steps', dest='discrim_train_steps', type=int, default=1, help='specified how often to update discrim')
    parser.add_argument('--gan_loss_style', dest='gan_loss_style', default='cross_entropy', help='specifies what GAN loss to use')
    parser.add_argument('--no_dataroot', dest='use_dataroot', action='store_false', help='if specified, do not need a dataroot (used for check runtime)')
    parser.add_argument('--camera_pos_file', dest='camera_pos_file', default='', help='if specified, use for no_dataroot mode')
    parser.add_argument('--feature_size_only', dest='feature_size_only', action='store_true', help='if specified, do not further create neural network, return after collecting the feature size')
    parser.add_argument('--automatic_subsample', dest='automatic_subsample', action='store_true', help='if specified, automatically decide program subsample rate (and raymarching and function def)')
    parser.add_argument('--automate_raymarching_def', dest='automate_raymarching_def', action='store_true', help='if specified, automatically choose schedule for raymarching and function def (but not subsampling rate')
    parser.add_argument('--train_temporal_seq', dest='train_temporal_seq', action='store_true', help='if set, training on temporal sequences instead of on single frames')
    parser.add_argument('--temporal_seq_length', dest='temporal_seq_length', type=int, default=6, help='length of generated temporal seq during training')
    parser.add_argument('--inference_seq_len', dest='inference_seq_len', type=int, default=8, help='sequence length for inference')
    parser.add_argument('--nframes_temporal_gen', dest='nframes_temporal_gen', type=int, default=3, help='number of frames generator considers in temporal seq mode')
    parser.add_argument('--nframes_temporal_discrim', dest='nframes_temporal_discrim', type=int, default=3, help='number of frames temporal discrim considers in temporal seq mode')
    parser.add_argument('--SELECT_FEATURE_THRE', dest='SELECT_FEATURE_THRE', type=int, default=200, help='when automatically decide subsample rate, this will decide the trace budget')
    parser.add_argument('--temporal_discrim_only', dest='temporal_discrim_only', action='store_true', help='if set, do not use single frame discriminator')
    parser.add_argument('--repeat_timing', dest='repeat_timing', type=int, default=1, help='if > 1, repeat inference multiple times to get stable timing')
    parser.add_argument('--compiler_problem_idx', dest='compiler_problem_idx', type=int, default=-1, help='if nonnegative, use this idx to find appropriate compiler problem')
    parser.add_argument('--render_no_video', dest='render_no_video', action='store_true', help='in this mode, render images only, do not generate video')
    parser.add_argument('--render_dirname', dest='render_dirname', default='render', help='directory used to store render result')
    parser.add_argument('--render_tile_start', dest='render_tile_start', default='', help='specifies the tile start for each rendering if render in test_training mode')
    parser.add_argument('--feature_reduction_ch', dest='feature_reduction_ch', type=int, default=-1, help='specifies dimensionality after feature reduction channel. By default it should be the same as following initial layer or dilation layers, but we might want to change the dimensionality larger for fair RGBx comparison')
    parser.add_argument('--collect_validate_loss', dest='collect_validate_loss', action='store_true', help='if true, collect validation loss (and training score) and write to tensorboard')
    parser.add_argument('--read_from_best_validation', dest='read_from_best_validation', action='store_true', help='if true, read from the best validation checkpoint')
    parser.add_argument('--feature_normalize_lo_pct', dest='feature_normalize_lo_pct', type=int, default=25, help='used to find feature_bias file')
    parser.add_argument('--get_col_aux_inds', dest='get_col_aux_inds', action='store_true', help='if true, write the inds for color and aux channels and do nothing else')
    parser.add_argument('--specified_ind', dest='specified_ind', default='', help='if specified, using the specified ind to define a subset of the trace for learning')
    parser.add_argument('--test_output_dir', dest='test_output_dir', default='', help='if specified, write output to this directory instead')
    parser.add_argument('--no_overwrite_option_file', dest='overwrite_option_file', action='store_false', help='if specified, do not overwrite option file even if the old one is outdated')
    parser.add_argument('--dataroot_parent', dest='dataroot_parent', default='', help='specifies the parent directory for all dataroot dirs')
    parser.add_argument('--epoch_per_shader', dest='epoch_per_shader', type=int, default=1, help='number of epochs run per shader')
    parser.add_argument('--multiple_feature_reduction_ch', dest='multiple_feature_reduction_ch', default='', help='specifies different feature reduction ch for different shader')
    parser.add_argument('--choose_shaders', dest='choose_shaders', type=int, default=0, help='specifies which set of shaders to use')
    
    parser.set_defaults(is_train=False)
    parser.set_defaults(use_batch=False)
    parser.set_defaults(debug_mode=False)
    parser.set_defaults(preload=True)
    parser.set_defaults(test_training=False)
    parser.set_defaults(generate_timeline=False)
    parser.set_defaults(add_initial_layers=False)
    parser.set_defaults(add_final_layers=False)
    parser.set_defaults(dilation_remove_large=False)
    parser.set_defaults(dilation_clamp_large=False)
    parser.set_defaults(dilation_remove_layer=False)
    parser.set_defaults(mean_estimator=False)
    parser.set_defaults(accurate_timing=False)
    parser.set_defaults(identity_initialize=False)
    parser.set_defaults(allow_nonzero=False)
    parser.set_defaults(identity_output_layer=True)
    parser.set_defaults(less_aggresive_ini=False)
    parser.set_defaults(train_res=False)
    parser.set_defaults(mean_estimator_memory_efficient=False)
    parser.set_defaults(efficient_trace=False)
    parser.set_defaults(collect_loop_statistic=False)
    parser.set_defaults(tiled_training=False)
    parser.set_defaults(first_last_only=False)
    parser.set_defaults(last_only=False)
    parser.set_defaults(render_fix_spatial_sample=False)
    parser.set_defaults(render_zero_spatial_sample=False)
    parser.set_defaults(mean_var_only=False)
    parser.set_defaults(every_nth_stratified=False)
    parser.set_defaults(aux_plus_manual_features=False)
    parser.set_defaults(additional_features=True)
    parser.set_defaults(include_noise_feature=False)
    parser.set_defaults(no_noise_feature=False)
    parser.set_defaults(perceptual_loss=False)
    parser.set_defaults(relax_clipping=False)
    parser.set_defaults(train_with_zero_samples=False)
    parser.set_defaults(tile_only=False)
    parser.set_defaults(write_summary=True)
    parser.set_defaults(lpips_loss=False)
    parser.set_defaults(l2_loss=True)
    parser.set_defaults(same_sample_all_pix=False)
    parser.set_defaults(analyze_channel=False)
    parser.set_defaults(analyze_current_only=False)
    parser.set_defaults(additional_input=False)
    parser.set_defaults(patch_gan_loss=False)
    parser.set_defaults(use_dataroot=True)
    parser.set_defaults(feature_size_only=False)
    parser.set_defaults(automatic_subsample=False)
    parser.set_defaults(automate_raymarching_def=False)
    parser.set_defaults(log_only_return_def_raymarching=True)
    parser.set_defaults(train_temporal_seq=False)
    parser.set_defaults(temporal_discrim_only=False)
    parser.set_defaults(spatial_GAN=True)
    parser.set_defaults(render_no_video=False)
    parser.set_defaults(collect_validate_loss=False)
    parser.set_defaults(read_from_best_validation=False)
    parser.set_defaults(get_col_aux_inds=False)
    parser.set_defaults(overwrite_option_file=True)
    
    return parser

def main():
    parser = generate_parser()
    
    args = parser.parse_args()

    main_network(args)

def copy_option(args):
    new_args = copy.copy(args)
    delattr(new_args, 'is_train')
    delattr(new_args, 'test_training')
    delattr(new_args, 'which_epoch')
    delattr(new_args, 'generate_timeline')
    delattr(new_args, 'debug_mode')
    delattr(new_args, 'mean_estimator')
    delattr(new_args, 'estimator_samples')
    delattr(new_args, 'preload')
    delattr(new_args, 'accurate_timing')
    delattr(new_args, 'render_only')
    delattr(new_args, 'render_camera_pos')
    delattr(new_args, 'render_t')
    delattr(new_args, 'mean_estimator_memory_efficient')
    delattr(new_args, 'render_fix_spatial_sample')
    delattr(new_args, 'render_zero_spatial_sample')
    delattr(new_args, 'render_fov')
    delattr(new_args, 'write_summary')
    delattr(new_args, 'analyze_channel')
    delattr(new_args, 'bad_example_base_dir')
    delattr(new_args, 'analyze_current_only')
    delattr(new_args, 'inference_seq_len')
    delattr(new_args, 'repeat_timing')
    delattr(new_args, 'compiler_problem_idx')
    delattr(new_args, 'render_no_video')
    delattr(new_args, 'render_dirname')
    delattr(new_args, 'render_tile_start')
    delattr(new_args, 'collect_validate_loss')
    delattr(new_args, 'read_from_best_validation')
    delattr(new_args, 'get_col_aux_inds')
    delattr(new_args, 'specified_ind')
    delattr(new_args, 'test_output_dir')
    delattr(new_args, 'name')
    delattr(new_args, 'overwrite_option_file')
    return new_args

def main_network(args):


    # batch_size can work with 2, but OOM with 4

    if args.name == '':
        args.name = ''.join(random.choice(string.digits) for _ in range(5))

    if not os.path.isdir(args.name):
        os.makedirs(args.name)

    option_file = os.path.join(args.name, 'option.txt')
    option_copy = copy_option(args)
    if os.path.exists(option_file):
        option_str = open(option_file).read()
        print(str(option_copy))
        try:
            assert option_str == str(option_copy)
        except:
            updated_keys = []
            for key in option_copy.__dict__.keys():
                if key not in option_str:
                    updated_keys.append(key)
            for key in updated_keys:
                delattr(option_copy, key)
            assert option_str == str(option_copy)
            option_copy = copy_option(args)
            if args.overwrite_option_file:
                open(option_file, 'w').write(str(option_copy))
    else:
        if args.overwrite_option_file:
            open(option_file, 'w').write(str(option_copy))

    # for simplicity, only allow batch_size=1 for inference
    # TODO: can come back to relax this contrain.
    inference_entire_img_valid = False
    if not args.is_train:
        args.use_batch = False
        args.batch_size = 1
        # at test time, always inference an entire image, rahter than tiles
        if not args.test_training:
            inference_entire_img_valid = True

    global actual_conv_channel
    actual_conv_channel *= args.conv_channel_multiplier
    if actual_conv_channel == 0:
        actual_conv_channel = args.conv_channel_no
    if args.initial_layer_channels < 0:
        args.initial_layer_channels = actual_conv_channel
    if args.final_layer_channels < 0:
        args.final_layer_channels = actual_conv_channel
        
    global dilation_threshold
    dilation_threshold = args.dilation_threshold
    global padding_offset
    padding_offset = 4 * args.dilation_threshold

    assert (not args.dilation_clamp_large) or (not args.dilation_remove_large) or (not args.dilation_remove_layer)
    global dilation_clamp_large
    dilation_clamp_large = args.dilation_clamp_large
    global dilation_remove_large
    dilation_remove_large = args.dilation_remove_large
    global dilation_remove_layer
    dilation_remove_layer = args.dilation_remove_layer

    if not args.add_final_layers:
        args.final_layer_channels = -1

    global width
    width = args.input_w
    global height
    height = args.input_h

    global allow_nonzero
    allow_nonzero = args.allow_nonzero

    global identity_output_layer
    identity_output_layer = args.identity_output_layer

    global less_aggresive_ini
    less_aggresive_ini = args.less_aggresive_ini

    if args.render_only:
        args.is_train = False
        if args.render_fov != '':
            args.fov = args.render_fov

    if args.mean_estimator_memory_efficient:
        assert not args.generate_timeline

    if args.tiled_training or args.tile_only:
        global conv_padding
        conv_padding = "VALID"
        
    if args.tiled_training:
        assert width % args.tiled_w == 0
        assert height % args.tiled_h == 0
        ntiles_w = width / args.tiled_w
        ntiles_h = height / args.tiled_h
    else:
        ntiles_w = 1
        ntiles_h = 1

    render_sigma = [args.render_sigma, args.render_sigma, 0]
    
    if (args.tiled_training or args.tile_only) and (not inference_entire_img_valid):
        output_pl_w = args.tiled_w
        output_pl_h = args.tiled_h
    else:
        output_pl_w = args.input_w
        output_pl_h = args.input_h
           
    
            
    
    
    
    
    avg_loss = 0
    tf.summary.scalar('avg_loss', avg_loss)

    avg_loss_l2 = 0
    tf.summary.scalar('avg_loss_l2', avg_loss_l2)


    avg_training_loss = 0
    tf.summary.scalar('avg_training_loss', avg_training_loss)

    avg_test_close = 0
    tf.summary.scalar('avg_test_close', avg_test_close)
    avg_test_far = 0
    tf.summary.scalar('avg_test_far', avg_test_far)
    avg_test_middle = 0
    tf.summary.scalar('avg_test_middle', avg_test_middle)
    avg_test_all = 0
    tf.summary.scalar('avg_test_all', avg_test_all)
    reg_loss = 0
    tf.summary.scalar('reg_loss', reg_loss)
    l2_loss = 0
    tf.summary.scalar('l2_loss', l2_loss)
    perceptual_loss = 0
    tf.summary.scalar('perceptual_loss', perceptual_loss)
    
    validate = 0
    tf.summary.scalar('validate', validate)

    
    merged = tf.summary.merge_all()
    
    orig_args = copy.copy(args)
    
    if args.is_train:
        global_epoch = args.epoch
    else:
        global_epoch = args.which_epoch + 1
        
    if args.manual_features_only:
        global all_shaders, all_shaders_aux
        all_shaders = all_shaders_aux
        
    if args.choose_shaders > 0:
        global all_shaders, shaders_pool
        all_shaders = shaders_pool[args.choose_shaders]
        
    all_train_writers = [None] * len(all_shaders)
    
    if args.multiple_feature_reduction_ch != '':
        multiple_feature_reduction_ch = [int(val) for val in args.multiple_feature_reduction_ch.split(',')]
        assert len(multiple_feature_reduction_ch) == len(all_shaders)
    else:
        multiple_feature_reduction_ch = None
    
    T0 = time.time()
    
    for global_e in range(args.which_epoch + 1, global_epoch + 1):
        
        print(global_e)
    
        for shader_ind in range(len(all_shaders)):

            tf.reset_default_graph()

            sess = tf.Session()

            shader_name, geometry, dataroot, extra_args = all_shaders[shader_ind]

            print('running shader %s' % shader_name)

            args = copy.copy(orig_args)

            args.shader_name = shader_name
            args.geometry = geometry
            args.dataroot = os.path.join(orig_args.dataroot_parent, dataroot)

            for key in extra_args.keys():
                if key == 'specified_ind':
                    setattr(args, key, os.path.join(orig_args.dataroot_parent, dataroot, extra_args[key]))
                else:
                    setattr(args, key, extra_args[key])
                
            if multiple_feature_reduction_ch is not None:
                args.feature_reduction_ch = multiple_feature_reduction_ch[shader_ind]

            output_names, val_img_names, map_names, val_map_names, grad_names, val_grad_names, add_names, val_add_names, validate_img_names = prepare_data_root(args.dataroot, additional_input=args.additional_input)
            if args.test_training:
                val_img_names = output_names
                val_map_names = map_names
                val_grad_names = grad_names
                val_add_names = add_names

            camera_pos = tf.placeholder(dtype, shape=[6, args.batch_size])
            shader_time = tf.placeholder(dtype, shape=args.batch_size)
            output_pl = tf.placeholder(tf.float32, shape=[None, output_pl_h, output_pl_w, 3])

            texture_maps = []
            if args.texture_maps != '':
                combined_texture_maps = np.load(args.texture_maps)
                for i in range(combined_texture_maps.shape[0]):
                    texture_maps.append(tf.convert_to_tensor(combined_texture_maps[i], dtype=dtype))

            if args.is_train or args.test_training:
                camera_pos_vals = np.load(os.path.join(args.dataroot, 'train.npy'))
                time_vals = np.load(os.path.join(args.dataroot, 'train_time.npy'))
                if args.tile_only:
                    tile_start_vals = np.load(os.path.join(args.dataroot, 'train_start.npy'))
            else:
                camera_pos_vals = np.concatenate((
                                    np.load(os.path.join(args.dataroot, 'test_close.npy')),
                                    np.load(os.path.join(args.dataroot, 'test_far.npy')),
                                    np.load(os.path.join(args.dataroot, 'test_middle.npy'))
                                    ), axis=0)

                time_vals = np.load(os.path.join(args.dataroot, 'test_time.npy'))

            nexamples = time_vals.shape[0]
            
            if args.specified_ind != '':
                my_specified_ind_file = os.path.join(args.name, '%s_specified_ind.npy' % args.shader_name)
                specified_ind = np.load(args.specified_ind)
                if not os.path.exists(my_specified_ind_file):
                    shutil.copyfile(args.specified_ind, my_specified_ind_file)
                else:
                    my_ind = np.load(my_specified_ind_file)
                    assert np.allclose(my_ind, specified_ind)
            else:
                specified_ind = None
            
            def feature_reduction_layer(input_to_network, _replace_normalize_weights=None, shadername=''):
                with tf.variable_scope("feature_reduction" + shadername, reuse=tf.AUTO_REUSE):

                    actual_nfeatures = args.input_nc

                    if args.feature_reduction_ch > 0:
                        actual_feature_reduction_ch = args.feature_reduction_ch
                    else:
                        actual_feature_reduction_ch = args.initial_layer_channels

                    w_shape = [1, 1, actual_nfeatures, actual_feature_reduction_ch]
                    conv = tf.nn.conv2d
                    strides = [1, 1, 1, 1]

                    weights = tf.get_variable('w0', w_shape, initializer=tf.contrib.layers.xavier_initializer() if not args.identity_initialize else identity_initializer(color_inds, ndims=2))

                    weights_to_input = weights

                    reduced_feat = conv(input_to_network, weights_to_input, strides, "SAME")

                    if args.initial_layer_channels <= actual_conv_channel:
                        ini_id = True
                    else:
                        ini_id = False

                    if args.add_initial_layers:
                        for nlayer in range(3):
                            reduced_feat = slim.conv2d(reduced_feat, actual_initial_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(allow_map_to_less=True), scope='initial_'+str(nlayer), padding=conv_padding)          

                return reduced_feat

            with tf.variable_scope("shader"):
                output_type = 'remove_constant'

                if args.mean_estimator and not args.mean_estimator_memory_efficient:
                    shader_samples = args.estimator_samples
                elif args.train_temporal_seq:
                    if args.is_train:
                        shader_samples = ngts
                    else:
                        shader_samples = args.batch_size
                else:
                    shader_samples = args.batch_size

                color_inds = []
                if args.tiled_training or args.tile_only:

                    h_start = tf.placeholder(dtype=dtype, shape=args.batch_size)
                    w_start = tf.placeholder(dtype=dtype, shape=args.batch_size)

                    if not inference_entire_img_valid:
                        h_offset = args.tiled_h + padding_offset
                        w_offset = args.tiled_w + padding_offset
                    else:
                        h_offset = args.input_h + padding_offset
                        w_offset = args.input_w + padding_offset

                    if args.is_train or args.tile_only:
                        feed_samples = None
                    else:
                        # for inference, need to ensure that noise samples used within an image is the same
                        if not args.mean_estimator:
                            feed_samples = [tf.placeholder(dtype=dtype, shape=[args.batch_size, height+padding_offset, width+padding_offset]), tf.placeholder(dtype=dtype, shape=[args.batch_size, height+padding_offset, width+padding_offset])]
                        else:
                            feed_samples = [tf.placeholder(dtype=dtype, shape=[args.estimator_samples, height+padding_offset, width+padding_offset]), tf.placeholder(dtype=dtype, shape=[args.estimator_samples, height+padding_offset, width+padding_offset])]


                else:
                    h_start = tf.constant([0.0])
                    w_start = tf.constant([0.0])
                    h_offset = height
                    w_offset = width
                    feed_samples = None
                zero_samples = False


                spatial_samples = None
                if args.render_fix_spatial_sample:
                    spatial_samples = [numpy.random.normal(size=(1, h_offset, w_offset)), numpy.random.normal(size=(1, h_offset, w_offset))]
                elif args.render_zero_spatial_sample:
                    spatial_samples = [numpy.zeros((1, h_offset, w_offset)), numpy.zeros((1, h_offset, w_offset))]

                if args.train_with_zero_samples:
                    zero_samples = True


                debug = []

                def generate_input_to_network_wrapper():
                    def func(texture_maps_input):

                        return get_tensors(args.dataroot, args.name, camera_pos, shader_time, output_type, shader_samples, shader_name=args.shader_name, geometry=args.geometry, color_inds=color_inds, manual_features_only=args.manual_features_only, efficient_trace=args.efficient_trace, collect_loop_statistic=args.collect_loop_statistic, h_start=h_start, h_offset=h_offset, w_start=w_start, w_offset=w_offset, samples=feed_samples, fov=args.fov, first_last_only=args.first_last_only, last_only=args.last_only, subsample_loops=args.subsample_loops, last_n=args.last_n, first_n=args.first_n, first_n_no_last=args.first_n_no_last, mean_var_only=args.mean_var_only, zero_samples=zero_samples, render_fix_spatial_sample=args.render_fix_spatial_sample, render_zero_spatial_sample=args.render_zero_spatial_sample, spatial_samples=spatial_samples, every_nth=args.every_nth, every_nth_stratified=args.every_nth_stratified, additional_features=args.additional_features, ignore_last_n_scale=args.ignore_last_n_scale, include_noise_feature=args.include_noise_feature, no_noise_feature=args.no_noise_feature, relax_clipping=args.relax_clipping, render_sigma=render_sigma, same_sample_all_pix=args.same_sample_all_pix, texture_maps=texture_maps_input, use_dataroot=args.use_dataroot, automatic_subsample=args.automatic_subsample, automate_raymarching_def=args.automate_raymarching_def, log_only_return_def_raymarching=args.log_only_return_def_raymarching, SELECT_FEATURE_THRE=args.SELECT_FEATURE_THRE, debug=debug, compiler_problem_idx=shader_ind, feature_normalize_lo_pct=args.feature_normalize_lo_pct, get_col_aux_inds=args.get_col_aux_inds, specified_ind=specified_ind, write_file=args.overwrite_option_file, alt_dir=args.test_output_dir)

                    return func

                generate_input_to_network = generate_input_to_network_wrapper()
                input_to_network = generate_input_to_network(texture_maps)

                if args.get_col_aux_inds:
                    return

                if args.feature_size_only:
                    print('feature size: ', int(input_to_network.shape[-1]))
                    return

                output = output_pl

                if (args.tiled_training or args.tile_only) and args.mean_estimator:
                    input_to_network = tf.slice(input_to_network, [0, padding_offset // 2, padding_offset // 2, 0], [args.estimator_samples, output_pl_h, output_pl_w, 3])
                elif args.additional_input:
                    additional_input = tf.pad(additional_input_pl, [[0, 0], [padding_offset // 2, padding_offset // 2], [padding_offset // 2, padding_offset // 2], [0, 0]], "SYMMETRIC")
                    print('concatenating additional input')
                    input_to_network = tf.concat((input_to_network, additional_input), axis=3)


                if len(color_inds) == 3:
                    color_inds = color_inds[::-1]

                if input_to_network is not None:
                    args.input_nc = int(input_to_network.shape[-1])
                    debug_input = input_to_network


            with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
                if args.debug_mode and args.mean_estimator:
                    with tf.variable_scope("shader"):
                        network = tf.reduce_mean(input_to_network, axis=0, keep_dims=True)
                    sparsity_loss = 0
                else:
                    if args.input_nc <= actual_conv_channel:
                        ini_id = True
                    else:
                        ini_id = False
                    alpha = tf.placeholder(tf.float32)
                    alpha_val = 1.0

                    replace_normalize_weights = None
                    normalize_weights = None

                    sparsity_loss = tf.constant(0.0, dtype=dtype)


                    if args.analyze_channel:
                        sparsity_vec = tf.ones(args.input_nc, dtype=dtype)
                        input_to_network = input_to_network * sparsity_vec

                    actual_initial_layer_channels = args.initial_layer_channels

                    feature_reduction_tensor = None
                    if not args.train_temporal_seq:


                        input_to_network = feature_reduction_layer(input_to_network, _replace_normalize_weights=replace_normalize_weights, shadername=args.shader_name)
                        feature_reduction_tensor = input_to_network



                        reduced_dim_feature = input_to_network

                        network=build(input_to_network, ini_id, final_layer_channels=args.final_layer_channels, identity_initialize=args.identity_initialize, output_nc=3)

                    else:
                        # 2D case
                        assert args.add_initial_layers
                        if args.is_train:
                            input_labels = []
                            generated_frames = []
                            current_input_ls = []
                            generated_seq = []
                            # all previous frames have input label and generated frames
                            # do not include current input label because it's part of the trace
                            ninputs = 2 * args.nframes_temporal_gen - 2
                            for i in range(args.nframes_temporal_gen-1):
                                input_labels.append(tf.stack([input_to_network[i:i+1, :, :, col] for col in color_inds], 3))
                                generated_frames.append(tf.pad(output[:, :, :, 3*i:3*i+3], [[0, 0], [padding_offset // 2, padding_offset // 2], [padding_offset // 2, padding_offset // 2], [0, 0]]))
                                current_input_ls.append(input_labels[-1])
                                current_input_ls.append(generated_frames[-1])

                            for i in range(args.nframes_temporal_gen-1, ngts):
                                input_labels.append(tf.stack([input_to_network[i:i+1, :, :, col] for col in color_inds], 3))
                                current_input_prev = tf.concat(current_input_ls, 3)
                                with tf.variable_scope("single_frame_generator", reuse=tf.AUTO_REUSE):
                                    current_input = feature_reduction_layer(input_to_network[i:i+1, :, :, :])
                                    current_input = tf.concat((current_input, tf.stop_gradient(current_input_prev)), 3)
                                    for nlayer in range(3):
                                        current_input = slim.conv2d(current_input, args.initial_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=tf.contrib.layers.xavier_initializer(), scope='initial_'+str(nlayer), padding=conv_padding)
                                    current_output = build(current_input, ini_id, final_layer_channels=args.final_layer_channels, identity_initialize=args.identity_initialize, output_nc=3)
                                    generated_seq.append(current_output)
                                generated_frames.append(tf.pad(current_output, [[0, 0], [padding_offset // 2, padding_offset // 2], [padding_offset // 2, padding_offset // 2], [0, 0]]))
                                current_input_ls.append(input_labels[-1])
                                current_input_ls.append(generated_frames[-1])
                                current_input_ls = current_input_ls[2:]
                        else:
                            # at inference, do not build the giant graph
                            # input previous frames with placeholder and feed_dict
                            # so we can generate the seq as long as we wish
                            input_label = tf.stack([input_to_network[:, :, :, col] for col in color_inds], 3)
                            with tf.variable_scope("single_frame_generator", reuse=tf.AUTO_REUSE):
                                current_input = feature_reduction_layer(input_to_network)
                                current_input = tf.concat((current_input, previous_input), 3)
                                for nlayer in range(3):
                                    current_input = slim.conv2d(current_input, args.initial_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=tf.contrib.layers.xavier_initializer(), scope='initial_'+str(nlayer), padding=conv_padding)
                                current_output = build(current_input, ini_id, final_layer_channels=args.final_layer_channels, identity_initialize=args.identity_initialize, output_nc=3)
                                network = current_output



            weight_map = tf.placeholder(tf.float32,shape=[None,None,None])

            if args.l2_loss:
                if (not args.is_train) or (not args.train_temporal_seq):
                    if (not args.train_res) or (args.debug_mode and args.mean_estimator):
                        diff = network - output
                    else:
                        input_color = tf.stack([debug_input[..., ind] for ind in color_inds], axis=-1)
                        diff = network + input_color - output
                        network += input_color

                    if args.RGB_norm % 2 != 0:
                        diff = tf.abs(diff)
                    powered_diff = diff ** args.RGB_norm

                    loss_per_sample = tf.reduce_mean(powered_diff, (1, 2, 3))
                    loss = tf.reduce_mean(loss_per_sample)

                else:
                    assert args.RGB_norm == 2
                    assert not args.train_res
                    loss = tf.reduce_mean((tf.concat(generated_seq, 3) - output[:, :, :, 3*(args.nframes_temporal_gen-1):]) ** 2)
            else:
                loss = tf.constant(0.0, dtype=dtype)

            loss_l2 = loss
            loss_add_term = loss

            if args.perceptual_loss:
                sys.path += ['../../tensorflow-vgg']
                import vgg16
                vgg_in = vgg16.Vgg16()
                vgg_in.build(network)
                vgg_out = vgg16.Vgg16()
                vgg_out.build(output)
                loss_vgg = tf.reduce_mean(tf.square(getattr(vgg_in, args.perceptual_loss_term) - getattr(vgg_out, args.perceptual_loss_term)))
                perceptual_loss_add = args.perceptual_loss_scale * loss_vgg
                loss += perceptual_loss_add
            elif args.lpips_loss:
                sys.path += ['../../lpips-tensorflow']
                import lpips_tf
                if args.train_temporal_seq and args.is_train:
                    loss_lpips = 0.0
                    for i in range(len(generated_seq)):
                        start_ind = args.nframes_temporal_gen - 1 + i
                        loss_lpips += lpips_tf.lpips(generated_seq[i], output[:, :, :, 3*start_ind:3*start_ind+3], model='net-lin', net=args.lpips_net)
                    loss_lpips /= len(generated_seq)
                else:
                    loss_lpips = lpips_tf.lpips(network, output, model='net-lin', net=args.lpips_net)

                perceptual_loss_add = args.lpips_loss_scale * loss_lpips
                if args.batch_size > 1:
                    perceptual_loss_add = tf.reduce_mean(perceptual_loss_add)
                loss += perceptual_loss_add
            else:
                perceptual_loss_add = tf.constant(0)

            if not args.spatial_GAN:
                assert args.patch_gan_loss
                assert args.train_temporal_seq

            if (args.debug_mode and args.mean_estimator):
                loss_to_opt = loss + sparsity_loss
                gen_loss_GAN = tf.constant(0.0)
                discrim_loss = tf.constant(0.0)
                savers = []
                save_names = []
            elif args.patch_gan_loss:
                loss = loss + sparsity_loss
                # descriminator adapted from
                # https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
                # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
                def create_discriminator(discrim_inputs, discrim_target, sliced_feat=None, other_target=None, is_temporal=False):
                    n_layers = args.discrim_nlayers
                    layers = []

                    if (not isinstance(discrim_inputs, list)):
                        discrim_inputs = [discrim_inputs]
                    if None in discrim_inputs:
                        discrim_inputs = None

                    if not isinstance(discrim_target, list):
                        discrim_target = [discrim_target]

                    if discrim_inputs is not None:
                        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
                        if (not args.discrim_use_trace) and (not args.discrim_paired_input):
                            concat_list = discrim_inputs + discrim_target
                        elif args.discrim_use_trace:
                            concat_list = discrim_inputs + [sliced_feat]
                            if args.discrim_paired_input:
                                concat_list += discrim_target
                                concat_list += [other_target]
                            else:
                                concat_list += discrim_target
                        else:
                            concat_list = discrim_target + [other_target]

                        input = tf.concat(concat_list, axis=-1)
                    else:
                        input = tf.concat(discrim_target, axis=-1)

                    d_network = input

                    if is_temporal:
                        n_ch = args.ndf_temporal
                    else:
                        n_ch = args.ndf

                    # layer_0: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
                    # layer_1: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
                    for i in range(n_layers):
                        with tf.variable_scope("d_layer_%d" % i):
                            # in the original pytorch implementation, no bias is added for layers except for 1st and last layer, for easy implementation, apply bias on all layers for no, should visit back to modify the code if this doesn't work
                            out_channels = n_ch * 2**i
                            if i == 0:
                                d_nm = None
                            else:
                                d_nm = slim.batch_norm
                            # use batch norm as this is the default setting for PatchGAN
                            d_network = slim.conv2d(d_network, out_channels, [4, 4], stride=2, activation_fn=lrelu, padding="VALID", normalizer_fn=d_nm)

                    # layer_2: [batch, 128, 128, ndf * 2] => [batch, 125, 125, ndf * 4]
                    # layer_3: [batch, 125, 125, ndf * 4] => [batch, 122, 122, 1]
                    with tf.variable_scope("layer_%d" % (n_layers)):
                        out_channels = n_ch * 2**n_layers
                        d_network = slim.conv2d(d_network, out_channels, [4, 4], stride=1, activation_fn=lrelu, padding="VALID", normalizer_fn=slim.batch_norm)

                    with tf.variable_scope("layer_%d" % (n_layers+1)):
                        d_network = slim.conv2d(d_network, 1, [4, 4], stride=1, activation_fn=None, padding="VALID", normalizer_fn=None)


                    return d_network


                if args.is_train or args.collect_validate_loss:
                    if args.train_temporal_seq:
                        input_color = input_labels
                    else:
                        input_color = tf.stack([debug_input[:, :, :, ind] for ind in color_inds], axis=3)

                    if args.tiled_training or args.tile_only:
                        # previously using first 3 channels of debug_input, is it a bug?
                        if args.train_temporal_seq:
                            condition_input = [tf.slice(col, [0, padding_offset // 2, padding_offset // 2, 0], [args.batch_size, output_pl_h, output_pl_w, 3]) for col in input_color]
                        else:
                            condition_input = tf.slice(input_color, [0, padding_offset // 2, padding_offset // 2, 0], [args.batch_size, output_pl_h, output_pl_w, 3])
                    else:
                        condition_input = input_color

                    if args.discrim_use_trace:
                        assert not args.train_temporal_seq
                        if args.discrim_trace_shared_weights:
                            sliced_feat = tf.slice(reduced_dim_feature, [0, padding_offset // 2, padding_offset // 2, 0], [args.batch_size, output_pl_h, output_pl_w, args.conv_channel_no])
                        else:
                            # this may lead to OOM
                            with tf.name_scope("discriminator_feature_reduction"):
                                discrim_feat = feature_reduction_layer(debug_input)
                                sliced_feat = tf.slice(discrim_feat, [0, padding_offset // 2, padding_offset // 2, 0], [args.batch_size, output_pl_h, output_pl_w, args.conv_channel_no])
                    else:
                        sliced_feat = None

                    predict_real = None
                    predict_fake = None
                    if args.train_temporal_seq:
                        assert not args.discrim_use_trace
                        assert not args.discrim_paired_input
                        predict_real_single = []
                        predict_fake_single = []
                        predict_real_seq = []
                        predict_fake_seq = []
                        # BUGFIX: this will not affect inference on trained models
                        # because inference does not involve discriminators
                        # but should re-train those temporal results once there's free GPU
                        condition_input = condition_input[(args.nframes_temporal_gen-1):]


                        if args.spatial_GAN:
                            for i in range(len(generated_seq)):
                                start_ind = args.nframes_temporal_gen - 1 + i
                                with tf.name_scope("discriminator_real"):
                                    with tf.variable_scope("discriminator_single", reuse=tf.AUTO_REUSE):
                                        predict_real_single.append(create_discriminator(condition_input[i], output[:, :, :, 3*start_ind:3*start_ind+3]))
                                with tf.name_scope("discriminator_fake"):
                                    with tf.variable_scope("discriminator_single", reuse=tf.AUTO_REUSE):
                                        predict_fake_single.append(create_discriminator(condition_input[i], generated_seq[i]))

                        for i in range(0, len(generated_seq), args.nframes_temporal_discrim):
                            start_ind = args.nframes_temporal_gen - 1 + i
                            with tf.name_scope("discriminator_real"):
                                with tf.variable_scope("discriminator_seq", reuse=tf.AUTO_REUSE):
                                    predict_real_seq.append( create_discriminator(condition_input[i:i+args.nframes_temporal_discrim], output[:, :, :, 3*start_ind:3*start_ind+3*args.nframes_temporal_discrim], is_temporal=True))
                            with tf.name_scope("discriminator_fake"):
                                with tf.variable_scope("discriminator_seq", reuse=tf.AUTO_REUSE):
                                    predict_fake_seq.append(create_discriminator(condition_input[i:i+args.nframes_temporal_discrim], generated_seq[i:i+args.nframes_temporal_discrim], is_temporal=True))

                    else:
                        with tf.name_scope("discriminator_real"):
                            with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
                                predict_real = create_discriminator(condition_input, output, sliced_feat, network)

                        with tf.name_scope("discriminator_fake"):
                            with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
                                predict_fake = create_discriminator(condition_input, network, sliced_feat, output)

                    if args.gan_loss_style == 'wgan':
                        with tf.name_scope("discriminator_sample"):
                            with tf.variable_scope("discriminator", reuse=True):
                                t = tf.random_uniform([], 0.0, 1.0)
                                sampled_data = t * output + (1 - t) * network
                                predict_sample = create_discriminator(condition_input, sampled_data, sliced_feat, output)
                    else:
                        predict_sample = None
                        sampled_data = None


                    #dest='gan_loss_style', default='cross_entropy'

                    def cross_entropy_gan(data, label):
                        if label == True:
                            return tf.nn.sigmoid_cross_entropy_with_logits(logits=data, labels=tf.ones_like(data))
                        else:
                            return tf.nn.sigmoid_cross_entropy_with_logits(logits=data, labels=tf.zeros_like(data))

                    def wgan(data, label):
                        if label == True:
                            return -data
                        else:
                            return data

                    def lsgan(data, label):
                        if label == True:
                            return (data - 1) ** 2
                        else:
                            return data ** 2

                    if args.gan_loss_style == 'cross_entropy':
                        gan_loss_func =  cross_entropy_gan
                    elif args.gan_loss_style == 'wgan':
                        gan_loss_func = wgan
                    elif args.gan_loss_style == 'lsgan':
                        gan_loss_func = lsgan
                    else:
                        raise

                    with tf.name_scope("discriminator_loss"):
                        #loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_real, labels=tf.ones_like(predict_real))
                        #loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_fake, labels=tf.zeros_like(predict_fake))

                        if predict_real is not None and predict_fake is not None:
                            loss_real = gan_loss_func(predict_real, True)
                            loss_fake = gan_loss_func(predict_fake, False)

                            discrim_loss = tf.reduce_mean(loss_real + loss_fake)
                        else:
                            loss_real_sp = 0.0
                            loss_fake_sp = 0.0
                            loss_real_tp = 0.0
                            loss_fake_tp = 0.0

                            for predict in predict_real_single:
                                loss_real_sp += tf.reduce_mean(gan_loss_func(predict, True))
                            loss_real_sp /= max(len(predict_real_single), 1)

                            for predict in predict_fake_single:
                                loss_fake_sp += tf.reduce_mean(gan_loss_func(predict, False))
                            loss_fake_sp /= max(len(predict_fake_single), 1)

                            for predict in predict_real_seq:
                                loss_real_tp += tf.reduce_mean(gan_loss_func(predict, True))
                            loss_real_tp /= len(predict_real_seq)

                            for predict in predict_fake_seq:
                                loss_fake_tp += tf.reduce_mean(gan_loss_func(predict, False))
                            loss_fake_tp /= len(predict_fake_seq)

                            discrim_loss = (loss_real_sp + loss_real_tp + loss_fake_sp + loss_fake_tp) * 0.5

                        if args.gan_loss_style == 'wgan':
                            sample_gradient = tf.gradients(predict_sample, sampled_data)[0]
                            # modify the penalty a little bit, don't want to deal with sqrt in tensorflow since it's not stable
                            gradient_penalty = tf.reduce_mean((sample_gradient ** 2.0 - 1.0) ** 2.0)
                            discrim_loss = discrim_loss + 10.0 * gradient_penalty

                    with tf.name_scope("discriminator_train"):

                        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
                        discrim_optim = tf.train.AdamOptimizer(learning_rate=args.learning_rate)

                        if args.discrim_train_steps > 1:
                            step_count = tf.placeholder(tf.int32)
                            discrim_step = tf.cond(tf.floormod(step_count, args.discrim_train_steps) < 1, lambda: discrim_optim.minimize(discrim_loss, var_list=discrim_tvars), lambda: tf.no_op())
                        else:
                            discrim_step = discrim_optim.minimize(discrim_loss, var_list=discrim_tvars)

                    discrim_saver = tf.train.Saver(discrim_tvars, max_to_keep=1000)


                    with tf.name_scope("generator_loss"):
                        #gen_loss_GAN = tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_fake, labels=tf.ones_like(predict_fake))
                        gen_loss_GAN = 0
                        if predict_fake is not None:
                            gen_loss_GAN = gan_loss_func(predict_fake, True)
                            gen_loss_GAN = tf.reduce_mean(gen_loss_GAN)
                        else:
                            gen_loss_fake_sp = 0.0
                            gen_loss_fake_tp = 0.0

                            for predict in predict_fake_single:
                                gen_loss_fake_sp += tf.reduce_mean(gan_loss_func(predict, True))
                            gen_loss_fake_sp /= max(len(predict_fake_single), 1)

                            for predict in predict_fake_seq:
                                gen_loss_fake_tp += tf.reduce_mean(gan_loss_func(predict, True))
                            gen_loss_fake_tp /= len(predict_fake_seq)

                            gen_loss_GAN = (gen_loss_fake_sp + gen_loss_fake_tp) * 0.5

                        gen_loss = args.gan_loss_scale * gen_loss_GAN + loss

                else:
                    gen_loss = loss

                with tf.name_scope("generator_train"):
                    if args.is_train:
                        dependency = [discrim_step]
                    else:
                        dependency = []
                    with tf.control_dependencies(dependency):
                        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                        gen_optim = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
                        gen_step = gen_optim.minimize(gen_loss, var_list=gen_tvars)

                opt = gen_step

                encoder_vars = [var for var in gen_tvars if 'feature_reduction' in var.name]
                other_gen_vars = [var for var in gen_tvars if 'feature_reduction' not in var.name]

                encoder_saver = tf.train.Saver(encoder_vars, max_to_keep=1000)
                gen_saver = tf.train.Saver(other_gen_vars, max_to_keep=1000)

                if args.is_train or args.collect_validate_loss:
                    savers = [gen_saver, encoder_saver, discrim_saver]
                    save_names = ['model_gen', '%s_encoder' % args.shader_name, 'model_discrim']
                else:
                    savers = [gen_saver, encoder_saver]
                    save_names = ['model_gen', '%s_encoder' % args.shader_name]

                loss_to_opt = loss

            else:
                loss_to_opt = loss + sparsity_loss
                gen_loss_GAN = tf.constant(0.0)
                discrim_loss = tf.constant(0.0)

                with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):

                    adam_optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
                    var_list = tf.trainable_variables()

                    adam_before = adam_optimizer

                    opt=adam_optimizer.minimize(loss_to_opt,var_list=var_list)

                all_vars = tf.trainable_variables()

                encoder_vars = [var for var in all_vars if 'feature_reduction' in var.name]

                generator_vars = [var for var in all_vars if 'feature_reduction' not in var.name]

                encoder_saver = tf.train.Saver(encoder_vars, max_to_keep=1000)
                gen_saver = tf.train.Saver(generator_vars, max_to_keep=1000)

                savers = [encoder_saver, gen_saver]
                save_names = ['%s_encoder' % args.shader_name, 'model_gen']




            #print("initialize local vars")
            sess.run(tf.local_variables_initializer())
            #print("initialize global vars")
            sess.run(tf.global_variables_initializer())

            read_from_epoch = False

            if (not (args.debug_mode and args.mean_estimator)) and (not args.collect_validate_loss):

                ckpts = [None] * len(savers)

                if args.read_from_best_validation:
                    assert not args.is_train
                    for c_i in range(len(savers)):
                        ckpts[c_i] = tf.train.get_checkpoint_state(os.path.join(args.name, 'best_val', save_names[c_i]))
                    if None in ckpts:
                        print('No best validation result exist')
                        raise
                    read_from_epoch = False
                
                else:
                    
                    if not args.is_train:
                        # in this version we do not save models to root directory anymore
                        assert args.which_epoch > 0
                        encoder_epoch = args.which_epoch
                        others_epoch = args.which_epoch
                        read_from_epoch = True
                    else:
                        encoder_epoch = global_e - 1

                        if shader_ind == 0:
                            others_epoch = global_e - 1
                        else:
                            others_epoch = global_e
                            
                        read_from_epoch = False
                        
                    encoder_saver_exist = True
                    other_saver_exist = True
                        
                    for c_i in range(len(savers)):
                        if savers[c_i] == encoder_saver:
                            ckpts[c_i] = tf.train.get_checkpoint_state(os.path.join(args.name, "%04d"%int(encoder_epoch), save_names[c_i]))
                            if ckpts[c_i] is None:
                                encoder_saver_exist = False
                        else:
                            ckpts[c_i] = tf.train.get_checkpoint_state(os.path.join(args.name, "%04d"%int(others_epoch), save_names[c_i]))
                            if ckpts[c_i] is None:
                                other_saver_exist = False
                                
                    if not other_saver_exist:
                        assert not read_from_epoch
                        assert global_e == 1 and shader_ind == 0
                    
                    if not encoder_saver_exist:
                        assert not read_from_epoch
                        assert global_e == 1
                    

                for c_i in range(len(ckpts)):
                    if ckpts[c_i] is not None:
                        ckpt = ckpts[c_i]
                        print('loaded '+ ckpt.model_checkpoint_path)
                        savers[c_i].restore(sess, ckpt.model_checkpoint_path)
                print('finished loading')




            num_epoch = args.epoch_per_shader



            def read_ind(img_arr, name_arr, id, is_npy):
                img_arr[id] = read_name(name_arr[id], is_npy)
                if img_arr[id] is None:
                    return False
                elif img_arr[id].shape[0] * img_arr[id].shape[1] > 2200000:
                    img_arr[id] = None
                    return False
                return True

            def read_name(name, is_npy, is_bin=False):
                if not os.path.exists(name):
                    return None
                if not is_npy and not is_bin:
                    return np.float32(cv2.imread(name, -1)) / 255.0
                elif is_npy:
                    ans = np.load(name)
                    return ans
                else:
                    return np.fromfile(name, dtype=np.float32).reshape([640, 960, args.input_nc])

            if args.preload and args.is_train:
                output_images = np.empty([camera_pos_vals.shape[0], output_pl_h, output_pl_w, 3])
                all_grads = [None] * camera_pos_vals.shape[0]
                all_adds = np.empty([camera_pos_vals.shape[0], output_pl_h, output_pl_w, 1])
                for id in range(camera_pos_vals.shape[0]):
                    output_images[id, :, :, :] = read_name(output_names[id], False)
                    print(id)
                    if args.additional_input:
                        all_adds[id, :, :, 0] = read_name(add_names[id], True)

            if args.analyze_channel:
                g_channel = tf.abs(tf.gradients(loss_l2, sparsity_vec, stop_gradients=tf.trainable_variables())[0])

                feature_map_grad = tf.gradients(loss_l2, debug_input, stop_gradients=tf.trainable_variables())[0]
                feature_map_taylor = tf.reduce_mean(tf.abs(tf.reduce_mean(feature_map_grad * debug_input, (1, 2))), 0)

                # use 3 metrics
                # distance go gt (good example)
                # distance to RGB+aux result (bad example)
                # distance to current inference (network true example)
                if not args.analyze_current_only:
                    g_good = np.zeros(args.input_nc)
                    g_bad = np.zeros(args.input_nc)
                    if args.test_training:
                        bad_dir = os.path.join(args.bad_example_base_dir, 'train')
                    else:
                        bad_dir = os.path.join(args.bad_example_base_dir, 'test')

                g_current = np.zeros(args.input_nc)

                taylor_exp_vals = np.zeros(args.input_nc)

                feed_dict = {}
                current_dir = 'train' if args.test_training else 'test'
                current_dir = os.path.join(args.name, current_dir)
                if args.tile_only:
                    if inference_entire_img_valid:
                        feed_dict[h_start] = np.array([- padding_offset // 2]).astype('i')
                        feed_dict[w_start] = np.array([- padding_offset // 2]).astype('i')

                if os.path.isdir(current_dir):
                    render_current = False
                else:
                    os.makedirs(current_dir)
                    render_current = True
                for i in range(len(val_img_names)):
                    print(i)
                    output_ground = np.expand_dims(read_name(val_img_names[i], False, False), 0)
                    if not args.analyze_current_only:
                        bad_example = np.expand_dims(read_name(os.path.join(bad_dir, '%06d.png' % (i+1)), False, False), 0)
                    camera_val = np.expand_dims(camera_pos_vals[i, :], axis=1)
                    feed_dict[camera_pos] = camera_val
                    feed_dict[shader_time] = time_vals[i:i+1]

                    if not inference_entire_img_valid:
                        feed_dict[h_start] = tile_start_vals[i:i+1, 0] - padding_offset // 2
                        feed_dict[w_start] = tile_start_vals[i:i+1, 1] - padding_offset // 2

                    if render_current:
                        current_output = sess.run(network, feed_dict=feed_dict)
                        cv2.imwrite('%s/%06d.png' % (current_dir, i+1), np.uint8(255.0 * np.clip(current_output[0, :, :, :], 0.0, 1.0)))
                    else:
                        current_output = np.expand_dims(read_name(os.path.join(current_dir, '%06d.png' % (i+1)), False, False), 0)
                    if not args.analyze_current_only:
                        feed_dict[output_pl] = output_ground
                        g_good += sess.run(g_channel, feed_dict=feed_dict)
                        feed_dict[output_pl] = bad_example
                        g_bad += sess.run(g_channel, feed_dict=feed_dict)

                    feed_dict[output_pl] = current_output
                    g_current += sess.run(g_channel, feed_dict=feed_dict)

                    feed_dict[output_pl] = output_ground
                    taylor_exp_vals += sess.run(feature_map_taylor, feed_dict=feed_dict)

                if not args.analyze_current_only:
                    g_good /= len(val_img_names)
                    g_bad /= len(val_img_names)
                    numpy.save(os.path.join(current_dir, 'g_good.npy'), g_good)
                    numpy.save(os.path.join(current_dir, 'g_bad.npy'), g_bad)

                g_current /= len(val_img_names)
                numpy.save(os.path.join(current_dir, 'g_current.npy'), g_current)

                taylor_exp_vals /= len(val_img_names)
                numpy.save(os.path.join(current_dir, 'taylor_exp_vals.npy'), taylor_exp_vals)

                logbins = np.logspace(-7, np.log10(np.max(np.abs(g_current))), 11)
                logbins[0] = 0

                if not args.analyze_current_only:
                    figure = pyplot.figure()
                    ax = pyplot.subplot(211)
                    pyplot.bar(np.arange(args.input_nc), np.abs(g_good), width=2)
                    ax.set_title('Barplot for gradient magnitude of each channel')
                    ax = pyplot.subplot(212)
                    pyplot.hist(np.abs(g_good), bins=logbins)
                    pyplot.xscale('log')
                    ax.set_title('Histogram for gradient magnitude of channels')
                    pyplot.tight_layout()
                    figure.savefig('%s/g_good.png' % current_dir)
                    pyplot.close(figure)

                    figure = pyplot.figure()
                    ax = pyplot.subplot(211)
                    pyplot.bar(np.arange(args.input_nc), numpy.sort(np.abs(g_good))[::-1], width=2)
                    ax.set_title('Barplot for sorted gradient magnitude of each channel')
                    ax = pyplot.subplot(212)
                    pyplot.hist(np.abs(g_good), bins=logbins)
                    pyplot.xscale('log')
                    ax.set_title('Histogram for gradient magnitude of channels')
                    pyplot.tight_layout()
                    figure.savefig('%s/g_good_sorted.png' % current_dir)
                    pyplot.close(figure)

                    figure = pyplot.figure()
                    ax = pyplot.subplot(211)
                    pyplot.bar(np.arange(args.input_nc), np.abs(g_bad), width=2)
                    ax.set_title('Barplot for gradient magnitude of each channel')
                    ax = pyplot.subplot(212)
                    pyplot.hist(np.abs(g_bad), bins=logbins)
                    pyplot.xscale('log')
                    ax.set_title('Histogram for gradient magnitude of channels')
                    pyplot.tight_layout()
                    figure.savefig('%s/g_bad.png' % current_dir)
                    pyplot.close(figure)

                figure = pyplot.figure()
                ax = pyplot.subplot(211)
                pyplot.bar(np.arange(args.input_nc), np.abs(g_current), width=2)
                ax.set_title('Barplot for gradient magnitude of each channel')
                ax = pyplot.subplot(212)
                pyplot.hist(np.abs(g_current), bins=logbins)
                pyplot.xscale('log')
                ax.set_title('Histogram for gradient magnitude of channels')
                pyplot.tight_layout()
                figure.savefig('%s/g_current.png' % current_dir)
                pyplot.close(figure)

                valid_inds = np.load(os.path.join(args.name, 'valid_inds.npy'))
                if not args.analyze_current_only:
                    max_g_good_ind = np.argsort(np.abs(g_good))[::-1]
                    print('max activation for g_good:')
                    print(valid_inds[max_g_good_ind[:20]])
                    max_g_bad_ind = np.argsort(np.abs(g_bad))[::-1]
                    print('max activation for g_bad:')
                    print(valid_inds[max_g_bad_ind[:20]])
                max_g_current_ind = np.argsort(np.abs(g_current))[::-1]
                print('max activation for g_current:')
                print(valid_inds[max_g_current_ind[:20]])
                return

            if args.is_train:
                if args.write_summary:
                    if all_train_writers[shader_ind] is None:
                        all_train_writers[shader_ind] = tf.summary.FileWriter(os.path.join(args.name, args.shader_name), sess.graph)


                rec_arr_len = time_vals.shape[0]


                all=np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
                all_l2=np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
                all_training_loss = np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
                all_perceptual = np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
                all_gen_gan_loss = np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
                all_discrim_loss = np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)

                min_avg_loss = 1e20
                old_val_loss = 1e20
                

                for epoch in range(1, num_epoch+1):

                    cnt=0

                    permutation = np.random.permutation(int(nexamples * ntiles_h * ntiles_w))
                    nupdates = permutation.shape[0] if not args.use_batch else int(np.ceil(float(permutation.shape[0]) / args.batch_size))
                    sub_epochs = 1

                    feed_dict={}


                    for i in range(nupdates):


                        st=time.time()
                        start_id = i * args.batch_size
                        end_id = min(permutation.shape[0], (i+1)*args.batch_size)

                        frame_idx = (permutation[start_id:end_id] / (ntiles_w * ntiles_h)).astype('i')
                        tile_idx = (permutation[start_id:end_id] % (ntiles_w * ntiles_h)).astype('i')
                        run_options = None
                        run_metadata = None

                        if args.discrim_train_steps > 1:
                            feed_dict[step_count] = total_step_count
                            total_step_count += 1

                        T_before = time.time()
                            
                        if not args.preload:
                            

                            output_arr = np.empty([args.batch_size, output_pl_h, output_pl_w, 3])

                            for img_idx in range(frame_idx.shape[0]):
                                output_arr[img_idx, :, :, :] = read_name(output_names[frame_idx[img_idx]], False)

                            if args.additional_input:
                                additional_arr = np.empty([args.batch_size, output_pl.shape[1].value, output_pl.shape[2].value, 1])
                                for img_idx in range(frame_idx.shape[0]):
                                    additional_arr[img_idx, :, :, 0] = read_name(add_names[frame_idx[img_idx]], True)
                                    
                            

                        else:
                            output_arr = output_images[frame_idx]
                            if args.additional_input:
                                additional_arr = all_adds[frame_idx]
                                
                        T_load = time.time() - T_before

                        if args.tiled_training:
                            # TODO: finish logic when batch_size > 1
                            assert args.batch_size == 1
                            tile_idx = tile_idx[0]
                            # check if this tiled patch is all balck, if so skip this iter
                            tile_h = tile_idx // ntiles_w
                            tile_w  = tile_idx % ntiles_w
                            output_patch = output_arr[:, int(tile_h*height/ntiles_h):int((tile_h+1)*height/ntiles_h), int(tile_w*width/ntiles_w):int((tile_w+1)*width/ntiles_w), :]
                            #if numpy.sum(output_patch) == 0:
                            # skip if less than 1% of pixels are nonzero
                            if numpy.sum(output_patch > 0) < args.tiled_h * args.tiled_w * 3 / 100:
                                continue
                            output_arr = output_patch
                            for key, value in feed_dict.items():
                                if isinstance(value, numpy.ndarray) and len(value.shape) >= 3 and value.shape[1] == height and value.shape[2] == width:
                                    if len(value.shape) == 3:
                                        tiled_value = value[:, int(tile_h*height/ntiles_h):int((tile_h+1)*height/ntiles_h), int(tile_w*width/ntiles_w):int((tile_w+1)*width/ntiles_w)]
                                    else:
                                        tiled_value = value[:, int(tile_h*height/ntiles_h):int((tile_h+1)*height/ntiles_h), int(tile_w*width/ntiles_w):int((tile_w+1)*width/ntiles_w), :]
                                    feed_dict[key] = tiled_value
                            feed_dict[h_start] = numpy.array([tile_h * height / ntiles_h - padding_offset / 2])
                            feed_dict[w_start] = numpy.array([tile_w * width / ntiles_w - padding_offset / 2])

                        if args.tile_only:
                            feed_dict[h_start] = tile_start_vals[frame_idx, 0] - padding_offset / 2
                            feed_dict[w_start] = tile_start_vals[frame_idx, 1] - padding_offset / 2

                        feed_dict[output_pl] = output_arr
                        if args.additional_input:
                            feed_dict[additional_input_pl] = additional_arr

                        camera_val = camera_pos_vals[frame_idx, :].transpose()
                        feed_dict[camera_pos] = camera_val
                        feed_dict[shader_time] = time_vals[frame_idx]



                        st1 = time.time()

                        _,current, current_l2, current_training, current_perceptual, current_gen_loss_GAN, current_discrim_loss, =sess.run([opt,loss, loss_l2, loss_to_opt, perceptual_loss_add, gen_loss_GAN, discrim_loss],feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)

                        st2 = time.time()


                        if numpy.isnan(current):
                            print(frame_idx, tile_idx)
                            raise


                        if run_metadata is not None and args.generate_timeline:
                            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                            chrome_trace = fetched_timeline.generate_chrome_trace_format()
                            with open("%s/epoch%04d_step%d.json"%(args.name,epoch, i), 'w') as f:
                                f.write(chrome_trace)

                        current_slice = permutation[start_id:end_id]
                        all[current_slice]=current
                        all_l2[current_slice]=current_l2
                        all_training_loss[current_slice] = current_training
                        all_perceptual[current_slice] = current_perceptual
                        all_gen_gan_loss[current_slice] = current_gen_loss_GAN
                        all_discrim_loss[current_slice] = current_discrim_loss
                        cnt += args.batch_size if args.use_batch else 1
                        print("%d %d %.5f %.5f %.2f %.2f %.2f"%((global_e - 1) * num_epoch + epoch, cnt, current, np.mean(all[np.where(all)]), time.time()-st, st2-st1, T_load))

                    avg_loss = np.mean(all[np.where(all)])
                    avg_loss_l2 = np.mean(all_l2[np.where(all_l2)])
                    avg_training_loss = np.mean(all_training_loss)
                    avg_perceptual = np.mean(all_perceptual)
                    avg_gen_gan = np.mean(all_gen_gan_loss)
                    avg_discrim = np.mean(all_discrim_loss)

                    if min_avg_loss > avg_training_loss:
                        min_avg_loss = avg_training_loss

                    if args.write_summary:
                        summary = tf.Summary()
                        summary.value.add(tag='avg_loss', simple_value=avg_loss)
                        summary.value.add(tag='avg_loss_l2', simple_value=avg_loss_l2)
                        summary.value.add(tag='avg_training_loss', simple_value=avg_training_loss)
                        summary.value.add(tag='avg_perceptual', simple_value=avg_perceptual)
                        summary.value.add(tag='avg_gen_gan', simple_value=avg_gen_gan)
                        summary.value.add(tag='avg_discrim', simple_value=avg_discrim)
                        all_train_writers[shader_ind].add_summary(summary, (global_e - 1) * num_epoch + epoch)

                    smart_mkdir("%s/%04d/%04d/%s"%(args.name, global_e, epoch, args.shader_name))
                    target=open("%s/%04d/%04d/%s/score.txt"%(args.name, global_e, epoch, args.shader_name),'w')
                    target.write("%f"%np.mean(all[np.where(all)]))
                    target.close()

                if global_e % args.save_frequency == 0:
                    for s_i in range(len(savers)):
                        ckpt_dir = os.path.join("%s/%04d"%(args.name, global_e), save_names[s_i])
                        if not os.path.isdir(ckpt_dir):
                            os.makedirs(ckpt_dir)
                        savers[s_i].save(sess,"%s/model.ckpt" % ckpt_dir)



            if not args.is_train:

                if args.test_output_dir != '':
                    args.name = args.test_output_dir

                if args.collect_validate_loss:
                    assert args.test_training
                    dirs = sorted(os.listdir(args.name))
                    camera_pos_vals = np.load(os.path.join(args.dataroot, 'validate.npy'))
                    time_vals = np.load(os.path.join(args.dataroot, 'validate_time.npy'))
                    if args.tile_only:
                        tile_start_vals = np.load(os.path.join(args.dataroot, 'validate_start.npy'))

                    validate_imgs = []

                    for name in validate_img_names:
                        validate_imgs.append(np.expand_dims(read_name(name, False, False), 0))

                    # stored in the order of
                    # epoch, current, current_l2, current_perceptual, current_gen, current_discrim
                    all_vals = []

                    all_example_vals = np.empty([len(validate_img_names), 6])

                    for dir in dirs:
                        success = False
                        try:
                            global_e = int(dir)
                            success = True
                        except:
                            pass

                        if not success:
                            continue

                        ckpts = [None] * len(savers)
                        for c_i in range(len(savers)):
                            ckpts[c_i] = tf.train.get_checkpoint_state(os.path.join(args.name, dir, save_names[c_i]))

                        if None not in ckpts:
                            for c_i in range(len(ckpts)):
                                ckpt = ckpts[c_i]
                                print('loaded '+ ckpt.model_checkpoint_path)
                                savers[c_i].restore(sess, ckpt.model_checkpoint_path)
                        else:
                            continue

                        for ind in range(len(validate_img_names)):
                            feed_dict = {camera_pos: np.expand_dims(camera_pos_vals[ind], 1),
                                         shader_time: time_vals[ind:ind+1],
                                         output_pl: validate_imgs[ind]}

                            if args.tile_only:
                                feed_dict[h_start] = tile_start_vals[ind:ind+1, 0] - padding_offset / 2
                                feed_dict[w_start] = tile_start_vals[ind:ind+1, 1] - padding_offset / 2
                            else:
                                feed_dict[h_start] = np.array([- padding_offset / 2])
                                feed_dict[w_start] = np.array([- padding_offset / 2])

                            current, current_l2, current_perceptual, current_gen_loss_GAN, current_discrim_loss = sess.run([loss, loss_l2, perceptual_loss_add, gen_loss_GAN, discrim_loss], feed_dict=feed_dict)
                            all_example_vals[ind] = np.array([global_e, current, current_l2, current_perceptual, current_gen_loss_GAN, current_discrim_loss])

                        all_vals.append(np.mean(all_example_vals, 0))

                    all_vals = np.array(all_vals)
                    np.save(os.path.join(args.name, '%s_validation.npy' % args.shader_name), all_vals)
                    open(os.path.join(args.name, 'validation.txt'), 'w').write('validation dataset: %s\n raw data stored in: %s\n' % (args.dataroot, os.uname().nodename))
                    
                    min_idx = np.argsort(all_vals[:, 1])
                    min_epoch = all_vals[min_idx[0], 0]
                    
                    open(os.path.join(args.name, '%s_best_val_epoch.txt' % args.shader_name), 'w').write(str(int(min_epoch)))
                    
                    figure = plt.figure(figsize=(20,10))

                    plt.subplot(5, 1, 1)
                    plt.plot(all_vals[:, 0], all_vals[:, 1])
                    plt.ylabel('all non GAN loss')

                    plt.subplot(5, 1, 2)
                    plt.plot(all_vals[:, 0], all_vals[:, 2])
                    plt.ylabel('l2 loss')

                    plt.subplot(5, 1, 3)
                    plt.plot(all_vals[:, 0], all_vals[:, 3])
                    plt.ylabel('perceptual loss')

                    plt.subplot(5, 1, 4)
                    plt.plot(all_vals[:, 0], all_vals[:, 4])
                    plt.ylabel('GAN generator loss')

                    plt.subplot(5, 1, 5)
                    plt.plot(all_vals[:, 0], all_vals[:, 5])
                    plt.ylabel('GAN discrim loss')

                    plt.savefig(os.path.join(args.name, '%s_validation.png' % args.shader_name))
                    plt.close(figure)

                else:

                    if args.use_dataroot:
                        if args.render_only:
                            camera_pos_vals = np.load(args.render_camera_pos)
                            time_vals = np.load(args.render_t)
                            if not inference_entire_img_valid:
                                tile_start_vals = np.load(args.render_tile_start)

                    if args.render_only:
                        debug_dir = args.name + '/%s' % args.render_dirname
                    elif args.mean_estimator:
                        debug_dir = "%s/mean%d"%(args.name, args.estimator_samples)
                        debug_dir += '_test' if not args.test_training else '_train'
                        #debug_dir = "%s/mean%d"%('/localtmp/yuting', args.estimator_samples)
                    else:
                        #debug_dir = "%s/debug"%args.name
                        debug_dir = args.name + '/' + ('test' if not args.test_training else 'train')
                        if args.debug_mode:
                            debug_dir += '_debug'
                        #debug_dir = "%s/debug"%'/localtmp/yuting'



                    if read_from_epoch:
                        debug_dir += "_epoch_%04d"%args.which_epoch
                        
                    debug_dir = debug_dir + '_' + args.shader_name

                    if not os.path.isdir(debug_dir):
                        os.makedirs(debug_dir)

                    if args.render_only and os.path.exists(os.path.join(debug_dir, 'video.mp4')):
                        os.remove(os.path.join(debug_dir, 'video.mp4'))

                    if args.render_only:
                        shutil.copyfile(args.render_t, os.path.join(debug_dir, 'render_t.npy'))

                        shutil.copyfile(args.render_camera_pos, os.path.join(debug_dir, 'camera_pos.npy'))
                        if args.train_temporal_seq:
                            shutil.copyfile(args.render_temporal_texture, os.path.join(debug_dir, 'init_texture.npy'))

                    nburns = 10

                    if args.repeat_timing > 1:
                        nburns = 20
                        if args.train_temporal_seq:
                            time_stats = numpy.zeros(time_vals.shape[0] * args.repeat_timing * (args.inference_seq_len - args.nframes_temporal_gen + 1))
                        else:
                            time_stats = numpy.zeros(time_vals.shape[0] * args.repeat_timing)
                        time_count = 0

                    python_time = numpy.zeros(time_vals.shape[0])
                    if args.generate_timeline:
                        timeline_time = numpy.zeros(time_vals.shape[0] - nburns)


                    if args.generate_timeline:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                    else:
                        run_options = None
                        run_metadata = None

                    feed_dict = {}


                    if args.render_only:
                        if h_start.op.type == 'Placeholder':
                            feed_dict[h_start] = np.array([- padding_offset / 2])
                        if w_start.op.type == 'Placeholder':
                            feed_dict[w_start] = np.array([- padding_offset / 2])
                        else:
                            nexamples = time_vals.shape[0]

                        if args.train_temporal_seq:
                            all_previous_inputs = []

                            output_grounds = np.load(args.render_temporal_texture)
                            previous_buffer = np.zeros([1, input_pl_h, input_pl_w, 6*(args.nframes_temporal_gen-1)])

                        for i in range(nexamples):
                            #feed_dict = {camera_pos: camera_pos_vals[i:i+1, :].transpose(), shader_time: time_vals[i:i+1]}

                            if not inference_entire_img_valid:
                                feed_dict[h_start] = tile_start_vals[i:i+1, 0] - padding_offset / 2
                                feed_dict[w_start] = tile_start_vals[i:i+1, 1] - padding_offset / 2


                            feed_dict[camera_pos] = camera_pos_vals[i:i+1, :].transpose()
                            feed_dict[shader_time] = time_vals[i:i+1]

                            if args.additional_input:
                                feed_dict[additional_input_pl] = np.expand_dims(np.expand_dims(read_name(val_add_names[i], True), axis=2), axis=0)

                            if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                nruns = args.estimator_samples
                                output_buffer = numpy.zeros((1, 640, 960, 3))
                            else:
                                nruns = 1

                            if args.train_temporal_seq:
                                # 2D case
                                if i < args.nframes_temporal_gen - 1:
                                    all_previous_inputs.append(sess.run(input_label, feed_dict=feed_dict))
                                    #gt_name = os.path.join(args.dataroot, 'test_img/test_ground%d00009.png' % (i))
                                    #output_ground = np.float32(cv2.imread(gt_name, -1)) / 255.0
                                    output_ground = output_grounds[..., 3*i:3*i+3]
                                    all_previous_inputs.append(output_ground)
                                    output_image = np.expand_dims(all_previous_inputs[-1], 0)
                                else:
                                    for k in range(2*(args.nframes_temporal_gen-1)):
                                        if k % 2 == 0:
                                            previous_buffer[:, :, :, 3*k:3*k+3] = all_previous_inputs[k]
                                        else:
                                            previous_buffer[0, padding_offset//2:-padding_offset//2, padding_offset//2:-padding_offset//2, 3*k:3*k+3] = all_previous_inputs[k]
                                    feed_dict[previous_input] = previous_buffer
                                    output_image, generated_label = sess.run([network, input_label], feed_dict=feed_dict)
                                    all_previous_inputs.append(generated_label)
                                    all_previous_inputs.append(output_image[0])
                                    all_previous_inputs = all_previous_inputs[2:]


                            else:
                                for _ in range(nruns):
                                    output_image = sess.run(network, feed_dict=feed_dict)
                                    if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                        output_buffer += output_image[:, :, :, ::-1]



                            if args.mean_estimator:
                                output_image = output_image[:, :, :, ::-1]
                            if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                output_buffer /= args.estimator_samples
                                output_image[:] = output_buffer[:]

                            output_image = np.clip(output_image,0.0,1.0)
                            output_image *= 255.0
                            cv2.imwrite("%s/%06d.png"%(debug_dir, i+1),np.uint8(output_image[0,:,:,:]))

                            print('finished', i)

                        if not args.render_no_video:
                            os.system('ffmpeg %s -i %s -r 30 -c:v libx264 -preset slow -crf 0 -r 30 %s'%('', os.path.join(debug_dir, '%06d.png'), os.path.join(debug_dir, 'video.mp4')))
                            open(os.path.join(debug_dir, 'index.html'), 'w+').write("""
            <html>
            <body>
            <br><video controls><source src="video.mp4" type="video/mp4"></video><br>
            </body>
            </html>""")
                        return
                    else:

                        nexamples = time_vals.shape[0]


                        all_test = np.zeros(nexamples, dtype=float)
                        all_grad = np.zeros(nexamples, dtype=float)
                        all_l2 = np.zeros(nexamples, dtype=float)
                        all_perceptual = np.zeros(nexamples, dtype=float)
                        python_time = numpy.zeros(nexamples)

                        if args.train_temporal_seq:
                            previous_buffer = np.empty([1, input_pl_h, input_pl_w, 6*(args.nframes_temporal_gen-1)])

                        for i in range(nexamples):
                            print(i)

                            camera_val = np.expand_dims(camera_pos_vals[i, :], axis=1)
                            #feed_dict = {camera_pos: camera_val, shader_time: time_vals[i:i+1]}
                            feed_dict[camera_pos] = camera_val
                            feed_dict[shader_time] = time_vals[i:i+1]



                            if args.use_dataroot:
                                if args.train_temporal_seq:
                                    output_ground = None
                                else:
                                    output_ground = np.expand_dims(read_name(val_img_names[i], False, False), 0)


                            else:
                                output_ground = np.empty([1, args.input_h, args.input_w, 3])

                            if args.tile_only:
                                if not inference_entire_img_valid:
                                    feed_dict[h_start] = tile_start_vals[i:i+1, 0] - padding_offset / 2
                                    feed_dict[w_start] = tile_start_vals[i:i+1, 1] - padding_offset / 2
                                else:
                                    feed_dict[h_start] = np.array([- padding_offset / 2])
                                    feed_dict[w_start] = np.array([- padding_offset / 2])
                            if output_ground is not None:
                                feed_dict[output_pl] = output_ground
                            if args.additional_input:
                                feed_dict[additional_input_pl] = np.expand_dims(np.expand_dims(read_name(val_add_names[i], True), axis=2), axis=0)


                            #output_buffer = numpy.zeros(output_ground.shape)
                            if not args.train_temporal_seq:
                                output_buffer = np.zeros([1, args.input_h, args.input_w, 3])
                            else:
                                output_buffer = np.zeros([1, args.input_h, args.input_w, 3*(args.inference_seq_len - args.nframes_temporal_gen + 1)])

                            st = time.time()
                            if args.tiled_training:
                                st_sum = 0
                                timeline_sum = 0
                                l2_loss_val = 0
                                grad_loss_val = 0
                                perceptual_loss_val = 0
                                output_patch = numpy.zeros((1, int(height/ntiles_h), int(width/ntiles_w), 3))
                                if not args.mean_estimator:
                                    feed_dict[feed_samples[0]] = numpy.random.normal(size=(1, height+padding_offset, width+padding_offset))
                                    feed_dict[feed_samples[1]] = numpy.random.normal(size=(1, height+padding_offset, width+padding_offset))
                                else:
                                    feed_dict[feed_samples[0]] = numpy.random.normal(size=(args.estimator_samples, height+padding_offset, width+padding_offset))
                                    feed_dict[feed_samples[1]] = numpy.random.normal(size=(args.estimator_samples, height+padding_offset, width+padding_offset))
                                for tile_h in range(int(ntiles_h)):
                                    for tile_w in range(int(ntiles_w)):
                                        tiled_feed_dict = {}
                                        tiled_feed_dict[h_start] = np.array([tile_h * height / ntiles_h - padding_offset / 2])
                                        tiled_feed_dict[w_start] = np.array([tile_w * width / ntiles_w - padding_offset / 2])
                                        for key, value in feed_dict.items():
                                            if isinstance(value, numpy.ndarray) and len(value.shape) >= 3 and value.shape[1] == height and value.shape[2] == width:
                                                if len(value.shape) == 3:
                                                    tiled_value = value[:, int(tile_h*height/ntiles_h):int((tile_h+1)*height/ntiles_h), int(tile_w*width/ntiles_w):int((tile_w+1)*width/ntiles_w)]
                                                else:
                                                    tiled_value = value[:, int(tile_h*height/ntiles_h):int((tile_h+1)*height/ntiles_h), int(tile_w*width/ntiles_w):int((tile_w+1)*width/ntiles_w), :]
                                                tiled_feed_dict[key] = tiled_value
                                            else:
                                                tiled_feed_dict[key] = value
                                        st_before = time.time()
                                        if not args.accurate_timing:
                                            output_patch, l2_loss_patch, grad_loss_patch, perceptual_patch = sess.run([network, loss_l2, loss_add_term, perceptual_loss_add], feed_dict=tiled_feed_dict, options=run_options, run_metadata=run_metadata)
                                            st_after = time.time()
                                        else:
                                            sess.run([network], feed_dict=feed_dict)
                                            st_after = time.time()
                                            output_patch, l2_loss_patch, grad_loss_patch, perceptual_patch = sess.run([network, loss_l2, loss_add_term, perceptual_loss_add], feed_dict=tiled_feed_dict, options=run_options, run_metadata=run_metadata)
                                        st_sum += (st_after - st_before)
                                        print(st_after - st_before)
                                        output_buffer[0, int(tile_h*height/ntiles_h):int((tile_h+1)*height/ntiles_h), int(tile_w*width/ntiles_w):int((tile_w+1)*width/ntiles_w), :] = output_patch[0, :, :, :]
                                        l2_loss_val += l2_loss_patch
                                        grad_loss_val += grad_loss_patch
                                        perceptual_loss_val += perceptual_patch
                                        if args.generate_timeline:
                                            if i > nburns:
                                                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                                                #print("trace fetched")
                                                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                                                #print("chrome trace generated")
                                                timeline_data = json.loads(chrome_trace)['traceEvents']
                                                timeline_sum += read_timeline.read_time_dur(timeline_data)
                                                if i == time_vals.shape[0] - 1:
                                                    with open("%s/nn_%d_%d_%d.json"%(debug_dir, i+1, tile_w, tile_h), 'w') as f:
                                                        f.write(chrome_trace)
                                                #print("trace written")
                                print("timeline estimate:", timeline_sum)
                                output_image = output_buffer
                                if args.mean_estimator:
                                    output_image = output_image[:, :, :, ::-1]
                                l2_loss_val /= (ntiles_w * ntiles_h)
                                grad_loss_val /= (ntiles_w * ntiles_h)
                                perceptual_loss_val /=  (ntiles_w * ntiles_h)
                                print("rough time estimate:", st_sum)

                            elif args.train_temporal_seq:
                                output_grounds = []
                                all_previous_inputs = []
                                previous_buffer[:] = 0.0
                                l2_loss_val = 0.0
                                perceptual_loss_val = 0.0
                                st_sum = 0

                                for t_ind in range(args.inference_seq_len):
                                    if t_ind < args.nframes_temporal_gen - 1 or t_ind == args.inference_seq_len - 1:
                                        gt_name = os.path.join(args.dataroot, 'test_img/test_ground%d%05d.png' % (t_ind, i))
                                        output_grounds.append(np.float32(cv2.imread(gt_name, -1)) / 255.0)

                                    if t_ind < args.nframes_temporal_gen - 1:
                                        all_previous_inputs.append(sess.run(input_label, feed_dict=feed_dict))
                                        all_previous_inputs.append(output_grounds[-1])
                                    else:
                                        actual_ind = t_ind - (args.nframes_temporal_gen - 1)
                                        for k in range(2*(args.nframes_temporal_gen-1)):
                                            if k % 2 == 0:
                                                previous_buffer[:, :, :, 3*k:3*k+3] = all_previous_inputs[k]
                                            else:
                                                previous_buffer[0, padding_offset//2:-padding_offset//2, padding_offset//2:-padding_offset//2, 3*k:3*k+3] = all_previous_inputs[k]
                                        feed_dict[previous_input] = previous_buffer

                                        feed_dict[output] = np.expand_dims(output_grounds[-1], 0)



                                        for _ in range(args.repeat_timing):
                                            st_start = time.time()
                                            if not args.accurate_timing:
                                                # l2 and perceptual loss os incorrect when it's not the last seq
                                                generated_output, generated_label, l2_loss_val_single, perceptual_loss_val_single = sess.run([network, input_label, loss_l2, perceptual_loss_add], feed_dict=feed_dict)
                                                st_end = time.time()
                                            else:
                                                sess.run(network, feed_dict=feed_dict)
                                                st_end = time.time()
                                                generated_output, generated_label, l2_loss_val_single, perceptual_loss_val_single = sess.run([network, input_label, loss_l2, perceptual_loss_add], feed_dict=feed_dict)

                                            st_sum += st_end - st_start

                                            if args.repeat_timing > 1:
                                                time_stats[time_count] = st_end - st_start
                                                time_count += 1

                                        if t_ind == args.inference_seq_len - 1:
                                            l2_loss_val = l2_loss_val_single
                                            perceptual_loss_val = perceptual_loss_val_single
                                        #l2_loss_val += l2_loss_val_single
                                        #perceptual_loss_val += perceptual_loss_val_single
                                        all_previous_inputs.append(generated_label)
                                        all_previous_inputs.append(generated_output[0])
                                        all_previous_inputs = all_previous_inputs[-2*(args.nframes_temporal_gen-1):]

                                        output_buffer[0, :, :, 3*actual_ind:3*actual_ind+3] = generated_output[:]

                                    feed_dict[shader_time] += 1 / 30
                                output_image = output_buffer
                                grad_loss_val = 0.0
                                #output_ground = np.expand_dims(np.concatenate(output_grounds[args.nframes_temporal_gen-1:], 2), 0)

                            else:
                                if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                    nruns = args.estimator_samples
                                else:
                                    nruns = args.repeat_timing
                                st_sum = 0
                                timeline_sum = 0
                                for k in range(nruns):

                                    st_before = time.time()
                                    if not args.accurate_timing:
                                        output_image, l2_loss_val, grad_loss_val, perceptual_loss_val = sess.run([network, loss_l2, loss_add_term, perceptual_loss_add], options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
                                        st_after = time.time()
                                    else:
                                        sess.run(network, feed_dict=feed_dict)
                                        st_after = time.time()
                                        output_image, l2_loss_val, grad_loss_val, perceptual_loss_val = sess.run([network, loss_l2, loss_add_term, perceptual_loss_add], options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
                                    st_sum += (st_after - st_before)
                                    if args.repeat_timing > 1:
                                        time_stats[time_count] = st_after - st_before
                                        time_count += 1
                                    if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                        output_buffer += output_image[:, :, :, ::-1]

                                    output_images = [output_image]
                                st2 = time.time()

                                print("rough time estimate:", st_sum)

                                if args.mean_estimator:
                                    output_image = output_image[:, :, :, ::-1]
                                #print("output_image swap axis")
                                if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                    output_buffer /= args.estimator_samples
                                    output_image[:] = output_buffer[:]
                                if args.generate_timeline:
                                    if i > nburns:
                                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                                        #print("trace fetched")
                                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                                        #print("chrome trace generated")
                                        timeline_data = json.loads(chrome_trace)['traceEvents']
                                        timeline_sum += read_timeline.read_time_dur(timeline_data)
                                        if i == time_vals.shape[0] - 1:
                                            with open("%s/nn_%d.json"%(debug_dir, i+1), 'w') as f:
                                                f.write(chrome_trace)
                                        #print("trace written")
                                        #print("timeline estimate:", timeline_sum)
                            st2 = time.time()
                            if (not args.train_temporal_seq):
                                loss_val = np.mean((output_image - output_ground) ** 2)
                            else:
                                loss_val = l2_loss_val
                            #print("loss", loss_val, l2_loss_val * 255.0 * 255.0)
                            all_test[i] = loss_val
                            all_l2[i] = l2_loss_val
                            all_grad[i] = grad_loss_val
                            all_perceptual[i] = perceptual_loss_val

                            if output_image.shape[3] == 3:
                                output_image=np.clip(output_image,0.0,1.0)
                                output_image *= 255.0
                                cv2.imwrite("%s/%06d.png"%(debug_dir, i+1),np.uint8(output_image[0,:,:,:]))
                            else:
                                assert args.train_temporal_seq

                                assert output_image.shape[3] % 3 == 0
                                nimages = output_image.shape[3] // 3
                                output_image[:, :, :, :3*nimages] = np.clip(output_image[:, :, :, :3*nimages],0.0,1.0)
                                output_image[:, :, :, :3*nimages] *= 255.0
                                for img_id in range(nimages):
                                    if img_id == nimages - 1:
                                        cv2.imwrite("%s/%06d%d.png"%(debug_dir, i+1, img_id),np.uint8(output_image[0,:,:,3*img_id:3*img_id+3]))

                            python_time[i] = st_sum
                            if args.generate_timeline:
                                timeline_time[i-nburns] = timeline_sum
                            #print("output_image written")

                        if args.repeat_timing > 1:
                            time_stats = time_stats[nburns:]
                            numpy.save(os.path.join(debug_dir, 'time_stats.npy'), time_stats)
                            open('%s/time_stats.txt' % debug_dir, 'w').write('%f, %f, %f, %f' % (np.median(time_stats), np.percentile(time_stats, 25), np.percentile(time_stats, 75), np.std(time_stats)))

                        open("%s/all_loss.txt"%debug_dir, 'w').write("%f, %f"%(np.mean(all_l2), np.mean(all_grad)))
                        numpy.save(os.path.join(debug_dir, 'python_time.npy'), python_time)
                        #numpy.save(os.path.join(debug_dir, 'all_l2.npy'), all_l2)
                        if args.generate_timeline:
                            numpy.save(os.path.join(debug_dir, 'timeline_time.npy'), timeline_time)
                        open("%s/all_time.txt"%debug_dir, 'w').write("%f"%(np.median(python_time)))

                        print("all times saved")

                    test_dirname = debug_dir

                    target=open(os.path.join(test_dirname, 'score.txt'),'w')
                    target.write("%f"%np.mean(all_test[np.where(all_test)]))
                    target.close()
                    target=open(os.path.join(test_dirname, 'vgg.txt'),'w')
                    target.write("%f"%np.mean(all_perceptual[np.where(all_perceptual)]))
                    target.close()
                    target=open(os.path.join(test_dirname, 'vgg_same_scale.txt'),'w')
                    target.write("%f"%np.mean(all_perceptual[np.where(all_perceptual)]))
                    target.close()
                    if args.use_dataroot and all_test.shape[0] == 30:
                        score_close = np.mean(all_test[:5])
                        score_far = np.mean(all_test[5:10])
                        score_middle = np.mean(all_test[10:])
                        target=open(os.path.join(test_dirname, 'score_breakdown.txt'),'w')
                        target.write("%f, %f, %f"%(score_close, score_far, score_middle))
                        target.close()
                        perceptual_close = np.mean(all_perceptual[:5])
                        perceptual_far = np.mean(all_perceptual[5:10])
                        perceptual_middle = np.mean(all_perceptual[10:])
                        target=open(os.path.join(test_dirname, 'vgg_breakdown.txt'),'w')
                        target.write("%f, %f, %f"%(perceptual_close, perceptual_far, perceptual_middle))
                        target.close()
                        target=open(os.path.join(test_dirname, 'vgg_breakdown_same_scale.txt'),'w')
                        target.write("%f, %f, %f"%(perceptual_close, perceptual_far, perceptual_middle))
                        target.close()

                    if args.test_training:
                        grounddir = os.path.join(args.dataroot, 'train_img')
                    else:
                        grounddir = os.path.join(args.dataroot, 'test_img')

            print('time difference with last', time.time() - T0)
            T0 = time.time()

            sess.close()
    
    if orig_args.collect_validate_loss:
        # need to aggregate validation across multiple shaders
        
        all_shader_validations = []
        
        for shader_ind in range(len(all_shaders)):
            shader_name = all_shaders[shader_ind][0]
            all_vals = np.load(os.path.join(orig_args.name, '%s_validation.npy' % shader_name))
            all_shader_validations.append(all_vals)
            
        all_shader_validations = np.array(all_shader_validations)
        
        # per shader min error across all epochs
        min_err = np.min(all_shader_validations, 1)
        min_err = np.expand_dims(min_err, 1)
        
        normalized_validations = all_shader_validations / min_err
        accumulated_validations = np.sum(all_shader_validations, 0)
        
        min_idx = np.argsort(accumulated_validations[:, 1])
        min_epoch = all_shader_validations[0, min_idx[0], 0]

        val_dir = os.path.join(args.name, 'best_val')
        if os.path.isdir(val_dir):
            shutil.rmtree(val_dir)
        
        shutil.copytree(os.path.join(args.name, '%04d' % int(min_epoch)), val_dir)

        open(os.path.join(args.name, 'avg_best_val_epoch.txt'), 'w').write(str(int(min_epoch)))

if __name__ == '__main__':
    main()
