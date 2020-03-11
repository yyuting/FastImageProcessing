from __future__ import division

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot

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
#import compiler_problem
from unet import unet
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

allowed_dtypes = ['float64', 'float32', 'uint8']
no_L1_reg_other_layers = True

width = 500
height = 400

batch_norm_is_training = True

allow_nonzero = False

identity_output_layer = True

less_aggresive_ini = False

conv_padding = "SAME"
padding_offset = 32

deprecated_options = ['feature_reduction_channel_by_samples']

def get_tensors(dataroot, name, camera_pos, shader_time, output_type='remove_constant', nsamples=1, shader_name='zigzag', geometry='plane', learn_scale=False, soft_scale=False, scale_ratio=False, use_sigmoid=False, feature_w=[], color_inds=[], intersection=True, sigmoid_scaling=False, manual_features_only=False, aux_plus_manual_features=False, efficient_trace=False, collect_loop_statistic=False, h_start=0, h_offset=height, w_start=0, w_offset=width, samples=None, fov='regular', camera_pos_velocity=None, t_sigma=1/60.0, first_last_only=False, last_only=False, subsample_loops=-1, last_n=-1, first_n=-1, first_n_no_last=-1, mean_var_only=False, zero_samples=False, render_fix_spatial_sample=False, render_fix_temporal_sample=False, render_zero_spatial_sample=False, spatial_samples=None, temporal_samples=None, every_nth=-1, every_nth_stratified=False, one_hop_parent=False, target_idx=[], use_manual_index=False, manual_index_file='', additional_features=True, ignore_last_n_scale=0, include_noise_feature=False, crop_h=-1, crop_w=-1, no_noise_feature=False, relax_clipping=False, render_sigma=None, same_sample_all_pix=False, stratified_sample_higher_res=False, samples_int=[None], texture_maps=[], partial_trace=1.0, use_lstm=False, lstm_nfeatures_per_group=1, rotate=0, flip=0, use_dataroot=True, automatic_subsample=False, automate_raymarching_def=False, chron_order=False, def_loop_log_last=False, temporal_texture_buffer=False, texture_inds=[]):
    # 2x_1sample on margo
    #camera_pos = np.load('/localtmp/yuting/out_2x1_manual_carft/train.npy')[0, :]

    #feature_scale = np.load('/localtmp/yuting/out_2x1_manual_carft/train/zigzag_plane_normal_spheres/datas_rescaled_25_75_2_153/feature_scale.npy')
    #feature_bias = np.load('/localtmp/yuting/out_2x1_manual_carft/train/zigzag_plane_normal_spheres/datas_rescaled_25_75_2_153/feature_bias.npy')

    manual_features_only = manual_features_only or aux_plus_manual_features

    if output_type not in ['rgb', 'bgr']:
        if not sigmoid_scaling:
            if use_dataroot:
                feature_scale = np.load(os.path.join(dataroot, 'feature_scale.npy'))
                feature_bias = np.load(os.path.join(dataroot, 'feature_bias.npy'))
            else:
                feature_scale = 1.0
                feature_bias = 0.0

            #Q1 = np.load(os.path.join(dataroot, 'Q1.npy'))
            #Q3 = np.load(os.path.join(dataroot, 'Q3.npy'))
            #IQR = np.load(os.path.join(dataroot, 'IQR.npy'))
            tolerance = 2.0
        else:
            feature_mean = np.load(os.path.join(dataroot, 'feature_mean.npy'))
            feature_var = np.load(os.path.join(dataroot, 'feature_var.npy'))

    compiler_problem_full_name = os.path.abspath(os.path.join(name, 'compiler_problem.py'))
    if not os.path.exists(compiler_problem_full_name):
        if shader_name == 'zigzag':
            shader_args = ' render_zigzag ' + geometry + ' spheres '
        elif shader_name == 'sin_quadratic':
            shader_args = ' render_sin_quadratic ' + geometry + ' ripples '
        elif shader_name == 'bricks':
            shader_args = ' render_bricks ' + geometry + ' none '
        elif shader_name in ['mandelbrot', 'mandelbrot_tile_radius']:
            if partial_trace >= 1.0:
                shader_args = ' render_mandelbrot_tile_radius ' + geometry + ' none '
            else:
                shader_args = ' render_mandelbrot_tile_radius_short_%s ' % (str(partial_trace).replace('.', '')) + geometry + ' none '
        elif shader_name == 'mandelbrot_simplified_proxy':
            shader_args = ' render_mandelbrot_tile_radius_short_05 ' + geometry + ' none '
        elif shader_name == 'fire':
            shader_args = ' render_fire ' + geometry + ' spheres '
        elif shader_name == 'marble':
            if camera_pos_velocity is not None:
                shader_args = ' render_marble ' + geometry + ' ripples_still '
            else:
                shader_args = ' render_marble ' + geometry + ' ripples '
        elif shader_name == 'marble_manual_pick':
            shader_args = ' render_marble_manual_pick ' + geometry + ' ripples '
        elif shader_name == 'mandelbulb':
            shader_args = ' render_mandelbulb ' + geometry + ' none'
        elif shader_name == 'mandelbulb_simplified_proxy':
            shader_args = ' render_mandelbulb_simplified_proxy ' + geometry + ' none'
        elif shader_name == 'wood':
            shader_args = ' render_wood_real ' + geometry + ' none'
        elif shader_name == 'wood_staggered':
            shader_args = ' render_wood_staggered ' + geometry + ' none'
        elif shader_name == 'primitives_aliasing':
            shader_args = ' render_primitives_aliasing ' + geometry + ' none'
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
        elif shader_name in ['fluid_approxiamte', 'fluid_approximate_3pass_10f']:
            shader_args = ' render_fluid_approximate_3pass_10f ' + geometry + ' none'

        render_util_dir = os.path.abspath('../../global_opt/proj/apps')
        render_single_full_name = os.path.abspath(os.path.join(render_util_dir, 'render_single.py'))
        cwd = os.getcwd()
        os.chdir(render_util_dir)
        render_single_cmd = 'python ' + render_single_full_name + ' ' + os.path.join(cwd, name) + shader_args + ' --is-tf --code-only --log-intermediates --no_compute_g'
        if not intersection:
            render_single_cmd = render_single_cmd + ' --log_intermediates_level 1'
        #if manual_features_only:
        if True:
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
        if one_hop_parent:
            render_single_cmd = render_single_cmd + ' --one_hop_parent'
        if use_lstm or chron_order:
            render_single_cmd = render_single_cmd + ' --chron_order'
        if automatic_subsample:
            render_single_cmd = render_single_cmd + ' --automatic_subsample'
        if automatic_subsample or automate_raymarching_def:
            render_single_cmd = render_single_cmd + ' --automate_raymarching_def'
        if def_loop_log_last:
            render_single_cmd = render_single_cmd + ' --def_loop_log_last'
        entire_cmd = 'cd ' + render_util_dir + ' && ' + render_single_cmd + ' && cd ' + cwd
        ans = os.system(entire_cmd)
        #ans = subprocess.call('cd ' + render_util_dir + ' && source activate py36 && python ' + render_single_full_name + ' out ' + shader_args + ' --is-tf --code-only --log-intermediates && source activate tensorflow35 && cd ' + cwd)

        print(ans)
        os.chdir(cwd)
        #compiler_problem_old = os.path.abspath('../../global_opt/proj/apps/compiler_problem.py')
        #os.rename(compiler_problem_old, compiler_problem_full_name)

    spec = importlib.util.spec_from_file_location("module.name", compiler_problem_full_name)
    compiler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler_module)

    #Q1 = np.load('/localtmp/yuting/out_2x1_manual_carft/train/zigzag_plane_normal_spheres/Q1.npy')
    #Q3 = np.load('/localtmp/yuting/out_2x1_manual_carft/train/zigzag_plane_normal_spheres/Q3.npy')
    #IQR = np.load('/localtmp/yuting/out_2x1_manual_carft/train/zigzag_plane_normal_spheres/IQR.npy')

    # 1x_1sample on minion
    #camera_pos = np.load('/localtmp/yuting/out_1x1_manual_craft/out_1x1_manual_craft/train.npy')[0, :]

    #feature_scale = np.load('/localtmp/yuting/out_1x1_manual_craft/out_1x1_manual_craft/train/zigzag_plane_normal_spheres/feature_scale.npy')
    #feature_bias = np.load('/localtmp/yuting/out_1x1_manual_craft/out_1x1_manual_craft/train/zigzag_plane_normal_spheres/feature_bias.npy')
    #global width
    #width = 960
    #global height
    #height = 640

    #samples = [all_features[-2, :, :], all_features[-1, :, :]]

    features, vec_output, manual_features = get_render(camera_pos, shader_time, nsamples=nsamples, shader_name=shader_name, geometry=geometry, return_vec_output=True, compiler_module=compiler_module, manual_features_only=manual_features_only, aux_plus_manual_features=aux_plus_manual_features, h_start=h_start, h_offset=h_offset, w_start=w_start, w_offset=w_offset, samples=samples, fov=fov, camera_pos_velocity=camera_pos_velocity, t_sigma=t_sigma, zero_samples=zero_samples, render_fix_spatial_sample=render_fix_spatial_sample, render_fix_temporal_sample=render_fix_temporal_sample, render_zero_spatial_sample=render_zero_spatial_sample, spatial_samples=spatial_samples, temporal_samples=temporal_samples, additional_features=additional_features, include_noise_feature=include_noise_feature, no_noise_feature=no_noise_feature, render_sigma=render_sigma, same_sample_all_pix=same_sample_all_pix, stratified_sample_higher_res=stratified_sample_higher_res, samples_int=samples_int, texture_maps=texture_maps, temporal_texture_buffer=temporal_texture_buffer)
    
    # workaround if for some feature sparsification setup, RGB channels are not logged
    # also prevent aux feature from not being logged
    if efficient_trace:
        features = features + vec_output + manual_features
        
    if temporal_texture_buffer:
        out_textures = vec_output[:]
    
    if len(vec_output) > 3:
        loop_statistic = vec_output[3:]
        vec_output = vec_output[:3]
        features = features + loop_statistic
        
    
        
    if temporal_texture_buffer: 
        # hack: both features and manual_features contain texture input
        reshaped_texture_maps = []
        for i in range(len(texture_maps)):
            if w_offset == width and h_offset == height:
                reshaped_texture_maps.append(tf.expand_dims(texture_maps[i], 0))
            else:
                reshaped_texture_maps.append(tf.expand_dims(tf.pad(texture_maps[i], [[padding_offset // 2, padding_offset // 2], [padding_offset // 2, padding_offset // 2]], "CONSTANT"), 0))
            features.append(reshaped_texture_maps[-1])
            manual_features.append(reshaped_texture_maps[-1])

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

            if len(target_idx) > 0:
                target_features = [out_features[idx] for idx in target_idx]
                # a hack to make sure RGB channels are in randomly selected channels
                for vec in vec_output:
                    if vec not in target_features:
                        for k in range(len(target_features)):
                            if target_features[k] not in vec_output:
                                target_features[k] = vec
                                target_idx[k] = out_features.index(vec)
                                break
                out_features = target_features
                feature_bias = feature_bias[target_idx]
                feature_scale = feature_scale[target_idx]

            if use_manual_index:
                try:
                    manual_index_prefix = manual_index_file.replace('clustered_idx.npy', '')
                    #files = glob.glob(manual_index_prefix + '*')
                    filenames = ['cluster_result',
                                'log.txt']
                    for file in filenames:
                        old_file = manual_index_prefix + file
                        new_file = os.path.join(name, file)
                        shutil.copyfile(old_file, new_file)
                except:
                    pass

                manual_index = numpy.load(manual_index_file)
                _, idx_filename_no_path = os.path.split(manual_index_file)
                shutil.copyfile(manual_index_file, os.path.join(name, idx_filename_no_path))
                target_features = [out_features[idx] for idx in manual_index]
                # a hack to make sure RGB channels are in output features
                for vec in vec_output:
                    if vec not in target_features:
                        vec_index = out_features.index(vec)
                        target_features.append(vec)
                        manual_index = numpy.concatenate((manual_index, [vec_index]))
                out_features = target_features
                feature_bias = feature_bias[manual_index]
                feature_scale = feature_scale[manual_index]

            for vec in vec_output:
                #raw_ind = features.index(vec)
                #actual_ind = valid_inds.index(raw_ind)
                actual_ind = out_features.index(vec)
                color_inds.append(actual_ind)
                
            if temporal_texture_buffer:
                for i in range(3):
                    # hack, directly assume the first 3 channels of texture map is the starting color channel
                    actual_ind = out_features.index(reshaped_texture_maps[i])
                    color_inds.append(actual_ind)
                    
                for vec in out_textures:
                    actual_ind = out_features.index(vec)
                    texture_inds.append(actual_ind)

            if output_type not in ['rgb', 'bgr'] and (not sigmoid_scaling) and use_dataroot:
                for ind in color_inds:
                    feature_bias[ind] = 0.0
                    feature_scale[ind] = 1.0

            if stratified_sample_higher_res and use_dataroot:
                if not no_noise_feature:
                    feature_bias[-2] = 0.0
                    feature_bias[-1] = 0.5
                    feature_scale[-2] = 1.0
                    feature_scale[-1] = 1.0

            if use_lstm:
                ngroups = np.ceil(len(out_features) / lstm_nfeatures_per_group)
                npaddings = int(ngroups * lstm_nfeatures_per_group - len(out_features))
                out_features = [tf.zeros_like(out_features[0])] * npaddings + out_features
                feature_bias = np.concatenate((np.zeros(npaddings), feature_bias))
                feature_scale = np.concatenate((np.ones(npaddings), feature_scale))
            
            if output_type == 'remove_constant':
                features = tf.parallel_stack(out_features)
                features = tf.transpose(features, [1, 2, 3, 0])
                if False:
                    valid_features = [features[k] for k in valid_inds]
                    features_tensor = tf.Variable(np.empty([1, height, width, len(valid_inds)]), dtype=tf.float32, trainable=False)
                    assign_ops = []
                    for k in range(len(valid_inds)):
                        assign_ops.append(features_tensor[:, :, :, k].assign(valid_features[k]))
                    with tf.control_dependencies(assign_ops):
                        features = tf.identity(features_tensor)

            elif output_type == 'all':
                features = tf.cast(tf.stack(features, axis=3), tf.float32)
            elif output_type in ['rgb', 'bgr']:
                features = tf.cast(tf.stack(vec_output, axis=3), tf.float32)
                if output_type == 'bgr':
                    features = features[..., ::-1]
            else:
                raise

            if (output_type not in ['rgb', 'bgr']) and not learn_scale:
                if sigmoid_scaling:
                    # tf.sigmoid is able to handle inf and -inf correctly
                    features -= feature_mean
                    features /= feature_var
                    features = tf.sigmoid(features)
                    color_features = [tf.expand_dims(valid_features[k], axis=3) for k in color_inds]
                    features = tf.concat([features] + color_features, axis=3)
                    valid_features_len = len(valid_inds)
                    color_inds[0] = valid_features_len-3
                    color_inds[1] = valid_features_len-2
                    color_inds[2] = valid_features_len-1
                else:
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
            elif learn_scale:
                old_features = features
                features = tf.where(tf.is_finite(old_features), features, tf.zeros_like(features))
                if not scale_ratio:
                    feature_bias_w = tf.Variable(feature_bias, name='feature_bias')
                    feature_scale_w = tf.Variable(feature_scale, name='feature_scale')
                    features += feature_bias_w
                    features *= feature_scale_w
                else:
                    feature_bias_w = tf.Variable(numpy.ones_like(feature_bias), name='feature_bias')
                    feature_scale_w = tf.Variable(numpy.ones_like(feature_scale), name='feature_scale')
                    features += feature_bias_w
                    features *= feature_scale_w
                features = tf.where(tf.is_finite(old_features), features, tf.zeros_like(features))
                feature_w.append(feature_bias_w)
                feature_w.append(feature_scale_w)
            if soft_scale:
                if not use_sigmoid:
                    features = 0.5 * (tf.erf(10.0 * features) + tf.erf(10.0 * (1 - features)))
                else:
                    features = tf.sigmoid(30.0 * features) + tf.sigmoid(30.0 * (1 - features)) - 1.0
            elif not sigmoid_scaling:
                if not relax_clipping:
                    features_clipped = tf.clip_by_value(features, 0.0, 1.0)
                    features = features_clipped
                else:
                    features = tf.clip_by_value(features, -1.0, 2.0)
                #features = tf.minimum(tf.maximum(features, 0.0), 1.0)

            if not learn_scale:
                features = tf.where(tf.is_nan(features), tf.zeros_like(features), features)
    if crop_h > 0:
        features = features[:, :crop_h, :, :]
    if crop_w > 0:
        features = features[:, :, :crop_w, :]
        
    # TODO: it's a hack that rotate is a scalar
    # it should be a seperate value for each sample in a batch, and they could either be rotated or not
    # but it's much easeir to code if treat rotate as a scalar for the whole batch
    features = tf.cond(rotate > 0, lambda: tf.image.rot90(features, rotate), lambda: features)
    # as long as rotate and flip are 2 seperate generated random integers, these 2 values combined can generate all 8 different random permutation in the image space
    features = tf.cond(flip > 0, lambda: tf.image.flip_left_right(features), lambda: features)
    return features

    #numpy.save('valid_inds.npy', valid_inds)
    #return features

def get_render(camera_pos, shader_time, samples=None, nsamples=1, shader_name='zigzag', color_inds=None, return_vec_output=False, render_size=None, render_sigma=None, compiler_module=None, geometry='plane', zero_samples=False, debug=[], extra_args=[None], render_g=False, manual_features_only=False, aux_plus_manual_features=False, fov='regular', h_start=0, h_offset=height, w_start=0, w_offset=width, camera_pos_velocity=None, t_sigma=1/60.0, render_fix_spatial_sample=False, render_fix_temporal_sample=False, render_zero_spatial_sample=False, spatial_samples=None, temporal_samples=None, additional_features=True, include_noise_feature=False, no_noise_feature=False, same_sample_all_pix=False, stratified_sample_higher_res=False, samples_int=[None], texture_maps=[], temporal_texture_buffer=False):
    #vec_output_len = compiler_module.vec_output_len
    assert compiler_module is not None
    #if shader_name == 'zigzag':
    #    features_len = 266
    #elif shader_name == 'sin_quadratic':
    #    features_len = 267
    #elif shader_name == 'bricks':
    #    features_len = 142
    #elif shader_name == 'compiler_problem':
    #    compiler_problem_full_name = os.path.abspath('../../global_opt/proj/apps/compiler_problem')
    #    compiler_module = importlib.import_module(compiler_problem_full_name)

    #if geometry != 'none':
    #    features_len = compiler_module.f_log_intermediate_len + 7
    #else:
    #    features_len = compiler_module.f_log_intermediate_len + 2

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

    if camera_pos_velocity is not None:
        features_len_add += 7
        if fov == 'regular':
            print("creating dx/dt, dy/dt")
            # marble case
            features_len_add += 2

    features_len = compiler_module.f_log_intermediate_len + features_len_add

    vec_output_len = compiler_module.vec_output_len

    #if manual_features_only:
    if True:
        manual_features_len = compiler_module.f_log_intermediate_subset_len
        manual_depth_offset = 0
        if geometry not in ['none', 'texture', 'texture_approximate_10f']:
            manual_features_len += 1
            manual_depth_offset = 1
        if aux_plus_manual_features:
            manual_features_len += features_len_add
        if camera_pos_velocity is not None:
            manual_features_len += 6
            motion_start = compiler_module.f_log_intermediate_subset_len
        f_log_intermediate_subset = [None] * manual_features_len
    else:
        f_log_intermediate_subset = []
        manual_features_len = 0
        manual_depth_offset = 0
        
    if render_size is not None:
        global width
        global height
        width = render_size[0]
        height = render_size[1]
        
    if temporal_texture_buffer:
        texture_map_size = compiler_module.vec_output_len
        for i in range(texture_map_size):
            texture_maps.append(tf.placeholder(tf.float32, [height, width]))

    f_log_intermediate = [None] * features_len
    vec_output = [None] * vec_output_len

    if False:
        xv, yv = numpy.meshgrid(numpy.arange(width), numpy.arange(height), indexing='ij')
        xv = np.transpose(xv)
        yv = np.transpose(yv)
        xv = np.expand_dims(xv, 0)
        yv = np.expand_dims(yv, 0)
        xv_final = np.repeat(xv, nsamples, axis=0)
        yv_final = np.repeat(yv, nsamples, axis=0)
        tensor_x0 = tf.constant(xv, dtype=dtype)
        tensor_x1 = tf.constant(yv, dtype=dtype)
    #xv, yv = tf.meshgrid(w_start + tf.range(w_offset, dtype=dtype), h_start + tf.range(h_offset, dtype=dtype), indexing='ij')
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
    #tensor_x2 = shader_time * tf.constant(1.0, dtype=dtype, shape=xv.shape)
    tensor_x2 = tf.expand_dims(tf.expand_dims(shader_time, axis=1), axis=2) * tf.cast(tf.fill(tf.shape(xv), 1.0), dtype)

    if samples is None:
        #print("creating random samples")
        if stratified_sample_higher_res:
            sample1_int = tf.random_uniform(tf.shape(xv), minval=0, maxval=2, dtype=tf.int32)
            sample2_int = -tf.random_uniform(tf.shape(xv), minval=0, maxval=2, dtype=tf.int32)
            sample1 = tf.cast(sample1_int, dtype) * 0.5
            sample2 = tf.cast(sample2_int, dtype) * 0.5
            slicing = tf.stack([1 + tf.cast(2 * yv_orig, tf.int32) + sample2_int, 2 * tf.cast(xv_orig, tf.int32) + sample1_int], 3)
            samples_int[0] = slicing
        elif not same_sample_all_pix:
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
        if render_fix_temporal_sample and temporal_samples is not None:
            #print("fix temporal samples")
            sample3 = tf.constant(temporal_samples[0], dtype=dtype)

        if not stratified_sample_higher_res:
            vector3 = [tensor_x0 + render_sigma[0] * sample1, tensor_x1 + render_sigma[1] * sample2, tensor_x2]
        else:
            vector3 = [tensor_x0 + sample1, tensor_x1 + sample2, tensor_x2]
        if camera_pos_velocity is not None:
            vector3 = [vector3[0], vector3[1], vector3[2] + render_sigma[2] * sample3, render_sigma[2] * sample3]
    else:
        vector3 = [tensor_x0, tensor_x1, tensor_x2]
        sample1 = tf.zeros_like(sample1)
        sample2 = tf.zeros_like(sample2)
        #print("using zero samples")
        if camera_pos_velocity is not None:
            vector3 = [vector3[0], vector3[1], vector3[2] + render_sigma[2] * sample3, 0.0]
    #vector3 = [tensor_x0, tensor_x1, tensor_x2]
    f_log_intermediate[0] = shader_time
    f_log_intermediate[1] = camera_pos
    get_shader(vector3, f_log_intermediate, f_log_intermediate_subset, camera_pos, features_len, manual_features_len, shader_name=shader_name, color_inds=color_inds, vec_output=vec_output, compiler_module=compiler_module, geometry=geometry, debug=debug, extra_args=extra_args, render_g=render_g, manual_features_only=manual_features_only, aux_plus_manual_features=aux_plus_manual_features, fov=fov, camera_pos_velocity=camera_pos_velocity, features_len_add=features_len_add, manual_depth_offset=manual_depth_offset, additional_features=additional_features, texture_maps=texture_maps)

    # TODO: potential bug here
    # what to put if zero_samples = True
    if not no_noise_feature:
        if (additional_features or include_noise_feature):
            f_log_intermediate[features_len-2] = sample1
            f_log_intermediate[features_len-1] = sample2

        if aux_plus_manual_features:
            f_log_intermediate_subset[manual_features_len-2-manual_depth_offset] = sample1
            f_log_intermediate_subset[manual_features_len-1-manual_depth_offset] = sample2

    if camera_pos_velocity is not None:
        f_log_intermediate[features_len-3] = sample3
        f_log_intermediate[features_len-4] = camera_pos_velocity[0] * tf.cast(tf.fill(tf.shape(xv), 1.0), dtype)
        f_log_intermediate[features_len-5] = camera_pos_velocity[1] * tf.cast(tf.fill(tf.shape(xv), 1.0), dtype)
        f_log_intermediate[features_len-6] = camera_pos_velocity[2] * tf.cast(tf.fill(tf.shape(xv), 1.0), dtype)
        f_log_intermediate[features_len-7] = camera_pos_velocity[3] * tf.cast(tf.fill(tf.shape(xv), 1.0), dtype)
        f_log_intermediate[features_len-8] = camera_pos_velocity[4] * tf.cast(tf.fill(tf.shape(xv), 1.0), dtype)
        f_log_intermediate[features_len-9] = camera_pos_velocity[5] * tf.cast(tf.fill(tf.shape(xv), 1.0), dtype)
        if manual_features_only:
            f_log_intermediate_subset[motion_start  ] = f_log_intermediate[features_len-4]
            f_log_intermediate_subset[motion_start+1] = f_log_intermediate[features_len-5]
            f_log_intermediate_subset[motion_start+2] = f_log_intermediate[features_len-6]
            f_log_intermediate_subset[motion_start+3] = f_log_intermediate[features_len-7]
            f_log_intermediate_subset[motion_start+4] = f_log_intermediate[features_len-8]
            f_log_intermediate_subset[motion_start+5] = f_log_intermediate[features_len-9]

    if return_vec_output:
        return f_log_intermediate, vec_output, f_log_intermediate_subset
    else:
        return f_log_intermediate

def get_shader(x, f_log_intermediate, f_log_intermediate_subset, camera_pos, features_len, manual_features_len, shader_name='zigzag', color_inds=None, vec_output=None, compiler_module=None, geometry='plane', debug=[], extra_args=[None], render_g=False, manual_features_only=False, aux_plus_manual_features=False, fov='regular', camera_pos_velocity=None, features_len_add=7, manual_depth_offset=1, additional_features=True, texture_maps=[]):
    assert compiler_module is not None
    features_dt = []
    features = get_features(x, camera_pos, geometry=geometry, debug=debug, extra_args=extra_args, fov=fov, camera_pos_velocity=camera_pos_velocity, features_dt=features_dt)
    if vec_output is None:
        vec_output = [None] * 3

    #if manual_features_only:
    if True:
        # adding depth
        if geometry == 'plane':
            f_log_intermediate_subset[-1] = features[7]
        elif geometry in ['hyperboloid1', 'paraboloid']:
            f_log_intermediate_subset[-1] = extra_args[0]
        elif geometry not in ['none', 'texture', 'texture_approximate_10f']:
            raise

    if True:
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
                if aux_plus_manual_features:
                    f_log_intermediate_subset[manual_features_len-features_len_add-manual_depth_offset] = f_log_intermediate[features_len-features_len_add]
                    f_log_intermediate_subset[manual_features_len-features_len_add-manual_depth_offset+1] = f_log_intermediate[features_len-features_len_add+1]

                new_x = x[:]
                new_x[1] = x[1] - h
                features_neg_y = get_features(new_x, camera_pos, geometry=geometry, fov=fov)
                new_x[1] = x[1] + h
                features_pos_y = get_features(new_x, camera_pos, geometry=geometry, fov=fov)
                f_log_intermediate[features_len-features_len_add+2] = (features_pos_y[u_ind] - features_neg_y[u_ind]) / (2 * h)
                f_log_intermediate[features_len-features_len_add+3] = (features_pos_y[v_ind] - features_neg_y[v_ind]) / (2 * h)

                f_log_intermediate[features_len-features_len_add+4] = f_log_intermediate[features_len-features_len_add] * f_log_intermediate[features_len-features_len_add+3] - f_log_intermediate[features_len-features_len_add+1] * f_log_intermediate[features_len-features_len_add+2]

                if aux_plus_manual_features:
                    f_log_intermediate_subset[manual_features_len-features_len_add-manual_depth_offset+2] = f_log_intermediate[features_len-features_len_add+2]
                    f_log_intermediate_subset[manual_features_len-features_len_add-manual_depth_offset+3] = f_log_intermediate[features_len-features_len_add+3]
                    f_log_intermediate_subset[manual_features_len-features_len_add-manual_depth_offset+4] = f_log_intermediate[features_len-features_len_add+4]

            if camera_pos_velocity is not None and geometry == 'plane':
                f_log_intermediate[features_len-10] = features_dt[0]
                f_log_intermediate[features_len-11] = features_dt[1]

    if len(debug) > 0:
        vec_output[0] = debug[0]
    if not render_g:
        if len(texture_maps) > 0:
            #if not manual_features_only:
            if False:
                compiler_module.f(features, f_log_intermediate, vec_output, texture_maps=texture_maps)
            else:
                compiler_module.f(features, f_log_intermediate, vec_output, f_log_intermediate_subset, texture_maps=texture_maps)
        else:
            #if not manual_features_only:
            if False:
                compiler_module.f(features, f_log_intermediate, vec_output)
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

def get_features(x, camera_pos, geometry='plane', debug=[], extra_args=[None], fov='regular', camera_pos_velocity=None, features_dt=[]):
    if fov == 'regular':
        ray_dir = [x[0] - width / 2, x[1] + 1, width / 2]
        #print("use regular fov (90 degrees horizontally)")
    elif fov == 'small':
        ray_dir = [x[0] - width / 2, x[1] - height / 2, 1.73 * width / 2]
        #print("use small fov (60 degrees horizontally)")
    else:
        raise

    if camera_pos_velocity is None:
        ray_origin = [camera_pos[0], camera_pos[1], camera_pos[2]]
        ang1 = camera_pos[3]
        ang2 = camera_pos[4]
        ang3 = camera_pos[5]
    else:
        ray_origin = [camera_pos[0] + camera_pos_velocity[0] * x[3],
                      camera_pos[1] + camera_pos_velocity[1] * x[3],
                      camera_pos[2] + camera_pos_velocity[2] * x[3]]
        ang1 = camera_pos[3] + camera_pos_velocity[3] * x[3]
        ang2 = camera_pos[4] + camera_pos_velocity[4] * x[3]
        ang3 = camera_pos[5] + camera_pos_velocity[5] * x[3]

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

    if camera_pos_velocity is not None and geometry == 'plane' and fov == 'regular':
        # only for marble case

        h = 1e-4

        ray_dir_new = [t_ray * ray_dir_p[0] - h * camera_pos_velocity[0],
                       t_ray * ray_dir_p[1] - h * camera_pos_velocity[1],
                       t_ray * ray_dir_p[2] - h * camera_pos_velocity[2]]

        new_ang1 = ang1 + h * camera_pos_velocity[3]
        new_ang2 = ang2 + h * camera_pos_velocity[4]
        new_ang3 = ang3 + h * camera_pos_velocity[5]

        sin1 = tf.sin(new_ang1)
        cos1 = tf.cos(new_ang1)
        sin2 = tf.sin(new_ang2)
        cos2 = tf.cos(new_ang2)
        sin3 = tf.sin(new_ang3)
        cos3 = tf.cos(new_ang3)

        ray_dir_p_back = [cos2 * cos3 * ray_dir_new[0] + cos2 * sin3 * ray_dir_new[1] -sin2 * ray_dir_new[2],
                          (-cos1 * sin3 + sin1 * sin2 * cos3) * ray_dir_new[0] + (cos1 * cos3 + sin1 * sin2 * sin3) * ray_dir_new[1] + sin1 * cos2 * ray_dir_new[2],
                          (sin1 * sin3 + cos1 * sin2 * cos3) * ray_dir_new[0] + (-sin1 * cos3 + cos1 * sin2 * sin3) * ray_dir_new[1] + cos1 * cos2 * ray_dir_new[2]]

        pix_coord_x = ray_dir_p_back[0] / ray_dir_p_back[2] * width / 2 + width / 2
        pix_coord_y = ray_dir_p_back[1] / ray_dir_p_back[2] * width / 2 - 1

        numerical_dx_dt = (pix_coord_x - x[0]) / h
        numerical_dy_dt = (pix_coord_y - x[1]) / h
        features_dt.append(numerical_dx_dt)
        features_dt.append(numerical_dy_dt)

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


def image_gradients(image):
  """
  Copied from https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/image_ops_impl.py
  it's a hack only because we haven't upgraded tensorflow
  Returns image gradients (dy, dx) for each color channel.
  Both output tensors have the same shape as the input: [batch_size, h, w,
  d]. The gradient values are organized so that [I(x+1, y) - I(x, y)] is in
  location (x, y). That means that dy will always have zeros in the last row,
  and dx will always have zeros in the last column.
  Arguments:
    image: Tensor with shape [batch_size, h, w, d].
  Returns:
    Pair of tensors (dy, dx) holding the vertical and horizontal image
    gradients (1-step finite difference).
  Raises:
    ValueError: If `image` is not a 4D tensor.
  """
  if image.get_shape().ndims != 4:
    raise ValueError('image_gradients expects a 4D tensor '
                     '[batch_size, h, w, d], not %s.', image.get_shape())
  image_shape = tf.shape(image)
  batch_size, height, width, depth = tf.unstack(image_shape)
  dy = image[:, 1:, :, :] - image[:, :-1, :, :]
  dx = image[:, :, 1:, :] - image[:, :, :-1, :]

  # Return tensors with same size as original image by concatenating
  # zeros. Place the gradient [I(x+1,y) - I(x,y)] on the base pixel (x, y).
  shape = tf.stack([batch_size, 1, width, depth])
  dy = tf.concat([dy, tf.zeros(shape, image.dtype)], 1)
  dy = tf.reshape(dy, image_shape)

  shape = tf.stack([batch_size, height, 1, depth])
  dx = tf.concat([dx, tf.zeros(shape, image.dtype)], 2)
  dx = tf.reshape(dx, image_shape)

  return dy, dx

def lrelu(x):
    return tf.maximum(x*0.2,x)

def identity_initializer(in_channels=[], allow_map_to_less=False):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if not allow_nonzero:
            #print('initializing all zero')
            array = np.zeros(shape, dtype=float)
        else:
            x = np.sqrt(6.0 / (shape[2] + shape[3])) / 1.5
            array = numpy.random.uniform(-x, x, size=shape)
            #print('initializing xavier')
            #return tf.constant(array, dtype=dtype)
        cx, cy = shape[0]//2, shape[1]//2
        if len(in_channels) > 0:
            input_inds = in_channels
            output_inds = range(len(in_channels))
            #for k in range(len(in_channels)):
            #    array[cx, cy, in_channels[k], k] = 1
        elif allow_map_to_less:
            input_inds = range(min(shape[2], shape[3]))
            output_inds = input_inds
            #for i in range(min(shape[2], shape[3])):
            #    array[cx, cy, i, i] = 1
        else:
            input_inds = range(shape[2])
            output_inds = input_inds
            #for i in range(shape[2]):
            #    array[cx, cy, i, i] = 1
        for i in range(len(input_inds)):
            if less_aggresive_ini:
                array[cx, cy, input_inds[i], output_inds[i]] *= 10.0
            else:
                array[cx, cy, input_inds[i], output_inds[i]] = 1.0
        return tf.constant(array, dtype=dtype)
    return _initializer

batch_norm_only = False

def adaptive_nm(x):
    if not batch_norm_only:
        w0=tf.Variable(1.0,name='w0')
        w1=tf.Variable(0.0,name='w1')
        return w0*x+w1*slim.batch_norm(x, is_training=batch_norm_is_training) # the parameter "is_training" in slim.batch_norm does not seem to help so I do not use it
    else:
        return slim.batch_norm(x, is_training=batch_norm_is_training)

nm = adaptive_nm

conv_channel = 24
actual_conv_channel = conv_channel

dilation_remove_large = False
dilation_clamp_large = False
dilation_remove_layer = False
dilation_threshold = 8

def build(input, ini_id=True, regularizer_scale=0.0, share_weights=False, final_layer_channels=-1, identity_initialize=False, output_nc=3):
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
#    net=slim.conv2d(net,24,[3,3],rate=128,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv8')

    net=slim.conv2d(net,actual_conv_channel,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9',weights_regularizer=regularizer, padding=conv_padding)
    if final_layer_channels > 0:
        if actual_conv_channel > final_layer_channels and (not identity_initialize):
            net = slim.conv2d(net, final_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, scope='final_0', weights_regularizer=regularizer, padding=conv_padding)
            nlayers = [1, 2]
        else:
            nlayers = [0, 1, 2]
        for nlayer in nlayers:
            net = slim.conv2d(net, final_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(allow_map_to_less=True), scope='final_'+str(nlayer),weights_regularizer=regularizer, padding=conv_padding)

    if share_weights:
        net = tf.expand_dims(tf.reduce_mean(net, 0), 0)
    #print('identity last layer?', identity_initialize and identity_output_layer)
    net=slim.conv2d(net,output_nc,[1,1],rate=1,activation_fn=None,scope='g_conv_last',weights_regularizer=regularizer, weights_initializer=identity_initializer(allow_map_to_less=True) if (identity_initialize and identity_output_layer) else tf.contrib.layers.xavier_initializer(), padding=conv_padding)
    return net

def prepare_data(task):
    input_names=[]
    output_names=[]
    finetune_input_names=[]
    finetune_output_names=[]
    val_names=[]
    for dirname in ['MIT-Adobe_train_480p']:
        for i in range(1,2501):
            input_names.append("../data/%s/%06d.png"%(dirname,i))#training input image at 480p
            output_names.append("../original_results/%s/%s/%06d.png"%(task,dirname,i))#training output image at 480p
    for dirname in ['MIT-Adobe_train_random']:
        for i in range(1,2501):
            finetune_input_names.append("../data/%s/%06d.png"%(dirname,i))#training input image at random resolution
            finetune_output_names.append("../original_results/%s/%s/%06d.png"%(task,dirname,i))#training output image at random resolution
    for dirname in ['MIT-Adobe_test_1080p']:
        for i in range(1,2501):
            val_names.append("../data/%s/%06d.png"%(dirname,i))#test input image at 1080p
    return input_names,output_names,val_names,finetune_input_names,finetune_output_names

def prepare_data_zigzag(task):
    input_names=[]
    output_names=[]
    finetune_input_names=[]
    finetune_output_names=[]
    val_names=[]
    if task == "zigzag_supersampling_ground_only":
        train_input_dir = '../data/zigzag_ground/train_low_res_ground'
        train_output_dir = '../data/zigzag_ground/train_img'
        test_input_dir = '../data/zigzag_ground/test_low_res_ground'
    elif task == 'zigzag_supersampling_f_only':
        train_input_dir = '../data/zigzag_ground/train_low_res_f'
        train_output_dir = '../data/zigzag_ground/train_img'
        test_input_dir = '../data/zigzag_ground/test_low_res_f'
    elif task == 'zigzag_supersampling_features30' or task == 'zigzag_supersampling_features24':
        train_input_dir = '/localtmp/yuting/out8/datas/train_label'
        train_output_dir = '/localtmp/yuting/out8/datas/train_img'
        test_input_dir = '/localtmp/yuting/out8/datas/test_label'
    for file in sorted(os.listdir(train_input_dir)):
        input_names.append(os.path.join(train_input_dir, file))
    for file in sorted(os.listdir(train_output_dir)):
        output_names.append(os.path.join(train_output_dir, file))
    for file in sorted(os.listdir(test_input_dir)):
        val_names.append(os.path.join(test_input_dir, file))
    return input_names,output_names,val_names,finetune_input_names,finetune_output_names

def prepare_data_root(dataroot, use_weight_map=False, gradient_loss=False, additional_input=False):
    input_names=[]
    output_names=[]
    val_names=[]
    val_img_names=[]
    map_names = []
    val_map_names = []
    grad_names = []
    val_grad_names = []
    add_names = []
    val_add_names = []

    train_input_dir = os.path.join(dataroot, 'train_label')
    test_input_dir = os.path.join(dataroot, 'test_label')
    train_output_dir = os.path.join(dataroot, 'train_img')
    test_output_dir = os.path.join(dataroot, 'test_img')

    for file in sorted(os.listdir(train_input_dir)):
        input_names.append(os.path.join(train_input_dir, file))
    for file in sorted(os.listdir(train_output_dir)):
        output_names.append(os.path.join(train_output_dir, file))
    for file in sorted(os.listdir(test_input_dir)):
        val_names.append(os.path.join(test_input_dir, file))
    for file in sorted(os.listdir(test_output_dir)):
        val_img_names.append(os.path.join(test_output_dir, file))

    if use_weight_map:
        train_map_dir = os.path.join(dataroot, 'train_map')
        test_map_dir = os.path.join(dataroot, 'test_map')
        for file in sorted(os.listdir(train_map_dir)):
            map_names.append(os.path.join(train_map_dir, file))
        for file in sorted(os.listdir(test_map_dir)):
            val_map_names.append(os.path.join(test_map_dir, file))

    if gradient_loss:
        train_grad_dir = os.path.join(dataroot, 'train_grad')
        test_grad_dir = os.path.join(dataroot, 'test_grad')
        for file in sorted(os.listdir(train_grad_dir)):
            grad_names.append(os.path.join(train_grad_dir, file))
        for file in sorted(os.listdir(test_grad_dir)):
            val_grad_names.append(os.path.join(test_grad_dir, file))

    if additional_input:
        train_add_dir = os.path.join(dataroot, 'train_add')
        test_add_dir = os.path.join(dataroot, 'test_add')
        for file in sorted(os.listdir(train_add_dir)):
            add_names.append(os.path.join(train_add_dir, file))
        for file in sorted(os.listdir(test_add_dir)):
            val_add_names.append(os.path.join(test_add_dir, file))

    return input_names, output_names, val_names, val_img_names, map_names, val_map_names, grad_names, val_grad_names, add_names, val_add_names

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse_util.ArgumentParser(description='FastImageProcessing')
    parser.add_argument('--name', dest='name', default='', help='name of task')
    parser.add_argument('--dataroot', dest='dataroot', default='../data', help='directory to store training and testing data')
    parser.add_argument('--is_npy', dest='is_npy', action='store_true', help='whether input is npy format')
    parser.add_argument('--is_train', dest='is_train', action='store_true', help='state whether this is training or testing')
    parser.add_argument('--input_nc', dest='input_nc', type=int, default=-1, help='number of channels for input')
    parser.add_argument('--use_batch', dest='use_batch', action='store_true', help='whether to use batches in training')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='size of batches')
    parser.add_argument('--finetune', dest='finetune', action='store_true', help='fine tune on a previously tuned network')
    parser.add_argument('--orig_name', dest='orig_name', default='', help='name of original task that is fine tuned on')
    parser.add_argument('--orig_channel', dest='orig_channel', default='', help='list of input channels used in original tuning')
    parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='number of epochs to train, seperated by comma')
    parser.add_argument('--nsamples', dest='nsamples', type=int, default=1, help='number of samples in fine tuning input dataset')
    parser.add_argument('--debug_mode', dest='debug_mode', action='store_true', help='debug mode')
    parser.add_argument('--use_queue', dest='use_queue', action='store_true', help='whether to use queue instead of feed_dict ( when inputs are too large)')
    parser.add_argument('--no_preload', dest='preload', action='store_false', help='whether to preload data')
    parser.add_argument('--is_bin', dest='is_bin', action='store_true', help='whether input is stored in bin files')
    parser.add_argument('--upsample_scale', dest='upsample_scale', type=int, default=1, help='the scale to upsample input, must be power of 2')
    parser.add_argument('--upsample_single', dest='upsample_single', action='store_true', help='if set, upsample each channel seperately')
    parser.add_argument('--L1_regularizer_scale', dest='regularizer_scale', type=float, default=0.0, help='scale for L1 regularizer')
    parser.add_argument('--L2_regularizer_scale', dest='L2_regularizer_scale', type=float, default=0.0, help='scale for L2 regularizer')
    parser.add_argument('--upsample_shrink_feature', dest='upsample_shrink_feature', action='store_true', help="deconv layer's output channel is the shrinked")
    parser.add_argument('--clip_weights', dest='clip_weights', type=float, default=-1.0, help='abs value for clipping weights')
    parser.add_argument('--test_training', dest='test_training', action='store_true', help='use training data for testing purpose')
    parser.add_argument('--input_w', dest='input_w', type=int, default=960, help='supplemental information needed when using queue to read binary file')
    parser.add_argument('--input_h', dest='input_h', type=int, default=640, help='supplemental information needed when using queue to read binary file')
    parser.add_argument('--no_deconv', dest='deconv', action='store_false', help='use image resize to upsample')
    parser.add_argument('--deconv', dest='deconv', action='store_true', help='use deconv to upsample')
    parser.add_argument('--share_weights', dest='share_weights', action='store_true', help='share weights beetween samples')
    parser.add_argument('--naive_clip_weights_percentage', dest='clip_weights_percentage', type=float, default=0.0, help='clip weights according to given percentage')
    parser.add_argument('--which_epoch', dest='which_epoch', type=int, default=0, help='decide which epoch to read the checkpoint')
    parser.add_argument('--generate_timeline', dest='generate_timeline', action='store_true', help='generate timeline files')
    parser.add_argument('--feature_reduction', dest='encourage_sparse_features', action='store_true', help='if true, encourage selecting sparse number of features')
    parser.add_argument('--collect_validate_loss', dest='collect_validate_loss', action='store_true', help='if true, collect validation loss (and training score) and write to tensorboard')
    parser.add_argument('--validate_loss_freq', dest='validate_loss_freq', type=int, default=1, help='the frequency of computing validate loss')
    parser.add_argument('--collect_validate_while_training', dest='collect_validate_while_training', action='store_true', help='if true, collect validation loss while training')
    parser.add_argument('--clip_weights_percentage_after_normalize', dest='clip_weights_percentage_after_normalize', type=float, default=0.0, help='clip weights after being normalized in feature selection layer')
    parser.add_argument('--no_normalize', dest='normalize_weights', action='store_false', help='if specified, does not normalize weight on feature selection layer')
    parser.add_argument('--abs_normalize', dest='abs_normalize', action='store_true', help='when specified, use sum of abs values as normalization')
    parser.add_argument('--rowwise_L2_normalize', dest='rowwise_L2_normalize', action='store_true', help='when specified, normalize feature selection matrix by divide row-wise L2 norm sum, then regularize the resulting matrix with L1')
    parser.add_argument('--Frobenius_normalize', dest='Frobenius_normalize', action='store_true', help='when specified, use Frobenius norm to normalize feature selecton matrix, followed by L1 regularization')
    parser.add_argument('--add_initial_layers', dest='add_initial_layers', action='store_true', help='add initial conv layers without dilation')
    parser.add_argument('--initial_layer_channels', dest='initial_layer_channels', type=int, default=-1, help='number of channels in initial layers')
    parser.add_argument('--conv_channel_multiplier', dest='conv_channel_multiplier', type=int, default=1, help='multiplier for conv channel')
    parser.add_argument('--add_final_layers', dest='add_final_layers', action='store_true', help='add final conv layers without dilation')
    parser.add_argument('--final_layer_channels', dest='final_layer_channels', type=int, default=-1, help='number of channels in final layers')
    parser.add_argument('--dilation_remove_large', dest='dilation_remove_large', action='store_true', help='when specified, use ordinary conv layer instead of dilated conv layer with large dilation rate')
    parser.add_argument('--dilation_clamp_large', dest='dilation_clamp_large', action='store_true', help='when specified, clamp large dilation rate to a give threshold')
    parser.add_argument('--dilation_threshold', dest='dilation_threshold', type=int, default=8, help='threshold used to remove or clamp dilation')
    parser.add_argument('--dilation_remove_layer', dest='dilation_remove_layer', action='store_true', help='when specified, use less dilated conv layers')
    parser.add_argument('--update_bn', dest='update_bn', action='store_true', help='accurately update batch normalization')
    parser.add_argument('--conv_channel_no', dest='conv_channel_no', type=int, default=-1, help='directly specify number of channels for dilated conv layers')
    parser.add_argument('--mean_estimator', dest='mean_estimator', action='store_true', help='if true, use mean estimator instead of neural network')
    parser.add_argument('--estimator_samples', dest='estimator_samples', type=int, default=1, help='number of samples used in mean estimator')
    parser.add_argument('--accurate_timing', dest='accurate_timing', action='store_true', help='if true, do not calculate loss for more accurate timing')
    parser.add_argument('--bilinear_upsampling', dest='bilinear_upsampling', action='store_true', help='if true, use bilateral upsampling')
    parser.add_argument('--full_resolution', dest='full_resolution', action='store_true', help='if true, mean estimator sampled at full resolution with sample rate / sample_scale**2')
    parser.add_argument('--shader_name', dest='shader_name', default='zigzag', help='shader name used to generate shader input in GPU')
    parser.add_argument('--unet', dest='unet', action='store_true', help='if specified, use unet instead of dilated conv network')
    parser.add_argument('--unet_base_channel', dest='unet_base_channel', type=int, default=32, help='base channel (1st conv layer channel) for unet')
    parser.add_argument('--batch_norm_only', dest='batch_norm_only', action='store_true', help='if specified, use batch norm only (no adaptive normalization)')
    parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false', help='if specified, do not apply batch norm')
    parser.add_argument('--data_from_gpu', dest='data_from_gpu', action='store_true', help='if specified input data is generated from gpu on the fly')
    parser.add_argument('--learn_scale', dest='learn_scale', action='store_true', help='if specified, learn feature scale and feature biase instead of read it from disk')
    parser.add_argument('--soft_scale', dest='soft_scale', action='store_true', help='if specified, use soft scale instead of direct clipping')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.0001, help='learning rate for adam optimizer')
    parser.add_argument('--scale_ratio', dest='scale_ratio', action='store_true', help='if specified, learn the ratio of scale and bias')
    parser.add_argument('--use_sigmoid', dest='use_sigmoid', action='store_true', help='if specified, use sigmoid as soft scale')
    parser.add_argument('--identity_initialize', dest='identity_initialize', action='store_true', help='if specified, initialize weights such that output is 1 sample RGB')
    parser.add_argument('--nonzero_ini', dest='allow_nonzero', action='store_true', help='if specified, use xavier for all those supposed to be 0 entries in identity_initializer')
    parser.add_argument('--no_identity_output_layer', dest='identity_output_layer', action='store_false', help='if specified, do not use identity mapping for output layer')
    parser.add_argument('--less_aggresive_ini', dest='less_aggresive_ini', action='store_true', help='if specified, use a less aggresive way to initialize RGB weights (multiples of the original xavier weights)')
    parser.add_argument('--orig_rgb', dest='orig_rgb', action='store_true', help='if specified, original channel before finetune is rgb')
    parser.add_argument('--use_weight_map', dest='use_weight_map', action='store_true', help='if specified, use weight map to guide loss calculation')
    parser.add_argument('--render_only', dest='render_only', action='store_true', help='if specified, render using given camera pos, does not calculate loss')
    parser.add_argument('--render_camera_pos', dest='render_camera_pos', default='camera_pos.npy', help='used to render result')
    parser.add_argument('--render_t', dest='render_t', default='render_t.npy', help='used to render output')
    parser.add_argument('--render_camera_pos_velocity', dest='render_camera_pos_velocity', default='camera_pos_velocity.npy', help='used for render result')
    parser.add_argument('--render_temporal_texture', dest='render_temporal_texture', default='', help='used to provide initial temporal texture buffer')
    parser.add_argument('--gradient_loss', dest='gradient_loss', action='store_true', help='if specified, also use gradient at canny edge regions as a loss term')
    parser.add_argument('--normalize_grad', dest='normalize_grad', action='store_true', help='if specified, use normalized gradient as loss')
    parser.add_argument('--grayscale_grad', dest='grayscale_grad', action='store_true', help='if specified, use grayscale gradient as loss')
    parser.add_argument('--cos_sim', dest='cos_sim', action='store_true', help='use cosine similarity to compute gradient loss')
    parser.add_argument('--gradient_loss_scale', dest='gradient_loss_scale', type=float, default=1.0, help='scale multiplied to gradient loss')
    parser.add_argument('--gradient_loss_all_pix', dest='gradient_loss_all_pix', action='store_true', help='if specified, use all pixels to calculate gradient loss')
    parser.add_argument('--gradient_loss_canny_weight', dest='gradient_loss_canny_weight', action='store_true', help='if specified use weight map to calculate gradient loss')
    parser.add_argument('--train_res', dest='train_res', action='store_true', help='if specified, out_img = in_noisy_img + out_network')
    parser.add_argument('--two_stage_training', dest='two_stage_training', action='store_true', help='if specified, train first half epochs using RGB loss and next half epoch using RGB + gradient loss')
    parser.add_argument('--no_intersection', dest='intersection', action='store_false', help='if specified, do not include geometry intersection computation in intermediate variables')
    parser.add_argument('--multi_stage_new_minimizer', dest='new_minimizer', action='store_true', help='if specified, use a new minimizer if multiple stages exist')
    parser.add_argument('--geometry', dest='geometry', default='plane', help='geometry of shader')
    parser.add_argument('--RGB_norm', dest='RGB_norm', type=int, default=2, help='specify which p-norm to use for RGB loss')
    parser.add_argument('--weight_map_add', dest='weight_map_add', action='store_true', help='if specified, loss on weight map is added to original loss')
    parser.add_argument('--mean_estimator_memory_efficient', dest='mean_estimator_memory_efficient', action='store_true', help='if specified, use a memory efficient way to calculate mean estimator, but may not be accurate in time')
    parser.add_argument('--sigmoid_scaling', dest='sigmoid_scaling', action='store_true', help='if specified, use sigmoid scaling instead of 0-1 scaling')
    parser.add_argument('--visualize_scaling', dest='visualize_scaling', action='store_true', help='if specified, visualize every feature after scaling for the first test data')
    parser.add_argument('--visualize_ind', dest='visualize_ind', type=int, default=0, help='specifies the ind of testing dataset for visualization')
    parser.add_argument('--manual_features_only', dest='manual_features_only', action='store_true', help='if specified, use only manual features already specified in each shader program')
    parser.add_argument('--efficient_trace', dest='efficient_trace', action='store_true', help='if specified, use traces that are unique')
    parser.add_argument('--collect_loop_statistic', dest='collect_loop_statistic', action='store_true', help='if specified, use loop statistic only')
    parser.add_argument('--tiled_training', dest='tiled_training', action='store_true', help='if specified, use tiled training')
    parser.add_argument('--tiled_w', dest='tiled_w', type=int, default=240, help='default width for tiles if using tiled training')
    parser.add_argument('--tiled_h', dest='tiled_h', type=int, default=320, help='default height for tiles if using tiled training')
    parser.add_argument('--test_tiling', dest='test_tiling', action='store_true', help='debug mode to test tiling')
    parser.add_argument('--fov', dest='fov', default='regular', help='specified the camera field of view')
    parser.add_argument('--motion_blur', dest='motion_blur', action='store_true', help='if specified, input argument include velocity and angular velocity for camera pose')
    parser.add_argument('--first_last_only', dest='first_last_only', action='store_true', help='if specified, log only 1st and last iteration, do not log mean and std')
    parser.add_argument('--last_only', dest='last_only', action='store_true', help='log last iteration only')
    parser.add_argument('--subsample_loops', dest='subsample_loops', type=int, default=-1, help='log every n iter')
    parser.add_argument('--last_n', dest='last_n', type=int, default=-1, help='log last n iterations')
    parser.add_argument('--first_n', dest='first_n', type=int, default=-1, help='log first n iterations and last one')
    parser.add_argument('--first_n_no_last', dest='first_n_no_last', type=int, default=-1, help='log first n iterations')
    parser.add_argument('--mean_var_only', dest='mean_var_only', action='store_true', help='if flagged, use only mean and variance as loop statistic')
    parser.add_argument('--render_fix_spatial_sample', dest='render_fix_spatial_sample', action='store_true', help='if specified, fix spatial sample at rendering')
    parser.add_argument('--render_fix_temporal_sample', dest='render_fix_temporal_sample', action='store_true', help='if specified, fix temporal sample at rendering')
    parser.add_argument('--render_zero_spatial_sample', dest='render_zero_spatial_sample', action='store_true', help='if specified, use zero spatial sample')
    parser.add_argument('--render_fov', dest='render_fov', default='', help='if specified, can overwrite fov at render time')
    parser.add_argument('--every_nth', dest='every_nth', type=int, default=-1, help='log every nth var')
    parser.add_argument('--one_hop_parent', dest='one_hop_parent', action='store_true', help='if specified, skip log var if any of its parent is logged')
    parser.add_argument('--feature_sparsity_vec', dest='feature_sparsity_vec', action='store_true', help='if specified, multiply a vector to the features to potentially sparsely select channels')
    parser.add_argument('--zero_out_sparsity_vec', dest='zero_out_sparsity_vec', action='store_true', help='if specified, set small entries in sparsity vec to zero at inference time')
    parser.add_argument('--feature_sparsity_scale', dest='feature_sparsity_scale', type=float, default=1.0, help='multiply this weight before sparsity_loss')
    parser.add_argument('--feature_sparsity_schedule', dest='feature_sparsity_schedule', action='store_true', help='if true, use schedule the scale for feature sparsity to be linearly changing for every epoch')
    parser.add_argument('--feature_sparsity_schedule_start', dest='feature_sparsity_schedule_start', type=float, default=1.0, help='if use schedule for sparsity scale, start with this value at epoch 0')
    parser.add_argument('--feature_sparsity_schedule_end', dest='feature_sparsity_schedule_end', type=float, default=1.0, help='if use schedule for sparsity scale, end with this value for the last epoch')
    parser.add_argument('--logscale_schedule', dest='logscale_schedule', action='store_true', help='if specified, use logscale to schedule sparsity weights')
    parser.add_argument('--use_sparsity_pl', dest='use_sparsity_pl', action='store_true', help='if specified, use a placeholder to define sparsity nonzero entries')
    parser.add_argument('--sparsity_start_freeze', dest='sparsity_start_freeze', type=int, default=0, help='specifies the number of epochs that the sparsity schedule is fixed at the initial value (0)')
    parser.add_argument('--sparsity_end_freeze', dest='sparsity_end_freeze', type=int, default=0, help='spcifies the number of epochs that the sparsity schedule is fixed at the final value')
    parser.add_argument('--sparsity_vec_histogram', dest='sparsity_vec_histogram', action='store_true', help='if specified, compute the histogram of sparsity vec weights for the stored model')
    parser.add_argument('--feature_reduction_regularization_scale', dest='feature_reduction_regularization_scale', type=float, default=0.0, help='if nonzero, apply L1 regularization to the feature reduction layer with given scale')
    parser.add_argument('--sparsity_target_channel', dest='sparsity_target_channel', type=int, default=100, help='specifies the target nonzero channel entries for sparsity vec')
    parser.add_argument('--dynamic_training_samples', dest='dynamic_training_samples', action='store_true', help='if specified, for each epoch, training samples are dynamically chosen (a combination of high error samples and samples that have not been trained for a long time)')
    parser.add_argument('--dynamic_training_mode', dest='dynamic_training_mode', type=int, default=2, help='specifies the mode used for dynamic training')
    parser.add_argument('--random_target_channel', dest='random_target_channel', action='store_true', help='if specified, randomly select target number of channels, used as a baseline for sparsity experiments')
    parser.add_argument('--schedule_scale_after_freeze', dest='schedule_scale_after_freeze', action='store_true', help='if specified, weight scale is changed only after phase 1 (changing nonconvexity) is finished and phase 2 (maximum nonconvexity fixed) is started')
    parser.add_argument('--finetune_epoch', dest='finetune_epoch', type=int, default=-1, help='if specified, use checkpoint from specified epoch for finetune start point')
    parser.add_argument('--every_nth_stratified', dest='every_nth_stratified', action='store_true', help='if specified, do stratified sampling for every nth traces')
    parser.add_argument('--aux_plus_manual_features', dest='aux_plus_manual_features', action='store_true', help='if specified, use RGB+aux+manual features')
    parser.add_argument('--use_manual_index', dest='use_manual_index', action='store_true', help='if true, only use trace indexed in dataroot for training')
    parser.add_argument('--manual_index_file', dest='manual_index_file', default='index.npy', help='specifies file that stores index of trace used for training')
    parser.add_argument('--automatic_find_gpu', dest='automatic_find_gpu', action='store_true', help='if specified, automatically finds a gpu available on machine (instead of relying on slurm to allocate one)')
    parser.add_argument('--no_additional_features', dest='additional_features', action='store_false', help='if specified, do not use additional features during training')
    parser.add_argument('--ignore_last_n_scale', dest='ignore_last_n_scale', type=int, default=0, help='if nonzero, ignore the last n entries of stored feature_bias and feature_scale')
    parser.add_argument('--include_noise_feature', dest='include_noise_feature', action='store_true', help='if specified, include noise as additional features during trianing')
    parser.add_argument('--crop_w', dest='crop_w', type=int, default=-1, help='if specified, crop features / imgs on width dimension upon specified ind')
    parser.add_argument('--crop_h', dest='crop_h', type=int, default=-1, help='if specified, crop features / imgs on height dimension upon specified ind')
    parser.add_argument('--no_noise_feature', dest='no_noise_feature', action='store_true', help='if specified, do not include noise as additional features during training, will override include_noise_feature')
    parser.add_argument('--perceptual_loss', dest='perceptual_loss', action='store_true',help='if specified, use perceptual loss as well as L2 loss')
    parser.add_argument('--perceptual_loss_term', dest='perceptual_loss_term', default='conv1_1', help='specify to use which layer in vgg16 as perceptual loss')
    parser.add_argument('--perceptual_loss_scale', dest='perceptual_loss_scale', type=float, default=0.0001, help='used to scale perceptual loss')
    parser.add_argument('--relax_clipping', dest='relax_clipping', action='store_true', help='if specified relax the condition of clipping features from 0-1 to -1-2')
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
    parser.add_argument('--stratified_sample_higher_res', dest='stratified_sample_higher_res', action='store_true', help='in this mode, learn from a gt that is stratified sample from a higher res gt')
    parser.add_argument('--save_intermediate_epoch', dest='save_intermediate_epoch', action='store_true', help='if true, save all intermediate epochs')
    parser.add_argument('--texture_maps', dest='texture_maps', default='', help='if not empty, retrieve texture map from the file')
    parser.add_argument('--additional_input', dest='additional_input', action='store_true', help='if true, find additional input features from train/test_add')
    parser.add_argument('--partial_trace', dest='partial_trace', type=float, default=1.0, help='if less than 1, only record the first 100x percent of the trace wehre x = partial_trace')
    parser.add_argument('--use_lstm', dest='use_lstm', action='store_true', help='if specified, use lstm to process raw feature')
    parser.add_argument('--lstm_nfeatures_per_group', dest='lstm_nfeatures_per_group', type=int, default=1, help='specifies how many features processed per timestamp by LSTM')
    parser.add_argument('--patch_gan_loss', dest='patch_gan_loss', action='store_true', help='if specified, use a patch gan loss together with existing loss')
    parser.add_argument('--ndf', dest='ndf', type=int, default=32, help='number of discriminator filters on first layer if using patch GAN loss')
    parser.add_argument('--gan_loss_scale', dest='gan_loss_scale', type=float, default=1.0, help='the scale multiplied to GAN loss before adding to regular loss')
    parser.add_argument('--discrim_nlayers', dest='discrim_nlayers', type=int, default=2, help='number of layers of discriminator')
    parser.add_argument('--discrim_use_trace', dest='discrim_use_trace', action='store_true', help='if specified, use trace info for discriminator')
    parser.add_argument('--discrim_trace_shared_weights', dest='discrim_trace_shared_weights', action='store_true', help='if specified, when discriminator is using trace, it directly uses the 48 dimension used by the generator, and the feature reduction layer weights will be shared with the generator')
    parser.add_argument('--discrim_paired_input', dest='discrim_paired_input', action='store_true', help='if specified, use paired input (gt, generated) or (generated, gt) to discriminator and predict which order is correct')
    parser.add_argument('--save_frequency', dest='save_frequency', type=int, default=100, help='specifies the frequency to save a checkpoint')
    parser.add_argument('--discrim_train_steps', dest='discrim_train_steps', type=int, default=1, help='specified how often to update discrim')
    parser.add_argument('--gan_loss_style', dest='gan_loss_style', default='cross_entropy', help='specifies what GAN loss to use')
    parser.add_argument('--test_rotation', dest='test_rotation', action='store_true', help='if specified, test if axis rotation works')
    parser.add_argument('--train_with_random_rotation', dest='train_with_random_rotation', action='store_true', help='if specified, during training, will randomly rotate data in image space')
    parser.add_argument('--temporal_texture_buffer', dest='temporal_texture_buffer', action='store_true', help='if specified, render temporal correlated sequences, each frame is using texture rendered from previous frame')
    parser.add_argument('--learn_sigma', dest='learn_sigma', action='store_true', help='if true, sigma is learned, not set')
    parser.add_argument('--no_dataroot', dest='use_dataroot', action='store_false', help='if specified, do not need a dataroot (used for check runtime)')
    parser.add_argument('--camera_pos_file', dest='camera_pos_file', default='', help='if specified, use for no_dataroot mode')
    parser.add_argument('--camera_pos_len', dest='camera_pos_len', type=int, default=50, help='specifies the number of camera pos used in no_dataroot mode')
    parser.add_argument('--feature_size_only', dest='feature_size_only', action='store_true', help='if specified, do you further create neural network, return after collecting the feature size')
    parser.add_argument('--automatic_subsample', dest='automatic_subsample', action='store_true', help='if specified, automatically decide program subsample rate (and raymarching and function def)')
    parser.add_argument('--automate_raymarching_def', dest='automate_raymarching_def', action='store_true', help='if specified, automatically choose schedule for raymarching and function def (but not subsampling rate')
    parser.add_argument('--chron_order', dest='chron_order', action='store_true', help='if specified, log trace in their execution order')
    parser.add_argument('--def_loop_log_last', dest='def_loop_log_last', action='store_true', help='if true, log the last execution of function, else, log first execution')
    
    parser.set_defaults(is_npy=False)
    parser.set_defaults(is_train=False)
    parser.set_defaults(use_batch=False)
    parser.set_defaults(finetune=False)
    parser.set_defaults(debug_mode=False)
    parser.set_defaults(use_queue=False)
    parser.set_defaults(preload=True)
    parser.set_defaults(is_bin=False)
    parser.set_defaults(upsample_single=False)
    parser.set_defaults(upsample_shrink_feature=False)
    parser.set_defaults(test_training=False)
    parser.set_defaults(deconv=False)
    parser.set_defaults(share_weights=False)
    parser.set_defaults(generate_timeline=False)
    parser.set_defaults(encourage_sparse_features=False)
    parser.set_defaults(collect_validate_loss=False)
    parser.set_defaults(collect_validate_while_training=False)
    parser.set_defaults(normalize_weights=True)
    parser.set_defaults(abs_normalize=False)
    parser.set_defaults(rowwise_L2_normalize=False)
    parser.set_defaults(Frobenius_normalize=False)
    parser.set_defaults(add_initial_layers=False)
    parser.set_defaults(add_final_layers=False)
    parser.set_defaults(dilation_remove_large=False)
    parser.set_defaults(dilation_clamp_large=False)
    parser.set_defaults(dilation_remove_layer=False)
    parser.set_defaults(update_bn=False)
    parser.set_defaults(mean_estimator=False)
    parser.set_defaults(accurate_timing=False)
    parser.set_defaults(bilinear_upsampling=False)
    parser.set_defaults(full_resolution=False)
    parser.set_defaults(unet=False)
    parser.set_defaults(batch_norm_only=False)
    parser.set_defaults(batch_norm=True)
    parser.set_defaults(data_from_gpu=False)
    parser.set_defaults(learn_scale=False)
    parser.set_defaults(soft_scale=False)
    parser.set_defaults(scale_ratio=False)
    parser.set_defaults(use_sigmoid=False)
    parser.set_defaults(identity_initialize=False)
    parser.set_defaults(allow_nonzero=False)
    parser.set_defaults(identity_output_layer=True)
    parser.set_defaults(less_aggresive_ini=False)
    parser.set_defaults(orig_rgb=False)
    parser.set_defaults(use_weight_map=False)
    parser.set_defaults(gradient_loss=False)
    parser.set_defaults(normalize_grad=False)
    parser.set_defaults(grayscale_grad=False)
    parser.set_defaults(cos_sim=False)
    parser.set_defaults(gradient_loss_all_pix=False)
    parser.set_defaults(gradient_loss_canny_weight=False)
    parser.set_defaults(train_res=False)
    parser.set_defaults(two_stage_training=False)
    parser.set_defaults(intersection=True)
    parser.set_defaults(new_minimizer=False)
    parser.set_defaults(weight_map_add=False)
    parser.set_defaults(mean_estimator_memory_efficient=False)
    parser.set_defaults(sigmoid_scaling=False)
    parser.set_defaults(visualize_scaling=False)
    parser.set_defaults(efficient_trace=False)
    parser.set_defaults(collect_loop_statistic=False)
    parser.set_defaults(tiled_training=False)
    parser.set_defaults(test_tiling=False)
    parser.set_defaults(motion_blur=False)
    parser.set_defaults(first_last_only=False)
    parser.set_defaults(last_only=False)
    parser.set_defaults(render_fix_spatial_sample=False)
    parser.set_defaults(render_fix_temporal_sample=False)
    parser.set_defaults(render_zero_spatial_sample=False)
    parser.set_defaults(mean_var_only=False)
    parser.set_defaults(one_hop_parent=False)
    parser.set_defaults(feature_sparsity_vec=False)
    parser.set_defaults(zero_out_sparsity_vec=False)
    parser.set_defaults(feature_sparsity_schedule=False)
    parser.set_defaults(logscale_schedule=False)
    parser.set_defaults(use_sparsity_pl=False)
    parser.set_defaults(sparsity_vec_histogram=False)
    parser.set_defaults(dynamic_training_samples=False)
    parser.set_defaults(random_target_channel=False)
    parser.set_defaults(schedule_scale_after_freeze=False)
    parser.set_defaults(every_nth_stratified=False)
    parser.set_defaults(aux_plus_manual_features=False)
    parser.set_defaults(use_manual_index=False)
    parser.set_defaults(automatic_find_gpu=False)
    parser.set_defaults(additional_features=True)
    parser.set_defaults(include_noise_feature=False)
    parser.set_defaults(no_noise_feature=False)
    parser.set_defaults(perceptual_loss=False)
    parser.set_defaults(relax_clipping=False)
    parser.set_defaults(preload_grad=False)
    parser.set_defaults(train_with_zero_samples=False)
    parser.set_defaults(tile_only=False)
    parser.set_defaults(write_summary=True)
    parser.set_defaults(lpips_loss=False)
    parser.set_defaults(l2_loss=True)
    parser.set_defaults(same_sample_all_pix=False)
    parser.set_defaults(analyze_channel=False)
    parser.set_defaults(analyze_current_only=False)
    parser.set_defaults(stratified_sample_higher_res=False)
    parser.set_defaults(save_intermediate_epoch=False)
    parser.set_defaults(additional_input=False)
    parser.set_defaults(use_lstm=False)
    parser.set_defaults(patch_gan_loss=False)
    parser.set_defaults(discrim_use_trace=False)
    parser.set_defaults(discrim_trace_shared_weights=False)
    parser.set_defaults(discrim_paired_input=False)
    parser.set_defaults(test_rotation=False)
    parser.set_defaults(train_with_random_rotation=False)
    parser.set_defaults(temporal_texture_buffer=False)
    parser.set_defaults(learn_sigma=False)
    parser.set_defaults(use_dataroot=True)
    parser.set_defaults(feature_size_only=False)
    parser.set_defaults(automatic_subsample=False)
    parser.set_defaults(automate_raymarching_def=False)
    parser.set_defaults(chron_order=False)
    parser.set_defaults(def_loop_log_last=False)

    args = parser.parse_args()

    if args.automatic_find_gpu:
        print("automatically find available gpu")
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
        os.system('rm tmp')

    main_network(args)

def copy_option(args):
    new_args = copy.copy(args)
    delattr(new_args, 'is_train')
    delattr(new_args, 'dataroot')
    delattr(new_args, 'clip_weights')
    delattr(new_args, 'test_training')
    delattr(new_args, 'clip_weights_percentage')
    delattr(new_args, 'which_epoch')
    delattr(new_args, 'generate_timeline')
    delattr(new_args, 'collect_validate_loss')
    delattr(new_args, 'collect_validate_while_training')
    delattr(new_args, 'clip_weights_percentage_after_normalize')
    delattr(new_args, 'debug_mode')
    delattr(new_args, 'mean_estimator')
    delattr(new_args, 'estimator_samples')
    delattr(new_args, 'preload')
    delattr(new_args, 'accurate_timing')
    delattr(new_args, 'bilinear_upsampling')
    delattr(new_args, 'full_resolution')
    delattr(new_args, 'data_from_gpu')
    delattr(new_args, 'render_only')
    delattr(new_args, 'render_camera_pos')
    delattr(new_args, 'render_t')
    delattr(new_args, 'render_camera_pos_velocity')
    delattr(new_args, 'mean_estimator_memory_efficient')
    delattr(new_args, 'visualize_scaling')
    delattr(new_args, 'visualize_ind')
    delattr(new_args, 'test_tiling')
    delattr(new_args, 'render_fix_spatial_sample')
    delattr(new_args, 'render_fix_temporal_sample')
    delattr(new_args, 'render_zero_spatial_sample')
    delattr(new_args, 'render_fov')
    delattr(new_args, 'zero_out_sparsity_vec')
    delattr(new_args, 'sparsity_vec_histogram')
    delattr(new_args, 'automatic_find_gpu')
    delattr(new_args, 'ignore_last_n_scale')
    delattr(new_args, 'tile_only')
    delattr(new_args, 'write_summary')
    delattr(new_args, 'analyze_channel')
    delattr(new_args, 'bad_example_base_dir')
    delattr(new_args, 'analyze_current_only')
    delattr(new_args, 'save_intermediate_epoch')
    delattr(new_args, 'validate_loss_freq')
    delattr(new_args, 'test_rotation')
    delattr(new_args, 'render_temporal_texture')
    return new_args

def main_network(args):

    if args.is_bin:
        assert not args.preload

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
            for key in deprecated_options:
                if key in option_str:
                    idx = option_str.index(key)
                    next_sep = option_str.index(',', idx)
                    option_str = option_str.replace(option_str[idx:next_sep+2], '')
            assert option_str == str(option_copy)
            option_copy = copy_option(args)
            open(option_file, 'w').write(str(option_copy))
    else:
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

    assert np.log2(args.upsample_scale) == int(np.log2(args.upsample_scale))
    deconv_layers = int(np.log2(args.upsample_scale))

    global batch_norm_is_training
    batch_norm_is_training = args.is_train

    global batch_norm_only
    batch_norm_only = args.batch_norm_only

    global actual_conv_channel
    actual_conv_channel *= args.conv_channel_multiplier
    if actual_conv_channel == 0:
        actual_conv_channel = args.conv_channel_no
    if args.initial_layer_channels < 0:
        args.initial_layer_channels = actual_conv_channel
    if args.final_layer_channels < 0:
        args.final_layer_channels = actual_conv_channel
        
    #if args.which_epoch <= 0:
    #    args.which_epoch = args.epoch

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

    if not args.batch_norm:
        global nm
        nm = None
        args.update_bn = False

    global allow_nonzero
    allow_nonzero = args.allow_nonzero

    global identity_output_layer
    identity_output_layer = args.identity_output_layer

    global less_aggresive_ini
    less_aggresive_ini = args.less_aggresive_ini
    
    if args.test_rotation:
        assert not args.is_train
        
    if args.train_with_random_rotation or args.test_rotation:
        rotate = tf.placeholder(tf.int32)
        flip = tf.placeholder(tf.int32)
    else:
        rotate = tf.constant(0, tf.int32)
        flip = rotate

    if args.render_only:
        args.is_train = False
        if args.render_fov != '':
            args.fov = args.render_fov

    if args.mean_estimator_memory_efficient:
        assert not args.generate_timeline
        
    if args.temporal_texture_buffer:
        # will randomly choose sequence start point for each training iter
        args.preload = False

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

    if args.use_dataroot:
        input_names, output_names, val_names, val_img_names, map_names, val_map_names, grad_names, val_grad_names, add_names, val_add_names = prepare_data_root(args.dataroot, use_weight_map=args.use_weight_map or args.gradient_loss_canny_weight, gradient_loss=args.gradient_loss, additional_input=args.additional_input)
        if args.test_training:
            val_names = input_names
            val_img_names = output_names
            val_map_names = map_names
            val_grad_names = grad_names
            val_add_names = add_names
    else:
        args.write_summary = False

    read_data_from_file = (not args.debug_mode) and (not args.data_from_gpu)

    train_from_queue = False
    
    if not args.learn_sigma:
        render_sigma = [args.render_sigma, args.render_sigma, 0]
    else:
        learned_sigma = tf.Variable(args.render_sigma, name='generator/learned_sigma')
        render_sigma = [learned_sigma, learned_sigma, 0]

    if not read_data_from_file:
        if args.use_queue:
            args.use_queue = False
            train_from_queue = True

    if read_data_from_file:
        #is_train = tf.placeholder(tf.bool)

        if not args.use_queue:
            input=tf.placeholder(tf.float32,shape=[None,None,None,args.input_nc])
            output=tf.placeholder(tf.float32,shape=[None,None,None,3])
        else:
            if not args.use_batch:
                if args.is_train:
                    input_tensor = tf.convert_to_tensor(input_names)
                    output_tensor = tf.convert_to_tensor(output_names)
                    is_shuffle = True
                else:
                    input_tensor = tf.convert_to_tensor(val_names)
                    output_tensor = tf.convert_to_tensor(val_img_names)
                    is_shuffle = False

                input_queue, output_queue = tf.train.slice_input_producer([input_tensor, output_tensor], shuffle=is_shuffle, capacity=200)

                input = tf.decode_raw(tf.read_file(input_queue), tf.float32)

                input = tf.reshape(input, (args.input_h, args.input_w, args.input_nc))
                output = tf.to_float(tf.image.decode_image(tf.read_file(output_queue), channels=3)) / 255.0
                output.set_shape((640, 960, 3))

                #print("start batch")
                if args.is_train:
                    input, output = tf.train.batch([input, output], 1, num_threads=5, capacity=10)
                else:
                    input = tf.expand_dims(input, axis=0)
                    output = tf.expand_dims(output, axis=0)
                #print("end batch")

                #input = tf.expand_dims(input, 0)
                #output = tf.expand_dims(output, 0)

                if args.share_weights:
                    input = tf.reshape(input, (args.input_h, args.input_w, args.nsamples, nfeatures))
                    input = tf.transpose(input, perm=[2, 0, 1, 3])
                #else:
                #    input = tf.expand_dims(input, axis=0)
                #output = tf.expand_dims(output, axis=0)
            # TODO: finish logic when using batch queues
            else:
                raise
        input_to_network = input
        if args.input_nc == 3:
            color_inds = [0, 1, 2]
        elif args.input_nc == 7 and args.shader_name == 'oceanic':
            # a hack for oceanic aux
            color_inds = [1, 5, 6]
    else:
        if args.geometry == 'texture_approximate_10f':
            output_nc = 30
        else:
            output_nc = 3
                
        if (args.tiled_training or args.tile_only) and (not inference_entire_img_valid):
            output_pl_w = args.tiled_w
            output_pl_h = args.tiled_h
        else:
            output_pl_w = args.input_w
            output_pl_h = args.input_h
        if args.stratified_sample_higher_res and args.is_train:
            output_pl_w *= 2
            output_pl_h *= 2
        output_pl = tf.placeholder(tf.float32, shape=[None, output_pl_h, output_pl_w, output_nc])
        #if (args.tiled_training or args.tile_only) and (not inference_entire_img_valid):
        #    output_pl = tf.placeholder(tf.float32,shape=[None,args.tiled_h,args.tiled_w,3])
        #else:
        #    output_pl = tf.placeholder(tf.float32,shape=[None,args.input_h,args.input_w,3])
        if args.geometry != 'texture_approximate_10f':
            camera_pos = tf.placeholder(dtype, shape=[6, args.batch_size])
        else:
            camera_pos = tf.placeholder(dtype, shape=[33, args.batch_size])
        shader_time = tf.placeholder(dtype, shape=args.batch_size)
        if args.additional_input:
            additional_input_pl = tf.placeholder(dtype, shape=[None, output_pl_h, output_pl_w, 1])
        if args.motion_blur:
            camera_pos_velocity = tf.placeholder(dtype, shape=6)
        else:
            camera_pos_velocity = None
        if args.full_resolution:
            width *= args.upsample_scale
            height *= args.upsample_scale
        with tf.variable_scope("shader"):
            if args.input_nc == 266:
                output_type = 'all'
            elif args.mean_estimator:
                output_type = 'rgb'
            elif args.input_nc == 3:
                output_type = 'bgr'
            else:
                output_type = 'remove_constant'
            if args.mean_estimator and not args.mean_estimator_memory_efficient:
                shader_samples = args.estimator_samples
            else:
                shader_samples = args.batch_size
            if args.full_resolution:
                shader_samples /= (args.upsample_scale ** 2)
            #print("sample count", shader_samples)
            feature_w = []
            color_inds = []
            texture_inds = []
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

                    #if args.generate_timeline:
                    #    feed_samples = None

            elif args.test_tiling:
                h_start = tf.placeholder(dtype=dtype, shape=())
                #h_offset = tf.placeholder(dtype=dtype)
                w_start = tf.placeholder(dtype=dtype, shape=())
                #w_offset = tf.placeholder(dtype=dtype)
                h_offset = height / 2 + padding_offset
                w_offset = width / 2 + padding_offset
                global conv_padding
                conv_padding = "VALID"
                #feed_samples = [numpy.random.normal(size=(1, height+padding_offset, width+padding_offset)), numpy.random.normal(size=(1, height+padding_offset, width+padding_offset))]
                feed_samples = [tf.placeholder(dtype=dtype, shape=[1, height+padding_offset, width+padding_offset]), tf.placeholder(dtype=dtype, shape=[1, height+padding_offset, width+padding_offset])]
            else:
                h_start = tf.constant([0.0])
                w_start = tf.constant([0.0])
                h_offset = height
                w_offset = width
                feed_samples = None
            zero_samples = False
            #if args.render_only:
            #    zero_samples = True

            spatial_samples = None
            temporal_samples = None
            if args.render_fix_spatial_sample:
                #print("generate random fixed spatial sample")
                spatial_samples = [numpy.random.normal(size=(1, h_offset, w_offset)), numpy.random.normal(size=(1, h_offset, w_offset))]
            elif args.render_zero_spatial_sample:
                #print("generate zero spatial sample")
                spatial_samples = [numpy.zeros((1, h_offset, w_offset)), numpy.zeros((1, h_offset, w_offset))]
            if args.render_fix_temporal_sample:
                temporal_samples = [numpy.random.normal(size=(1, h_offset, w_offset))]

            target_idx = []
            target_idx_file = None
            if args.random_target_channel:
                target_idx_file = os.path.join(args.name, 'target_idx.npy')
                if os.path.exists(target_idx_file):
                    target_idx = numpy.load(target_idx_file)
                else:
                    target_idx = numpy.random.choice(args.input_nc, args.sparsity_target_channel, replace=False).astype('i')
                    numpy.save(target_idx_file, target_idx)

            if args.train_with_zero_samples:
                zero_samples = True

            

            samples_int = [None]

            if args.texture_maps != '':
                combined_texture_maps = np.load(args.texture_maps)
                texture_maps = []
                for i in range(combined_texture_maps.shape[0]):
                    texture_maps.append(tf.convert_to_tensor(combined_texture_maps[i], dtype=dtype))
            else:
                texture_maps = []


            input_to_network = get_tensors(args.dataroot, args.name, camera_pos, shader_time, output_type, shader_samples, shader_name=args.shader_name, geometry=args.geometry, learn_scale=args.learn_scale, soft_scale=args.soft_scale, scale_ratio=args.scale_ratio, use_sigmoid=args.use_sigmoid, feature_w=feature_w, color_inds=color_inds, intersection=args.intersection, sigmoid_scaling=args.sigmoid_scaling, manual_features_only=args.manual_features_only, aux_plus_manual_features=args.aux_plus_manual_features, efficient_trace=args.efficient_trace, collect_loop_statistic=args.collect_loop_statistic, h_start=h_start, h_offset=h_offset, w_start=w_start, w_offset=w_offset, samples=feed_samples, fov=args.fov, camera_pos_velocity=camera_pos_velocity, first_last_only=args.first_last_only, last_only=args.last_only, subsample_loops=args.subsample_loops, last_n=args.last_n, first_n=args.first_n, first_n_no_last=args.first_n_no_last, mean_var_only=args.mean_var_only, zero_samples=zero_samples, render_fix_spatial_sample=args.render_fix_spatial_sample, render_fix_temporal_sample=args.render_fix_temporal_sample, render_zero_spatial_sample=args.render_zero_spatial_sample, spatial_samples=spatial_samples, temporal_samples=temporal_samples, every_nth=args.every_nth, every_nth_stratified=args.every_nth_stratified, one_hop_parent=args.one_hop_parent, target_idx=target_idx, use_manual_index=args.use_manual_index, manual_index_file=args.manual_index_file, additional_features=args.additional_features, ignore_last_n_scale=args.ignore_last_n_scale, include_noise_feature=args.include_noise_feature, crop_h=args.crop_h, crop_w=args.crop_w, no_noise_feature=args.no_noise_feature, relax_clipping=args.relax_clipping, render_sigma=render_sigma, same_sample_all_pix=args.same_sample_all_pix, stratified_sample_higher_res=args.stratified_sample_higher_res, samples_int=samples_int, texture_maps=texture_maps, partial_trace=args.partial_trace, use_lstm=args.use_lstm, lstm_nfeatures_per_group=args.lstm_nfeatures_per_group, rotate=rotate, flip=flip, use_dataroot=args.use_dataroot, automatic_subsample=args.automatic_subsample, automate_raymarching_def=args.automate_raymarching_def, chron_order=args.chron_order, def_loop_log_last=args.def_loop_log_last, temporal_texture_buffer=args.temporal_texture_buffer, texture_inds=texture_inds)
            
            if args.feature_size_only:
                print('feature size: ', int(input_to_network.shape[3]))
                return

            if args.stratified_sample_higher_res:
                slicing = samples_int[0]
                assert slicing is not None
                if (args.tiled_training or args.tile_only):
                    slicing = tf.slice(slicing, [0, padding_offset // 2, padding_offset // 2, 0], [args.batch_size, args.tiled_h, args.tiled_w, 2])
                    slicing -= padding_offset
                    batch_dim = np.empty((args.batch_size, args.tiled_h, args.tiled_w, 1), dtype=np.int32)
                else:
                    batch_dim = np.empty((args.batch_size, args.input_h, args.input_w, 1), dtype=np.int32)
                for batch_id in range(args.batch_size):
                    batch_dim[batch_id] = batch_id
                slicing = tf.concat((tf.convert_to_tensor(batch_dim), slicing), 3)
                output = tf.gather_nd(output_pl, slicing)
            else:
                output = output_pl

            if args.random_target_channel:
                numpy.save(target_idx_file, target_idx)
                print("random target channel")
                print(target_idx)

            if (args.tiled_training or args.tile_only) and args.mean_estimator:
                input_to_network = tf.slice(input_to_network, [0, padding_offset // 2, padding_offset // 2, 0], [args.estimator_samples, output_pl_h, output_pl_w, 3])
            elif args.additional_input:
                additional_input = tf.pad(additional_input_pl, [[0, 0], [padding_offset // 2, padding_offset // 2], [padding_offset // 2, padding_offset // 2], [0, 0]], "SYMMETRIC")
                input_to_network = tf.concat((input_to_network, additional_input), axis=3)

            args.input_nc = int(input_to_network.shape[-1])
            color_inds = color_inds[::-1]
            debug_input = input_to_network
            
    def feature_reduction_layer(input_to_network, _manual_regularize=False, _replace_normalize_weights=None):
        with tf.variable_scope("feature_reduction"):
            r_loss = tf.constant(0.0)
            if (args.regularizer_scale > 0 or args.L2_regularizer_scale > 0) and not _manual_regularize:
                regularizer = slim.l1_l2_regularizer(scale_l1=args.regularizer_scale, scale_l2=args.L2_regularizer_scale)
            else:
                regularizer = None

            actual_nfeatures = args.input_nc

            weights = tf.get_variable('w0', [1, 1, actual_nfeatures, args.initial_layer_channels], initializer=tf.contrib.layers.xavier_initializer() if not args.identity_initialize else identity_initializer(color_inds), regularizer=regularizer)
            if args.normalize_weights:
                if args.abs_normalize:
                    column_sum = tf.reduce_sum(tf.abs(weights), [0, 1, 2])
                elif args.rowwise_L2_normalize:
                    column_sum = tf.reduce_sum(tf.abs(tf.square(weights)), [0, 1, 2])
                elif args.Frobenius_normalize:
                    column_sum = tf.reduce_sum(tf.abs(tf.square(weights)))
                else:
                    column_sum = tf.reduce_sum(weights, [0, 1, 2])
                weights_to_input = weights / column_sum
                if args.clip_weights_percentage_after_normalize:
                    weights_to_input = tf.cond(_replace_normalize_weights, lambda: normalize_weights, lambda: weights_to_input)
            else:
                weights_to_input = weights

            reduced_feat = tf.nn.conv2d(input_to_network, weights_to_input, [1, 1, 1, 1], "SAME")
            if _manual_regularize:
                r_loss = args.regularizer_scale * tf.reduce_mean(tf.abs(weights_to_input))
            if args.initial_layer_channels <= actual_conv_channel:
                ini_id = True
            else:
                ini_id = False

            if args.feature_reduction_regularization_scale > 0:
                r_loss = args.feature_reduction_regularization_scale * tf.reduce_mean(tf.abs(weights_to_input))
            
        return reduced_feat, r_loss

    #with tf.control_dependencies([input_to_network]):
    with tf.variable_scope("generator"):
        if args.debug_mode and args.mean_estimator:
            with tf.variable_scope("shader"):
                network = tf.reduce_mean(input_to_network, axis=0, keep_dims=True)
                if not args.full_resolution:
                    network = tf.image.resize_images(network, tf.stack([tf.shape(input_to_network)[1] * args.upsample_scale, tf.shape(input_to_network)[2] * args.upsample_scale]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR if not args.bilinear_upsampling else tf.image.ResizeMethod.BILINEAR)
            regularizer_loss = 0
            sparsity_loss = 0
            sparsity_schedule = None
            target_channel_schedule = None
        elif not args.unet:
            if args.input_nc <= actual_conv_channel:
                ini_id = True
            else:
                ini_id = False
            orig_channel = None
            alpha = tf.placeholder(tf.float32)
            alpha_val = 1.0

            if args.finetune:
                if args.orig_channel != '':
                    orig_channel = args.orig_channel.split(',')
                    orig_channel = [int(i) for i in orig_channel]
                    input_stacks = []
                    for i in range(args.input_nc):
                        ind = i % nfeatures
                        if ind in orig_channel:
                            input_stacks.append(input[:, :, :, i])
                        else:
                            input_stacks.append(input[:, :, :, i] * alpha)
                    input_to_network = tf.stack(input_stacks, axis=3)
                elif args.orig_rgb and args.data_from_gpu:
                    orig_channel = color_inds

            if args.clip_weights_percentage_after_normalize > 0.0:
                assert args.encourage_sparse_features
                assert not args.is_train

                replace_normalize_weights = tf.placeholder(tf.bool)
                normalize_weights = tf.placeholder(tf.float32,shape=[1, 1, args.input_nc, actual_conv_channel])
            else:
                replace_normalize_weights = None
                normalize_weights = None

            sparsity_loss = tf.constant(0.0, dtype=dtype)
            sparsity_schedule = None
            target_channel_schedule = None
            if args.feature_sparsity_vec:
                target_channel = args.sparsity_target_channel
                target_channel_max = target_channel
                if args.input_nc > target_channel_max:
                #if True:
                    if args.use_sparsity_pl:
                        target_channel = tf.placeholder(dtype=tf.int32, shape=())
                        target_channel_schedule = numpy.linspace(args.input_nc, target_channel_max, args.epoch - args.sparsity_start_freeze - args.sparsity_end_freeze, dtype=int)
                        target_channel_start_vals = args.input_nc * numpy.ones(args.sparsity_start_freeze, dtype=int)
                        target_channel_end_vals = target_channel_max * numpy.ones(args.sparsity_end_freeze, dtype=int)
                        print("target channel schedules")
                        print(target_channel_start_vals.shape, target_channel_schedule.shape, target_channel_end_vals.shape)
                        target_channel_schedule = numpy.concatenate((target_channel_start_vals, target_channel_schedule, target_channel_end_vals))
                    sparsity_vec = tf.Variable(numpy.ones(args.input_nc), dtype=dtype)
                    input_to_network = input_to_network * sparsity_vec
                    sparsity_abs = tf.abs(sparsity_vec)
                    residual_weights, residual_idx = tf.nn.top_k(-sparsity_abs, tf.maximum((args.input_nc - target_channel), 1))
                    sparsity_loss = -tf.reduce_mean(residual_weights)
                    sparsity_loss = tf.where(tf.equal(target_channel, args.input_nc), 0.0, sparsity_loss)
                    if not args.feature_sparsity_schedule:
                        sparsity_loss *= args.feature_sparsity_scale
                    else:
                        sparsity_scale = tf.placeholder(dtype=dtype, shape=())
                        sparsity_loss *= sparsity_scale
                        if not args.logscale_schedule:
                            print("using linear scale")
                            sparsity_schedule = numpy.linspace(args.feature_sparsity_schedule_start, args.feature_sparsity_schedule_end, args.epoch)
                        else:
                            print("using logscale")
                            if args.feature_sparsity_schedule_start <= 0:
                                log_start = -4
                            else:
                                log_start = numpy.log(args.feature_sparsity_schedule_start)
                            log_end = numpy.log(args.feature_sparsity_schedule_end)
                            if (not args.schedule_scale_after_freeze) or (not args.use_sparsity_pl):
                                sparsity_schedule = numpy.logspace(log_start, log_end, args.epoch, base=numpy.e)
                            else:
                                sparsity_schedule_end = numpy.logspace(log_start, log_end, args.sparsity_end_freeze)
                                sparsity_schedule = numpy.concatenate((args.feature_sparsity_schedule_start * numpy.ones(args.epoch - args.sparsity_end_freeze), sparsity_schedule_end))
                        print(sparsity_schedule)

            if args.analyze_channel:
            #if True:
                #sparsity_vec = tf.Variable(numpy.ones(args.input_nc), dtype=dtype, trainable=False)
                sparsity_vec = tf.ones(args.input_nc, dtype=dtype)
                input_to_network = input_to_network * sparsity_vec

            regularizer_loss = tf.constant(0.0)
            manual_regularize = args.rowwise_L2_normalize or args.Frobenius_normalize
            if args.encourage_sparse_features:
                actual_initial_layer_channels = args.initial_layer_channels
                regularizer = None
                if not args.use_lstm:
                    input_to_network, regularizer_loss = feature_reduction_layer(input_to_network, _manual_regularize=manual_regularize, _replace_normalize_weights=replace_normalize_weights)
                    if False:
                        with tf.variable_scope("feature_reduction"):
                            if (args.regularizer_scale > 0 or args.L2_regularizer_scale > 0) and not manual_regularize:
                                regularizer = slim.l1_l2_regularizer(scale_l1=args.regularizer_scale, scale_l2=args.L2_regularizer_scale)

                            actual_nfeatures = args.input_nc

                            weights = tf.get_variable('w0', [1, 1, actual_nfeatures, actual_initial_layer_channels], initializer=tf.contrib.layers.xavier_initializer() if not args.identity_initialize else identity_initializer(color_inds), regularizer=regularizer)
                            if args.normalize_weights:
                                if args.abs_normalize:
                                    column_sum = tf.reduce_sum(tf.abs(weights), [0, 1, 2])
                                elif args.rowwise_L2_normalize:
                                    column_sum = tf.reduce_sum(tf.abs(tf.square(weights)), [0, 1, 2])
                                elif args.Frobenius_normalize:
                                    column_sum = tf.reduce_sum(tf.abs(tf.square(weights)))
                                else:
                                    column_sum = tf.reduce_sum(weights, [0, 1, 2])
                                weights_to_input = weights / column_sum
                                if args.clip_weights_percentage_after_normalize:
                                    weights_to_input = tf.cond(replace_normalize_weights, lambda: normalize_weights, lambda: weights_to_input)
                            else:
                                weights_to_input = weights

                            input_to_network = tf.nn.conv2d(input_to_network, weights_to_input, [1, 1, 1, 1], "SAME")
                            if manual_regularize:
                                regularizer_loss = args.regularizer_scale * tf.reduce_mean(tf.abs(weights_to_input))
                            if actual_initial_layer_channels <= actual_conv_channel:
                                ini_id = True
                            else:
                                ini_id = False

                            if args.feature_reduction_regularization_scale > 0:
                                regularizer_loss = args.feature_reduction_regularization_scale * tf.reduce_mean(tf.abs(weights_to_input))
                            
                        
                else:
                    ngroups = input_to_network.shape[-1] // args.lstm_nfeatures_per_group
                    orig_shape = input_to_network.shape
                    input_to_network = tf.reshape(input_to_network, [-1, ngroups, args.lstm_nfeatures_per_group])
                    rnn_cell = tf.nn.rnn_cell.LSTMCell(args.initial_layer_channels)
                    #layer = tf.keras.layers.RNN(rnn_cell)
                    #input_to_network = layer(input_to_network)
                    _, states = tf.nn.dynamic_rnn(rnn_cell, input_to_network, dtype=dtype)
                    input_to_network = tf.reshape(states.h, [orig_shape[0], orig_shape[1], orig_shape[2], args.initial_layer_channels])
                    
                reduced_dim_feature = input_to_network

                if args.add_initial_layers:
                    for nlayer in range(3):
                        input_to_network = slim.conv2d(input_to_network, actual_initial_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(), scope='initial_'+str(nlayer), weights_regularizer=regularizer, padding=conv_padding)

            if deconv_layers > 0:
                if args.deconv:
                    regularizer = None
                    if not no_L1_reg_other_layers and args.regularizer_scale > 0.0:
                        regularizer = slim.l1_regularizer(args.regularizer_scale)
                    out_feature = args.input_nc if not args.encourage_sparse_features else actual_conv_channel
                    if not args.upsample_single:
                        if args.upsample_shrink_feature:
                            assert not args.encourage_sparse_features
                            out_feature = min(args.input_nc, actual_conv_channel)
                            ini_id = True
                        for i in range(deconv_layers):
                            input_to_network = slim.conv2d_transpose(input_to_network, out_feature, 3, stride=2, weights_initializer=identity_initializer(), scope='deconv'+str(i+1), weights_regularizer=regularizer)
                    else:
                        upsample_stacks = []
                        for c in range(out_feature):
                            current_channel = tf.expand_dims(input_to_network[:, :, :, c], axis=3)
                            for i in range(deconv_layers):
                                current_channel = slim.conv2d_transpose(current_channel, 1, 3, stride=2, weights_initializer=identity_initializer(), scope='deconv'+str(c+1)+str(i+1), weights_regularizer=regularizer)
                            upsample_stacks.append(tf.squeeze(current_channel, axis=3))
                        input_to_network = tf.stack(upsample_stacks, axis=3)
                else:
                    input_to_network = tf.image.resize_images(input_to_network, tf.stack([tf.shape(input_to_network)[1] * args.upsample_scale, tf.shape(input_to_network)[2] * args.upsample_scale]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            
            
            network=build(input_to_network, ini_id, regularizer_scale=args.regularizer_scale, share_weights=args.share_weights, final_layer_channels=args.final_layer_channels, identity_initialize=args.identity_initialize, output_nc=output_nc)

            if args.share_weights:
                assert not args.use_batch
                #loss = tf.reduce_mean(tf.square(tf.reduce_mean(network, 0) - tf.squeeze(output)))
        else:
            with tf.variable_scope("unet"):
                input_to_network = tf.image.resize_images(input_to_network, tf.stack([tf.shape(input_to_network)[1] * args.upsample_scale, tf.shape(input_to_network)[2] * args.upsample_scale]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                network = unet(input_to_network, args.unet_base_channel, args.update_bn, batch_norm_is_training)
                regularizer_loss = 0
                manual_regularize = 0
                alpha = tf.placeholder(tf.float32)
                alpha_val = 1.0

    weight_map = tf.placeholder(tf.float32,shape=[None,None,None])

    if (not args.train_res) or (args.debug_mode and args.mean_estimator):
        diff = network - output
    else:
        input_color = tf.stack([debug_input[:, :, :, ind] for ind in color_inds], axis=3)
        diff = network + input_color - output
        network += input_color

    if args.RGB_norm % 2 != 0:
        diff = tf.abs(diff)
    powered_diff = diff ** args.RGB_norm

    if args.l2_loss:
        if not args.use_weight_map:
            loss = tf.reduce_mean(powered_diff)
        else:
            loss_map = tf.reduce_mean(powered_diff, axis=3)
            if args.weight_map_add:
                loss = tf.reduce_mean(powered_diff) + tf.reduce_mean(loss_map * weight_map)
            else:
                loss = tf.reduce_mean(loss_map * weight_map)
    else:
        loss = tf.constant(0.0, dtype=dtype)

    loss_l2 = loss
    loss_add_term = loss
    
    if args.gradient_loss:
        canny_edge = tf.placeholder(tf.float32, shape=[None, None, None])
        if not args.grayscale_grad:
            dx_ground = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            dy_ground = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            dx_network, dy_network = image_gradients(network)
        else:
            dx_ground = tf.placeholder(tf.float32, shape=[None, None, None, 1])
            dy_ground = tf.placeholder(tf.float32, shape=[None, None, None, 1])
            bgr_weights = [0.0721, 0.7154, 0.2125]
            network_gray = tf.expand_dims(tf.tensordot(network, bgr_weights, [[-1], [-1]]), axis=3)
            dx_network, dy_network = image_gradients(network_gray)
        if args.normalize_grad:
            grad_norm_network = tf.sqrt(tf.square(dx_network) + tf.square(dy_network) + 1e-8)
            grad_norm_ground = tf.sqrt(tf.square(dx_ground) + tf.square(dy_ground) + 1e-8)
            dx_ground /= grad_norm_ground
            dy_ground /= grad_norm_ground
            dx_network /= grad_norm_network
            dy_network /= grad_norm_network
        if not args.cos_sim:
            gradient_loss_term = tf.reduce_mean(tf.square(dx_network - dx_ground) + tf.square(dy_network - dy_ground), axis=3)
        else:
            gradient_loss_term = -tf.reduce_mean(dx_network * dx_ground + dy_network * dy_ground, axis=3)

        if args.gradient_loss_all_pix:
            loss_add_term = tf.reduce_mean(gradient_loss_term)
        elif args.gradient_loss_canny_weight:
            loss_add_term = tf.reduce_mean(gradient_loss_term * weight_map)
        else:
            loss_add_term = tf.reduce_sum(gradient_loss_term * canny_edge) / tf.maximum(tf.reduce_sum(canny_edge), 1.0)

        loss += args.gradient_loss_scale * loss_add_term

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
        if output_nc == 3:
            loss_lpips = lpips_tf.lpips(network, output, model='net-lin', net=args.lpips_net)
        else:
            loss_lpips = 0
            assert output_nc % 3 == 0
            for i in range(output_nc // 3):
                loss_lpips += lpips_tf.lpips(network[:, :, :, 3*i:3*i+3], output[:, :, :, 3*i:3*i+3], model='net-lin', net=args.lpips_net)
        perceptual_loss_add = args.lpips_loss_scale * loss_lpips
        if args.batch_size > 1:
            perceptual_loss_add = tf.reduce_mean(perceptual_loss_add)
        loss += perceptual_loss_add
    else:
        perceptual_loss_add = tf.constant(0)
    
    
    if (args.debug_mode and args.mean_estimator):
        loss_to_opt = loss + regularizer_loss + sparsity_loss
        gen_loss_GAN = tf.constant(0.0)
        discrim_loss = tf.constant(0.0)
        savers = []
        save_names = []
    elif args.patch_gan_loss:
        # descriminator adapted from
        # https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
        def create_discriminator(discrim_inputs, discrim_target, sliced_feat=None, other_target=None):
            n_layers = args.discrim_nlayers
            layers = []

            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            if (not args.discrim_use_trace) and (not args.discrim_paired_input):
                concat_list = [discrim_inputs, discrim_target]
            elif args.discrim_use_trace:
                concat_list = [discrim_inputs, sliced_feat]
                if args.discrim_paired_input:
                    concat_list += [discrim_target, other_target]
                else:
                    concat_list += [discrim_target]
            else:
                concat_list = [discrim_target, other_target]
            
            input = tf.concat(concat_list, axis=3)
            d_network = input
            #n_ch = 32
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
        
        
        if args.is_train:
            input_color = tf.stack([debug_input[:, :, :, ind] for ind in color_inds], axis=3)
            
            if args.tiled_training or args.tile_only:
                # previously using first 3 channels of debug_input, is it a bug?
                condition_input = tf.slice(input_color, [0, padding_offset // 2, padding_offset // 2, 0], [args.batch_size, output_pl_h, output_pl_w, 3])
            else:
                condition_input = input_color
            
            if args.discrim_use_trace:
                if args.discrim_trace_shared_weights:
                    sliced_feat = tf.slice(reduced_dim_feature, [0, padding_offset // 2, padding_offset // 2, 0], [args.batch_size, output_pl_h, output_pl_w, args.conv_channel_no])
                else:
                    # this may lead to OOM
                    with tf.name_scope("discriminator_feature_reduction"):
                        discrim_feat, _ = feature_reduction_layer(debug_input)
                        sliced_feat = tf.slice(discrim_feat, [0, padding_offset // 2, padding_offset // 2, 0], [args.batch_size, output_pl_h, output_pl_w, args.conv_channel_no])
            else:
                sliced_feat = None

            with tf.name_scope("discriminator_real"):
                with tf.variable_scope("discriminator"):
                    predict_real = create_discriminator(condition_input, output, sliced_feat, network)

            with tf.name_scope("discriminator_fake"):
                with tf.variable_scope("discriminator", reuse=True):
                    predict_fake = create_discriminator(condition_input, network, sliced_feat, output)
                    
            if args.temporal_texture_buffer:
                with tf.name_scope("discriminator_fake_still"):
                    with tf.variable_scope("discriminator", reuse=True):
                        # another discriminator loss saying that still images are false
                        # use first 3 channels because at some point earlier in the code, color_inds are reversed
                        predict_fake_still = create_discriminator(condition_input, tf.tile(condition_input[:, :, :, :3], (1, 1, 1, 10)), sliced_feat, output)
            
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
            
            if args.gan_loss_style == 'cross_entropy':
                gan_loss_func =  cross_entropy_gan
            elif args.gan_loss_style == 'wgan':
                gan_loss_func = wgan
            else:
                raise
            
            with tf.name_scope("discriminator_loss"):
                #loss_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_real, labels=tf.ones_like(predict_real))
                #loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_fake, labels=tf.zeros_like(predict_fake))
                loss_real = gan_loss_func(predict_real, True)
                loss_fake = gan_loss_func(predict_fake, False)
                if args.temporal_texture_buffer:
                    loss_fake = loss_fake + gan_loss_func(predict_fake_still, False)
                discrim_loss = tf.reduce_mean(loss_real + loss_fake)
                
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
                gen_loss_GAN = gan_loss_func(predict_fake, True)
                gen_loss_GAN = tf.reduce_mean(gen_loss_GAN)
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
        
        gen_saver = tf.train.Saver(gen_tvars, max_to_keep=1000)
        
        if args.is_train:
            savers = [gen_saver, discrim_saver]
            save_names = ['model_gen', 'model_discrim']
        else:
            savers = [gen_saver]
            save_names = ['model_gen']
            
        loss_to_opt = loss

    else:
        loss_to_opt = loss + regularizer_loss + sparsity_loss
        gen_loss_GAN = tf.constant(0.0)
        discrim_loss = tf.constant(0.0)

        adam_optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        var_list = tf.trainable_variables()
        if args.two_stage_training and args.new_minimizer:
            adam_before = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        else:
            adam_before = adam_optimizer
        if args.update_bn:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                opt=adam_optimizer.minimize(loss_to_opt,var_list=var_list)
                if args.two_stage_training:
                    opt_before = adam_before.minimize(loss_l2, var_list=var_list)
        else:
            opt=adam_optimizer.minimize(loss_to_opt,var_list=var_list)
            if args.two_stage_training:
                opt_before = adam_before.minimize(loss_l2, var_list=var_list)
        saver=tf.train.Saver(tf.trainable_variables(), max_to_keep=1000)
        savers = [saver]
        save_names = ['.']
                    
        
    avg_loss = 0
    tf.summary.scalar('avg_loss', avg_loss)

    avg_loss_l2 = 0
    tf.summary.scalar('avg_loss_l2', avg_loss_l2)
    avg_loss_sparsity = 0
    tf.summary.scalar('avg_loss_sparsity', avg_loss_sparsity)
    avg_loss_regularization = 0
    tf.summary.scalar('avg_loss_regularization', avg_loss_regularization)

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
    gradient_loss = 0
    tf.summary.scalar('gradient_loss', gradient_loss)
    l2_loss = 0
    tf.summary.scalar('l2_loss', l2_loss)
    perceptual_loss = 0
    tf.summary.scalar('perceptual_loss', perceptual_loss)

    #print("start sess")
    #sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=10, intra_op_parallelism_threads=3))
    sess = tf.Session()
    #print("after start sess")
    merged = tf.summary.merge_all()

    #print("initialize local vars")
    sess.run(tf.local_variables_initializer())
    #print("initialize global vars")
    sess.run(tf.global_variables_initializer())

    if args.encourage_sparse_features:
        exclude_prefix = 'feature_reduction'
    elif args.add_initial_layers:
        exclude_prefix = 'initial_0'
    else:
        exclude_prefix = 'g_conv1'
    var_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    var_first_layer_only = [var for var in var_all if var.name.startswith(exclude_prefix)]

    if args.use_queue and read_data_from_file:
        #print("start coord")
        coord = tf.train.Coordinator()
        #print("start queue")
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        #print("start sleep")
        time.sleep(30)
        #print("after sleep")

    read_from_epoch = False

    if not (args.debug_mode and args.mean_estimator):
        read_from_epoch = True
        ckpts = [None] * len(savers)
        for c_i in range(len(savers)):
            ckpts[c_i] = tf.train.get_checkpoint_state(os.path.join(args.name, "%04d"%int(args.which_epoch), save_names[c_i]))
        if None in ckpts:
            ckpts = [None] * len(savers)
            for c_i in range(len(savers)):
                ckpts[c_i] = tf.train.get_checkpoint_state(os.path.join(args.name, save_names[c_i]))
            read_from_epoch = False

        if None not in ckpts:
            for c_i in range(len(ckpts)):
                ckpt = ckpts[c_i]
                print('loaded '+ ckpt.model_checkpoint_path)
                savers[c_i].restore(sess, ckpt.model_checkpoint_path)
            print('finished loading')
        elif args.finetune:
            ckpt_origs = [None] * len(savers)
            for c_i in range(len(savers)):
                if args.finetune_epoch > 0:
                    ckpt_origs[c_i] = tf.train.get_checkpoint_state(os.path.join(args.orig_name, "%04d"%int(args.finetune_epoch), save_names[c_i]))
                else:
                    ckpt_origs[c_i] = tf.train.get_checkpoint_state(os.path.join(args.orig_name, save_names[c_i]))
            if None not in ckpt_origs:
                if args.learn_scale:
                    #var_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    var_restore = [var for var in var_all if ('feature_scale' not in var.name) and ('feature_bias' not in var.name)]
                    saver_orig = tf.train.Saver(var_restore)
                    saver_orig.restore(sess, ckpt_orig.model_checkpoint_path)
                else:
                    # from scope g_conv2 everything should be the same
                    # only g_conv1 is different
                    #var_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                    #var_gconv1_exclude = [var for var in var_all if not var.name.startswith('g_conv1')]
                    #var_gconv1_only = [var for var in var_all if var.name.startswith('g_conv1')]
                    try:
                        for c_i in range(len(ckpt_origs)):
                            savers[c_i].restore(sess, ckpt_origs[c_i].model_checkpoint_path)
                            print('loaded '+ckpt_origs[c_i].model_checkpoint_path)
                    except:
                        var_exclude_first_layer = [var for var in var_all if not var.name.startswith(exclude_prefix)]
                        orig_saver = tf.train.Saver(var_list=var_exclude_first_layer)
                        orig_saver.restore(sess, ckpt_orig.model_checkpoint_path)
                        first_layer_dict = load_obj("%s/first_layer.pkl"%(args.orig_name))
                        assert len(first_layer_dict) == len(var_first_layer_only)
                        for var in var_first_layer_only:
                            orig_val = first_layer_dict[var.name]
                            if list(orig_val.shape) == var.get_shape().as_list():
                                sess.run(tf.assign(var, orig_val))
                            else:
                                var_shape = var.get_shape().as_list()
                                assert len(orig_val.shape) == len(var_shape)
                                assert len(orig_channel) == orig_val.shape[2]
                                current_init_val = sess.run(var)
                                for c in range(len(orig_channel)):
                                    for n in range(args.nsamples):
                                        current_init_val[:, :, orig_channel[c] + n * nfeatures, :] = orig_val[:, :, c, :] / args.nsamples
                                sess.run(tf.assign(var, current_init_val))

    save_frequency = 1
    num_epoch = args.epoch
    #assert num_epoch % save_frequency == 0

    if args.use_dataroot:
        if args.is_train or args.test_training:
            camera_pos_vals = np.load(os.path.join(args.dataroot, 'train.npy'))
            time_vals = np.load(os.path.join(args.dataroot, 'train_time.npy'))
            if args.motion_blur:
                camera_pos_velocity_vals = np.load(os.path.join(args.dataroot, 'train_velocity.npy'))
            if args.tile_only:
                tile_start_vals = np.load(os.path.join(args.dataroot, 'train_start.npy'))
        else:
            if (not args.motion_blur) and (not args.temporal_texture_buffer):
                camera_pos_vals = np.concatenate((
                                    np.load(os.path.join(args.dataroot, 'test_close.npy')),
                                    np.load(os.path.join(args.dataroot, 'test_far.npy')),
                                    np.load(os.path.join(args.dataroot, 'test_middle.npy'))
                                    ), axis=0)
            else:
                camera_pos_vals = np.load(os.path.join(args.dataroot, 'test.npy'))
            time_vals = np.load(os.path.join(args.dataroot, 'test_time.npy'))
            if args.motion_blur:
                camera_pos_velocity_vals = np.load(os.path.join(args.dataroot, 'test_velocity.npy'))
    else:
        if len(args.camera_pos_file):
            camera_pos_vals = np.load(args.camera_pos_file)[:args.camera_pos_len]
        else:
            camera_pos_vals = np.random.random([args.camera_pos_len, 6])
        time_vals = np.zeros(camera_pos_vals.shape[0])

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
            #if not args.debug_mode:
            if True:
                return ans
            else:
                all_ind = np.empty(0)
                base_ind = np.array([2, 5, 10, 11, 12, 14, 22, 23, 124, 125, 129, 130, 138, 146, 147, 151, 153, 154, 158, 159, 164, 172, 174, 225, 226, 250, 253, 255, 258, 259, 260]).astype('i')
                for k in range(4):
                    all_ind = np.concatenate((all_ind, base_ind + k * 261))
                all_ind = np.concatenate((all_ind, np.array([1044, 1045, 1046]).astype('i')))
                return ans[:, :, all_ind.astype('i')]

            #return np.load(name)
        else:
            return np.fromfile(name, dtype=np.float32).reshape([640, 960, args.input_nc])

    if read_data_from_file and args.preload and (not args.use_queue):
        eval_images = [None] * len(val_names)
        eval_out_images = [None] * len(val_names)
        for id in range(len(val_names)):
            read_ind(eval_images, val_names, id, args.is_npy)
            eval_images[id] = np.expand_dims(eval_images[id], axis=0)
            read_ind(eval_out_images, val_img_names, id, False)
            eval_out_images[id] = np.expand_dims(eval_out_images[id], axis=0)

    if args.preload and args.is_train:
        output_images = np.empty([camera_pos_vals.shape[0], output_pl.shape[1].value, output_pl.shape[2].value, 3])
        all_grads = [None] * camera_pos_vals.shape[0]
        all_adds = np.empty([camera_pos_vals.shape[0], output_pl.shape[1].value, output_pl.shape[2].value, 1])
        for id in range(camera_pos_vals.shape[0]):
            output_images[id, :, :, :] = read_name(output_names[id], False)
            if args.gradient_loss:
                all_grads[id] = read_name(grad_names[id], True)
            print(id)
            if args.additional_input:
                all_adds[id, :, :, 0] = read_name(add_names[id], True)

    #if False:
    if args.analyze_channel:
        g_channel = tf.abs(tf.gradients(loss_l2, sparsity_vec, stop_gradients=tf.trainable_variables())[0])
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
        feed_dict = {}
        current_dir = 'train' if args.test_training else 'test'
        current_dir = os.path.join(args.name, current_dir)
        if args.tile_only:
            feed_dict[h_start] = np.array([- padding_offset / 2])
            feed_dict[w_start] = np.array([- padding_offset / 2])
        if os.path.isdir(current_dir):
            render_current = False
        else:
            os.makedirs(current_dir)
            render_current = True
        for i in range(len(val_img_names)):
            print(i)
            if not args.analyze_current_only:
                output_ground = np.expand_dims(read_name(val_img_names[i], False, False), 0)
                bad_example = np.expand_dims(read_name(os.path.join(bad_dir, '%06d.png' % (i+1)), False, False), 0)
            camera_val = np.expand_dims(camera_pos_vals[i, :], axis=1)
            feed_dict[camera_pos] = camera_val
            feed_dict[shader_time] = time_vals[i:i+1]
            #if args.gradient_loss:
            if False:
                grad_arr = read_name(val_grad_names[i], True)
                feed_dict[canny_edge] = grad_arr[:, :, :, 0]
                if args.grayscale_grad:
                    feed_dict[dx_ground] = grad_arr[:, :, :, 1:2]
                    feed_dict[dy_ground] = grad_arr[:, :, :, 2:3]
                else:
                    feed_dict[dx_ground] = grad_arr[:, :, :, 1:4]
                    feed_dict[dy_ground] = grad_arr[:, :, :, 4:]
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
        if not args.analyze_current_only:
            g_good /= len(val_img_names)
            g_bad /= len(val_img_names)
            numpy.save(os.path.join(current_dir, 'g_good.npy'), g_good)
            numpy.save(os.path.join(current_dir, 'g_bad.npy'), g_bad)

        g_current /= len(val_img_names)
        numpy.save(os.path.join(current_dir, 'g_current.npy'), g_current)

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
            train_writer = tf.summary.FileWriter(args.name, sess.graph)
        # shader specific hack: if in fluid approximate mode (or temporal_texture_buffer=True), then sample number is 11 less than number of png / npy files in dataroot dir
        nexamples = time_vals.shape[0]
        if args.temporal_texture_buffer:
            nexamples -= 10
        all=np.zeros(int(nexamples * ntiles_w * ntiles_h), dtype=float)
        all_l2=np.zeros(int(nexamples * ntiles_w * ntiles_h), dtype=float)
        all_sparsity=np.zeros(int(nexamples * ntiles_w * ntiles_h), dtype=float)
        all_regularization = np.zeros(int(nexamples * ntiles_w * ntiles_h), dtype=float)
        all_training_loss = np.zeros(int(nexamples * ntiles_w * ntiles_h), dtype=float)
        all_perceptual = np.zeros(int(nexamples * ntiles_w * ntiles_h), dtype=float)
        all_gen_gan_loss = np.zeros(int(nexamples * ntiles_w * ntiles_h), dtype=float)
        all_discrim_loss = np.zeros(int(nexamples * ntiles_w * ntiles_h), dtype=float)

        occurance = np.zeros(int(nexamples * ntiles_w * ntiles_h), dtype=int)

        if read_data_from_file and args.preload and (not args.use_queue):
            input_images=[None]*len(input_names)
            output_images=[None]*len(input_names)

            for id in range(len(input_names)):
                if read_ind(input_images, input_names, id, args.is_npy):
                    read_ind(output_images, output_names, id, False)
                if not args.use_batch:
                    input_images[id] = np.expand_dims(input_images[id], axis=0)
                    output_images[id] = np.expand_dims(output_images[id], axis=0)

            if args.use_batch:
                input_images = np.array(input_images)
                output_images = np.array(output_images)
                assert input_images.dtype in allowed_dtypes
                assert output_images.dtype in allowed_dtypes

        alpha_start = -3
        alpha_end = 0
        #num_transition_epoch = 50
        num_transition_epoch = 20
        alpha_schedule = np.logspace(alpha_start, alpha_end, num_transition_epoch)
        printval = False
        
        total_step_count = 0

        min_avg_loss = 1e20

        if args.two_stage_training:
            opt_old = opt
            opt = opt_before
            loss_old = loss
            loss = loss_l2

        for epoch in range(1, num_epoch+1):

            if args.two_stage_training and epoch > num_epoch // 2:
                opt = opt_old
                loss = loss_old

            if read_from_epoch:
                if epoch <= args.which_epoch:
                    continue
            else:
                next_save_point = epoch
                # for patchGAN loss, the checkpoint in the root directory may not be very up to date (if discrim is too strong it is very possible that the model saves a best l2/perceptual error, which is a lot of epochs ago)
                if os.path.isdir("%s/%04d"%(args.name,next_save_point)) and not args.patch_gan_loss:
                    continue

            cnt=0

            if (not args.dynamic_training_samples) or (not numpy.any(all)):
                permutation = np.random.permutation(int(nexamples * ntiles_h * ntiles_w))
                nupdates = permutation.shape[0] if not args.use_batch else int(np.ceil(float(time_vals.shape[0]) / args.batch_size))
                sub_epochs = 1
                occurance += 1
            else:
                if args.dynamic_training_mode == 2:
                    sub_epochs = 4
                elif args.dynamic_training_mode == 3:
                    sub_epochs = 2
                else:
                    raise
                permutation = None
                nupdates = None

            if args.finetune and epoch <= num_transition_epoch:
                alpha_val = alpha_schedule[epoch-1]

            feed_dict={}
            if sparsity_schedule is not None:
                feed_dict[sparsity_scale] = sparsity_schedule[epoch-1]
            if target_channel_schedule is not None:
                print("using target channel schedule")
                feed_dict[target_channel] = target_channel_schedule[epoch-1]

            #for id in np.random.permutation(len(input_names)):
            for sub_epoch in range(sub_epochs):
                if (permutation is None) or (sub_epoch > 0):
                    nupdates = int(time_vals.shape[0] * ntiles_h * ntiles_w)
                    nupdates /= sub_epochs
                    nupdates = int(nupdates)
                    nupdates /= args.batch_size
                    # randomly permute indices to prevent sort always returning the same order
                    base_permutation = np.random.permutation(int(time_vals.shape[0] * ntiles_h * ntiles_w))
                    if args.dynamic_training_mode == 2:
                        if sub_epoch < 3:
                            # choose samples with highest error
                            arr_to_sort = -all[base_permutation]
                        else:
                            arr_to_sort = occurance[base_permutation]
                        sorted_idx = numpy.argsort(arr_to_sort)
                        permutation = base_permutation[sorted_idx[:nupdates]]
                    elif args.dynamic_training_mode == 3:
                        sorted_idx = numpy.argsort(-all[base_permutation])
                        sample_prob = numpy.ones(base_permutation.shape)
                        sample_prob[sorted_idx[:nupdates]] *= 2.0
                        sample_prob /= numpy.sum(sample_prob)
                        permutation = numpy.random.choice(base_permutation, nupdates, replace=False, p=sample_prob)
                    else:
                        raise
                    permutation = numpy.random.permutation(permutation)
                    print("randomized permutation")
                    occurance[permutation] += 1

                for i in range(nupdates):


                    st=time.time()
                    start_id = i * args.batch_size
                    end_id = min(permutation.shape[0], (i+1)*args.batch_size)

                    #frame_idx = int(permutation[i] // (ntiles_w * ntiles_h))
                    #tile_idx = int(permutation[i] % (ntiles_w * ntiles_h))
                    frame_idx = (permutation[start_id:end_id] / (ntiles_w * ntiles_h)).astype('i')
                    tile_idx = (permutation[start_id:end_id] % (ntiles_w * ntiles_h)).astype('i')

                    if False:
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                    else:
                        run_options = None
                        run_metadata = None

                    if args.use_weight_map or args.gradient_loss_canny_weight:
                        feed_dict[weight_map] = np.expand_dims(read_name(map_names[frame_idx], True), axis=0)
                    if args.gradient_loss:
                        if not args.preload:
                            grad_arr = read_name(grad_names[frame_idx], True)
                        else:
                            # TODO: assuming batch size 1 whenever we use gradient loss
                            grad_arr = all_grads[frame_idx[0]]
                        feed_dict[canny_edge] = grad_arr[:, :, :, 0]
                        if args.grayscale_grad:
                            feed_dict[dx_ground] = grad_arr[:, :, :, 1:2]
                            feed_dict[dy_ground] = grad_arr[:, :, :, 2:3]
                        else:
                            feed_dict[dx_ground] = grad_arr[:, :, :, 1:4]
                            feed_dict[dy_ground] = grad_arr[:, :, :, 4:]
                            
                    if args.discrim_train_steps > 1:
                        feed_dict[step_count] = total_step_count
                        total_step_count += 1

                    if (not args.use_queue) and read_data_from_file:
                        if args.preload:
                            if not args.use_batch:
                                input_image = input_images[frame_idx]
                                output_image = output_images[frame_idx]
                                if input_image is None:
                                    continue
                            else:
                                input_image = input_images[permutation[start_id:end_id], :, :, :]
                                output_image = output_images[permutation[start_id:end_id], :, :, :]
                        else:
                            if not args.use_batch:
                                input_image = np.expand_dims(read_name(input_names[frame_idx], args.is_npy, args.is_bin), axis=0)
                                output_image = np.expand_dims(read_name(output_names[frame_idx], False), axis=0)
                            else:
                                # TODO: should complete this logic
                                raise
                        feed_dict[input] = input_image
                        feed_dict[output_pl] = output_image
                        feed_dict[alpha] = alpha_val
                        #_,current, current_l2, current_sparsity, current_regularization =sess.run([opt,loss, loss_l2, sparsity_loss, regularizer_loss],feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                        _,current, current_l2, current_sparsity, current_regularization, current_training, current_perceptual = sess.run([opt,loss, loss_l2, sparsity_loss, regularizer_loss, loss_to_opt, perceptual_loss_add],feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                    elif args.use_queue:
                        if not printval:
                            #print("first time arriving before sess, wish me best luck")
                            printval = True
                        _,current=sess.run([opt,loss],feed_dict={alpha: alpha_val}, options=run_options, run_metadata=run_metadata)
                    else:
                        if not args.preload:
                            #if args.tile_only:
                            #    output_arr = np.empty([args.batch_size, args.tiled_h, args.tiled_w, 3])
                            #else:
                            #    output_arr = np.empty([args.batch_size, args.input_h, args.input_w, 3])
                            output_arr = np.empty([args.batch_size, output_pl.shape[1].value, output_pl.shape[2].value, output_nc])
                            
                            for img_idx in range(frame_idx.shape[0]):
                                if not args.temporal_texture_buffer:
                                    output_arr[img_idx, :, :, :] = read_name(output_names[frame_idx[img_idx]], False)
                                else:
                                    # a hack to read 10 frames after selected idx in fluid approx mode
                                    # e.g. if idx = 1 (texture is from idx 1)
                                    # then gt images should be idx = 2 - 11
                                    for seq_id in range(10):
                                        output_arr[img_idx, :, :, seq_id*3:seq_id*3+3] = read_name(output_names[frame_idx[img_idx]+seq_id+1], False)
                            #output_arr = np.expand_dims(read_name(output_names[frame_idx], False), axis=0)
                            if args.additional_input:
                                additional_arr = np.empty([args.batch_size, output_pl.shape[1].value, output_pl.shape[2].value, 1])
                                for img_idx in range(frame_idx.shape[0]):
                                    additional_arr[img_idx, :, :, 0] = read_name(add_names[frame_idx[img_idx]], True)
                        else:
                            output_arr = output_images[frame_idx]
                            if args.additional_input:
                                additional_arr = all_adds[frame_idx]
                        #output_arr = numpy.ones([1, args.input_h, args.input_w, 3])
                        if train_from_queue:
                            output_arr = output_arr[..., ::-1]
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

                        if args.train_with_random_rotation:
                            random_rotation = np.random.randint(0, 4)
                            feed_dict[rotate] = random_rotation
                            random_flip = np.random.randint(0, 2)
                            feed_dict[flip] = random_flip
                            if random_rotation > 0:
                                output_arr = numpy.rot90(output_arr, k=random_rotation, axes=(1, 2))
                            if random_flip > 0:
                                output_arr = output_arr[:, :, ::-1, :]
                        feed_dict[output_pl] = output_arr
                        if args.additional_input:
                            feed_dict[additional_input_pl] = additional_arr

                        if not args.temporal_texture_buffer:
                            camera_val = camera_pos_vals[frame_idx, :].transpose()
                            feed_dict[camera_pos] = camera_val
                            feed_dict[shader_time] = time_vals[frame_idx]
                        else:
                            assert args.batch_size == 1
                            camera_val = np.empty([33, 1])
                            # e.g. if idx = 1
                            # camera_pos_val[1] is last mouse for 1st frame
                            # and we collect 10 next mouse pos
                            camera_val[:, 0] = np.reshape(camera_pos_vals[frame_idx[0]:frame_idx[0]+11, 3:], 33)
                            feed_dict[camera_pos] = camera_val
                            feed_dict[shader_time] = time_vals[frame_idx]
                            current_texture_maps = np.transpose(np.load(input_names[frame_idx[0]]), (2, 0, 1))
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                if not (current_texture_maps.shape[1] == height and current_texture_maps.shape[2] == width):
                                    current_texture_maps = skimage.transform.resize(current_texture_maps, (current_texture_maps.shape[0], height, width))
                            for k in range(len(texture_maps)):
                                feed_dict[texture_maps[k]] = current_texture_maps[k]
                                
                        if args.motion_blur:
                            feed_dict[camera_pos_velocity] = camera_pos_velocity_vals[frame_idx, :]
                        # sanity check for mode stratified_sample_higher_res
                        #if args.stratified_sample_higher_res:
                        if False:
                            xv, yv = np.meshgrid(np.arange(320), np.arange(320), indexing='ij')
                            xv = np.transpose(xv)
                            yv = np.transpose(yv)
                            ans, slicing_val, subsample_val = sess.run([debug_input, slicing, output], feed_dict=feed_dict)
                            noise1 = ans[:, :, :, -2]
                            noise2 = ans[:, :, :, -1]
                            noise1 = noise1[:, 16:-16, 16:-16]
                            noise2 = noise2[:, 16:-16, 16:-16]
                            assert np.allclose((xv + noise1) * 2, slicing_val[:, :, :, -1])
                            assert np.allclose((yv + noise2 - 0.5) * 2 + 1, slicing_val[:, :, :, -2])
                            #vec_output = ans[:, :, :, color_inds]
                            #assert np.allclose(subsample_val, np.clip(vec_output, 0.0, 1.0))
                        st1 = time.time()
                        _,current, current_l2, current_sparsity, current_regularization, current_training, current_perceptual, current_gen_loss_GAN, current_discrim_loss =sess.run([opt,loss, loss_l2, sparsity_loss, regularizer_loss, loss_to_opt, perceptual_loss_add, gen_loss_GAN, discrim_loss],feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                        st2 = time.time()
                        #_ = sess.run(opt, feed_dict=feed_dict)
                        #current, current_l2, current_sparsity, current_regularization, current_training, current_perceptual = sess.run([loss, loss_l2, sparsity_loss, regularizer_loss, loss_to_opt, perceptual_loss_add], feed_dict=feed_dict)
                        #current =sess.run(loss,feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                        if numpy.isnan(current):
                            print(frame_idx, tile_idx)
                            raise
                            #arr0 = sess.run(input_to_network, feed_dict=feed_dict)
                            #print(numpy.sum(numpy.isnan(arr0)), arr0.shape)
                            #arr1 = sess.run(debug_input, feed_dict=feed_dict)
                            #print(numpy.sum(numpy.isnan(arr1)), arr1.shape)


                    if args.learn_scale:
                        feature_val = sess.run(feature_w)
                        if numpy.sum(numpy.isnan(feature_val)) > 0:
                            print(feature_val)

                    if run_metadata is not None and args.generate_timeline:
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open("%s/epoch%04d_step%d.json"%(args.name,epoch, i), 'w') as f:
                            f.write(chrome_trace)
                    #_,current=sess.run([opt,loss],feed_dict={input:input_image,output:output_image, alpha: alpha_val})
                    all[permutation[start_id:end_id]]=current
                    all_l2[permutation[start_id:end_id]]=current_l2
                    all_sparsity[permutation[start_id:end_id]]=current_sparsity
                    all_regularization[permutation[start_id:end_id]] = current_regularization
                    all_training_loss[permutation[start_id:end_id]] = current_training
                    all_perceptual[permutation[start_id:end_id]] = current_perceptual
                    all_gen_gan_loss[permutation[start_id:end_id]] = current_gen_loss_GAN
                    all_discrim_loss[permutation[start_id:end_id]] = current_discrim_loss
                    cnt += args.batch_size if args.use_batch else 1
                    print("%d %d %.5f %.5f %.2f %.2f %s"%(epoch,cnt,current,np.mean(all[np.where(all)]),time.time()-st, st2-st1,os.getcwd().split('/')[-2]))

            print(occurance)
            avg_loss = np.mean(all[np.where(all)])
            avg_loss_l2 = np.mean(all_l2[np.where(all_l2)])
            #avg_loss_sparsity = np.mean(all_sparsity[np.where(all_sparsity)])
            avg_loss_sparsity = np.mean(all_sparsity)
            avg_loss_regularization = np.mean(all_regularization)
            avg_training_loss = np.mean(all_training_loss)
            avg_perceptual = np.mean(all_perceptual)
            avg_gen_gan = np.mean(all_gen_gan_loss)
            avg_discrim = np.mean(all_discrim_loss)

            if not (args.two_stage_training and epoch <= num_epoch // 2):
                if min_avg_loss > avg_training_loss:
                    min_avg_loss = avg_training_loss

            if args.write_summary:
                summary = tf.Summary()
                summary.value.add(tag='avg_loss', simple_value=avg_loss)
                summary.value.add(tag='avg_loss_l2', simple_value=avg_loss_l2)
                summary.value.add(tag='avg_loss_sparsity', simple_value=avg_loss_sparsity)
                summary.value.add(tag='avg_loss_regularization', simple_value=avg_loss_regularization)
                summary.value.add(tag='avg_training_loss', simple_value=avg_training_loss)
                summary.value.add(tag='avg_perceptual', simple_value=avg_perceptual)
                summary.value.add(tag='avg_gen_gan', simple_value=avg_gen_gan)
                summary.value.add(tag='avg_discrim', simple_value=avg_discrim)
                if manual_regularize:
                    summary.value.add(tag='reg_loss', simple_value=sess.run(regularizer_loss) / args.regularizer_scale)

                if args.collect_validate_while_training:
                    all_test=np.zeros(len(val_names), dtype=float)
                    for ind in range(len(val_names)):
                        if not args.use_queue:
                            if args.preload:
                                input_image = eval_images[ind]
                                output_image = eval_out_images[ind]
                            else:
                                input_image = np.expand_dims(read_name(val_names[ind], args.is_npy, args.is_bin), axis=0)
                                output_image = np.expand_dims(read_name(val_names[ind], False, False), axis=0)
                            if input_image is None:
                                continue
                            st=time.time()
                            current=sess.run(loss,feed_dict={input:input_image, output: output_image, alpha: alpha_val})
                            print("%.3f"%(time.time()-st))
                        else:
                            st=time.time()
                            current=sess.run(loss,feed_dict={alpha: alpha_val})
                            print("%.3f"%(time.time()-st))
                        all_test[ind] = current

                    avg_test_close = np.mean(all_test[:5])
                    avg_test_far = np.mean(all_test[5:10])
                    avg_test_middle = np.mean(all_test[10:])
                    avg_test_all = np.mean(all_test)

                    summary.value.add(tag='avg_test_close', simple_value=avg_test_close)
                    summary.value.add(tag='avg_test_far', simple_value=avg_test_far)
                    summary.value.add(tag='avg_test_middle', simple_value=avg_test_middle)
                    summary.value.add(tag='avg_test_all', simple_value=avg_test_all)

                #train_writer.add_run_metadata(run_metadata, 'epoch%d' % epoch)
                train_writer.add_summary(summary, epoch)

            os.makedirs("%s/%04d"%(args.name,epoch))
            target=open("%s/%04d/score.txt"%(args.name,epoch),'w')
            target.write("%f"%np.mean(all[np.where(all)]))
            target.close()

            #target = open("%s/%04d/score_breakdown.txt"%(args.name,epoch),'w')
            #target.write("%f, %f, %f, %f"%(avg_test_close, avg_test_far, avg_test_middle, avg_test_all))
            #target.close()

            if min_avg_loss == avg_training_loss:
                for s_i in range(len(savers)):
                    ckpt_dir = os.path.join(args.name, save_names[s_i])
                    if not os.path.isdir(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    savers[s_i].save(sess,"%s/model.ckpt" % ckpt_dir)
            if epoch % args.save_frequency == 0:
                for s_i in range(len(savers)):
                    ckpt_dir = os.path.join("%s/%04d"%(args.name,epoch), save_names[s_i])
                    if not os.path.isdir(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    savers[s_i].save(sess,"%s/model.ckpt" % ckpt_dir)

        #var_list_gconv1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='g_conv1')
        #g_conv1_dict = {}
        #for var_gconv1 in var_list_gconv1:
        #    g_conv1_dict[var_gconv1.name] = sess.run(var_gconv1)
        #save_obj(g_conv1_dict, "%s/g_conv1.pkl"%(args.name))

    if not args.is_train:

        if args.sparsity_vec_histogram:
            sparsity_vec_vals = numpy.abs(sess.run(sparsity_vec))
            figure = pyplot.figure()
            pyplot.hist(sparsity_vec_vals, bins=10)
            pyplot.title('entries > 0.2: %d' % numpy.sum(sparsity_vec_vals > 0.2))
            figure.savefig(os.path.join(args.name, 'sparsity_hist' + ("_epoch_%04d"%args.which_epoch if read_from_epoch else '') + '.png'))
            return

        if not read_data_from_file:
            if args.use_dataroot:
                if args.render_only:
                    camera_pos_vals = np.load(args.render_camera_pos)
                    time_vals = np.load(args.render_t)
                    if args.motion_blur:
                        camera_pos_velocity_vals = np.load(args.render_camera_pos_velocity)
                elif args.test_training:
                    camera_pos_vals = np.load(os.path.join(args.dataroot, 'train.npy'))
                    time_vals = np.load(os.path.join(args.dataroot, 'train_time.npy'))
                    if args.motion_blur:
                        camera_pos_velocity_vals = np.load(os.path.join(args.dataroot, 'train_velocity.npy'))
                    if args.tile_only and (not inference_entire_img_valid):
                        tile_start_vals = np.load(os.path.join(args.dataroot, 'train_start.npy'))
                else:
                    if (not args.motion_blur) and (not args.temporal_texture_buffer):
                        camera_pos_vals = np.concatenate((
                                            np.load(os.path.join(args.dataroot, 'test_close.npy')),
                                            np.load(os.path.join(args.dataroot, 'test_far.npy')),
                                            np.load(os.path.join(args.dataroot, 'test_middle.npy'))
                                            ), axis=0)
                    else:
                        camera_pos_vals = np.load(os.path.join(args.dataroot, 'test.npy'))
                    time_vals = np.load(os.path.join(args.dataroot, 'test_time.npy'))
                    if args.motion_blur:
                        camera_pos_velocity_vals = np.load(os.path.join(args.dataroot, 'test_velocity.npy'))

            if args.render_only:
                debug_dir = args.name + '/render'
            elif args.mean_estimator:
                debug_dir = "%s/mean%d"%(args.name, args.estimator_samples)
                debug_dir += '_test' if not args.test_training else '_train'
                #debug_dir = "%s/mean%d"%('/localtmp/yuting', args.estimator_samples)
            elif args.visualize_scaling:
                debug_dir = args.name + '/scaled_features'
            else:
                #debug_dir = "%s/debug"%args.name
                debug_dir = args.name + '/' + ('test' if not args.test_training else 'train')
                if args.debug_mode:
                    debug_dir += '_debug'
                #debug_dir = "%s/debug"%'/localtmp/yuting'

            debug_dir += '_bilinear' if args.bilinear_upsampling else ''

            debug_dir += '_full' if args.full_resolution else ''

            debug_dir += '_zero_out_sparsity_vec' if args.zero_out_sparsity_vec else ''

            if read_from_epoch:
                debug_dir += "_epoch_%04d"%args.which_epoch

            if not os.path.isdir(debug_dir):
                os.makedirs(debug_dir)

            if args.render_only and os.path.exists(os.path.join(debug_dir, 'video.mp4')):
                os.remove(os.path.join(debug_dir, 'video.mp4'))

            if args.render_only:
                shutil.copyfile(args.render_camera_pos, os.path.join(debug_dir, 'camera_pos.npy'))
                shutil.copyfile(args.render_t, os.path.join(debug_dir, 'render_t.npy'))
                if args.motion_blur:
                    shutil.copyfile(args.render_camera_pos_velocity, os.path.join(debug_dir, 'camera_pos_velocity.npy'))

            python_time = numpy.zeros(time_vals.shape[0])
            nburns = 10
            timeline_time = numpy.zeros(time_vals.shape[0] - nburns)
            if args.generate_timeline:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = None
                run_metadata = None

            feed_dict = {}
            if sparsity_schedule is not None:
                if not read_from_epoch:
                    feed_dict[sparsity_scale] = sparsity_schedule[-1]
                else:
                    feed_dict[sparsity_scale] = sparsity_schedule[args.which_epoch - 1]
            if target_channel_schedule is not None:
                if not read_from_epoch:
                    feed_dict[target_channel] = target_channel_schedule[-1]
                else:
                    feed_dict[target_channel] = target_channel_schedule[args.which_epoch - 1]

            if args.feature_sparsity_vec and args.zero_out_sparsity_vec:
                #if True:
                if args.input_nc > target_channel_max:
                    # workaround due to an incorrect tensorflow dependency on placeholders...
                    #feed_dict = {camera_pos: camera_pos_vals[0, :], shader_time: time_vals[0:1]}
                    feed_dict[camera_pos] = camera_pos_vals[0:1, :].transpose()
                    feed_dict[shader_time] = time_vals[0:1]
                    # workaround for tensorflow's weird compute graph that's asking for placeholder values more than needed
                    vec_val = sess.run(sparsity_vec)
                    if isinstance(target_channel, int):
                        inference_channel = target_channel
                    else:
                        inference_channel = feed_dict[target_channel]
                    _, idx_val = sess.run(tf.nn.top_k(-tf.abs(sparsity_vec), args.input_nc - inference_channel))
                    #vec_val, idx_val = sess.run([sparsity_vec, residual_idx], feed_dict=feed_dict)
                    vec_val[idx_val] = 0.0
                    sess.run(tf.assign(sparsity_vec, vec_val))

            #builder = tf.profiler.ProfileOptionBuilder
            #opts = builder(builder.time_and_memory()).order_by('micros').build()
            #with tf.contrib.tfprof.ProfileContext('/tmp/train_dir', trace_steps=[], dump_steps=[]) as pctx:
            if not args.collect_validate_loss:
                if args.render_only:
                    feed_dict[h_start] = np.array([- padding_offset / 2])
                    feed_dict[w_start] = np.array([- padding_offset / 2])
                    if args.test_rotation or args.train_with_random_rotation:
                        feed_dict[rotate] = 0
                        feed_dict[flip] = 0
                    if args.temporal_texture_buffer:
                        init_texture = np.transpose(np.load(args.render_temporal_texture), (2, 0, 1))
                        for k in range(len(texture_maps)):
                            feed_dict[texture_maps[k]] = init_texture[k]
                        output_texture = tf.stack([debug_input[:, :, :, ind] for ind in texture_inds], axis=3)
                        
                    for i in range(time_vals.shape[0]):
                        #feed_dict = {camera_pos: camera_pos_vals[i:i+1, :].transpose(), shader_time: time_vals[i:i+1]}
                        feed_dict[camera_pos] = camera_pos_vals[i:i+1, :].transpose()
                        feed_dict[shader_time] = time_vals[i:i+1]
                        if args.additional_input:
                            feed_dict[additional_input_pl] = np.expand_dims(np.expand_dims(read_name(val_add_names[i], True), axis=2), axis=0)
                        #feed_dict[camera_pos] = camera_pos_vals[0:1, :].transpose()
                        #feed_dict[shader_time] = time_vals[0:1]
                        if args.motion_blur:
                            feed_dict[camera_pos_velocity] = camera_pos_velocity_vals[i, :]
                        if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                            nruns = args.estimator_samples
                            output_buffer = numpy.zeros((1, 640, 960, 3))
                        else:
                            nruns = 1
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
                        if output_nc == 3:
                            cv2.imwrite("%s/%06d.png"%(debug_dir, i+1),np.uint8(output_image[0,:,:,:]))
                        else:
                            assert output_nc % 3 == 0
                            for img_id in range(output_nc // 3):
                                cv2.imwrite("%s/%05d%d.png"%(debug_dir, i+1, img_id),np.uint8(output_image[0,:,:,3*img_id+3*img_id+3]))
                        print('finished', i)
                    os.system('ffmpeg -i %s -r 30 -c:v libx264 -preset slow -crf 0 -r 30 %s'%(os.path.join(debug_dir, '%06d.png'), os.path.join(debug_dir, 'video.mp4')))
                    open(os.path.join(debug_dir, 'index.html'), 'w+').write("""
<html>
<body>
<br><video controls><source src="video.mp4" type="video/mp4"></video><br>
</body>
</html>""")
                    return
                elif args.visualize_scaling:
                    #feed_dict = {camera_pos: camera_pos_vals[args.visualize_ind, :], shader_time: time_vals[args.visualize_ind:args.visualize_ind+1]}
                    feed_dict[camera_pos] = np.expand_dims(camera_pos_vals[args.visualize_ind, :], axis=1)
                    feed_dict[shader_time] = time_vals[args.visualize_ind:args.visualize_ind+1]
                    if args.motion_blur:
                        feed_dict[camera_pos_velocity] = camera_pos_velocity_vals[args.visualize_ind, :]
                    all_features = sess.run(debug_input, feed_dict=feed_dict)
                    for i in range(all_features.shape[3]):
                        cv2.imwrite("%s/%05d.png"%(debug_dir, i), all_features[0, :, :, i] * 255.0)
                    return
                else:
                    nexamples = time_vals.shape[0]
                    if args.temporal_texture_buffer:
                        # for regular test, only inference once
                        # start from a texture buffer rendered from gt
                        # then inference the next 10 frames
                        # for efficiency, do not render overlapping frames
                        nexamples -= 1
                        nexamples = nexamples // 10
                    all_test = np.zeros(nexamples, dtype=float)
                    all_grad = np.zeros(nexamples, dtype=float)
                    all_l2 = np.zeros(nexamples, dtype=float)
                    all_perceptual = np.zeros(nexamples, dtype=float)
                    for i in range(nexamples):
                        if args.test_rotation or args.train_with_random_rotation:
                            feed_dict[rotate] = 0
                            feed_dict[flip] = 0
                        camera_val = np.expand_dims(camera_pos_vals[i, :], axis=1)
                        #feed_dict = {camera_pos: camera_val, shader_time: time_vals[i:i+1]}
                        feed_dict[camera_pos] = camera_val
                        feed_dict[shader_time] = time_vals[i:i+1]
                        
                        if args.use_dataroot:
                            if not args.temporal_texture_buffer:
                                output_ground = np.expand_dims(read_name(val_img_names[i], False, False), 0)
                            else:
                                output_ground = np.empty([1, output_pl.shape[1].value, output_pl.shape[2].value, output_nc])
                                # a hack to read 10 frames after selected idx in fluid approx mode
                                # at first we only test inference on input with every 10 frames
                                # so that output frame will be non overlapping
                                # for index i, output gt is i+1 to i+10
                                for seq_id in range(10):
                                    output_ground[0, :, :, seq_id*3:seq_id*3+3] = read_name(val_img_names[i*10+seq_id+1], False)

                                camera_val = np.empty([33, 1])
                                camera_val[:, 0] = np.reshape(camera_pos_vals[i*10:i*10+11, 3:], 33)
                                feed_dict[camera_pos] = camera_val
                                feed_dict[shader_time] = time_vals[i*10:i*10+1]
                                current_texture_maps = np.transpose(np.load(val_names[i*10]), (2, 0, 1))
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    if not (current_texture_maps.shape[1] == height and current_texture_maps.shape[2] == width):
                                        current_texture_maps = skimage.transform.resize(current_texture_maps, (current_texture_maps.shape[0], height, width))
                                for k in range(len(texture_maps)):
                                    feed_dict[texture_maps[k]] = current_texture_maps[k]
                            
                        else:
                            output_ground = np.empty([1, args.input_h, args.input_w, 3])
                        #print("output_ground get")
                        
                        if args.motion_blur:
                            feed_dict[camera_pos_velocity] = camera_pos_velocity_vals[i, :]

                        if args.use_weight_map or args.gradient_loss_canny_weight:
                            feed_dict[weight_map] = np.expand_dims(read_name(val_map_names[i], True), 0)
                        if args.gradient_loss:
                            grad_arr = read_name(val_grad_names[i], True)
                            feed_dict[canny_edge] = grad_arr[:, :, :, 0]
                            if args.grayscale_grad:
                                feed_dict[dx_ground] = grad_arr[:, :, :, 1:2]
                                feed_dict[dy_ground] = grad_arr[:, :, :, 2:3]
                            else:
                                feed_dict[dx_ground] = grad_arr[:, :, :, 1:4]
                                feed_dict[dy_ground] = grad_arr[:, :, :, 4:]
                        #print("feed_dict generated")
                        #pctx.trace_next_step()
                        #pctx.dump_next_step()
                        if args.tile_only:
                            if not inference_entire_img_valid:
                                feed_dict[h_start] = tile_start_vals[i:i+1, 0] - padding_offset / 2
                                feed_dict[w_start] = tile_start_vals[i:i+1, 1] - padding_offset / 2
                            else:
                                feed_dict[h_start] = np.array([- padding_offset / 2])
                                feed_dict[w_start] = np.array([- padding_offset / 2])
                        feed_dict[output_pl] = output_ground
                        if args.additional_input:
                            feed_dict[additional_input_pl] = np.expand_dims(np.expand_dims(read_name(val_add_names[i], True), axis=2), axis=0)
                        output_buffer = numpy.zeros(output_ground.shape)

                        st = time.time()
                        if args.tiled_training:
                            st_sum = 0
                            timeline_sum = 0
                            l2_loss_val = 0
                            grad_loss_val = 0
                            perceptual_loss_val = 0
                            output_patch = numpy.zeros((1, int(height/ntiles_h), int(width/ntiles_w), 3))
                            #if not args.generate_timeline:
                            if True:
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
                                    #l2_loss_patch, grad_loss_patch = sess.run([loss_l2, loss_add_term], feed_dict=tiled_feed_dict, options=run_options, run_metadata=run_metadata)
                                    st_before = time.time()
                                    #if not args.generate_timeline:
                                    if not args.accurate_timing:
                                        output_patch, l2_loss_patch, grad_loss_patch, perceptual_patch = sess.run([network, loss_l2, loss_add_term, perceptual_loss_add], feed_dict=tiled_feed_dict, options=run_options, run_metadata=run_metadata)
                                        st_after = time.time()
                                    else:
                                        sess.run([network], feed_dict=feed_dict)
                                        st_after = time.time()
                                        output_patch, l2_loss_patch, grad_loss_patch, perceptual_patch = sess.run([network, loss_l2, loss_add_term, perceptual_loss_add], feed_dict=tiled_feed_dict, options=run_options, run_metadata=run_metadata)
                                    st_sum += (st_after - st_before)
                                    print(st_after - st_before)
                                    #l2_loss_patch, grad_loss_patch = sess.run([loss_l2, loss_add_term], feed_dict=tiled_feed_dict, options=run_options, run_metadata=run_metadata)
                                    output_buffer[0, int(tile_h*height/ntiles_h):int((tile_h+1)*height/ntiles_h), int(tile_w*width/ntiles_w):int((tile_w+1)*width/ntiles_w), :] = output_patch[0, :, :, :]
                                    l2_loss_val += l2_loss_patch
                                    grad_loss_val += grad_loss_patch
                                    perceptual_loss_val += perceptual_patch
                                    if args.generate_timeline:
                                    #if False:
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

                        elif args.test_tiling:
                            output_image = numpy.empty(output_ground.shape)
                            feed_dict[feed_samples[0]] = numpy.random.normal(size=(1, height+padding_offset, width+padding_offset))
                            feed_dict[feed_samples[1]] = numpy.random.normal(size=(1, height+padding_offset, width+padding_offset))
                            #feed_dict[feed_samples[0]] = numpy.pad(numpy.random.normal(size=(1, height, width)), ((0, 0), (padding_offset // 2, padding_offset // 2), (padding_offset // 2, padding_offset // 2)), 'constant', constant_values=0.0)
                            #feed_dict[feed_samples[1]] = numpy.pad(numpy.random.normal(size=(1, height, width)), ((0, 0), (padding_offset // 2, padding_offset // 2), (padding_offset // 2, padding_offset // 2)), 'constant', constant_values=0.0)
                            for tile_h in range(2):
                                for tile_w in range(2):
                                    feed_dict[h_start] = np.array([tile_h * height / 2 - padding_offset / 2])
                                    feed_dict[w_start] = np.array([tile_w * width / 2 - padding_offset / 2])
                                    output_tile = sess.run(network, feed_dict=feed_dict)
                                    output_image[0, int(tile_h*height/2):int((tile_h+1)*height/2), int(tile_w*width/2):int((tile_w+1)*width/2), :] = output_tile[0, :, :, :]
                            cv2.imwrite('test4.png', numpy.uint8(numpy.clip(output_image[0, :, :, :], 0.0, 1.0) * 255.0))
                            return
                        else:
                            if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                nruns = args.estimator_samples
                            else:
                                nruns = 1
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
                                if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                    output_buffer += output_image[:, :, :, ::-1]
                            st2 = time.time()
                            #print("rough time estimate:", st2 - st)
                            print("rough time estimate:", st_sum)
                            if args.test_rotation:
                                feed_dict[flip] = 1
                                feed_dict[rotate] = 1
                                img1 = sess.run(network, feed_dict=feed_dict)
                                assert np.allclose(img1, numpy.rot90(output_image, axes=(1, 2))[:, :, ::-1, :], atol=1e-6)
                                feed_dict[rotate] = 2
                                img2 = sess.run(network, feed_dict=feed_dict)
                                assert np.allclose(img2, numpy.rot90(output_image, k=2, axes=(1, 2))[:, :, ::-1, :], atol=1e-6)
                                feed_dict[rotate] = 3
                                img3 = sess.run(network, feed_dict=feed_dict)
                                assert np.allclose(img3, numpy.rot90(output_image, k=3, axes=(1, 2))[:, :, ::-1, :], atol=1e-6)
                            #pctx.profiler.profile_operations(options=opts)
                            if train_from_queue or args.mean_estimator:
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
                        #if not args.stratified_sample_higher_res:
                        loss_val = np.mean((output_image - output_ground) ** 2)
                        #print("loss", loss_val, l2_loss_val * 255.0 * 255.0)
                        all_test[i] = loss_val
                        all_l2[i] = l2_loss_val
                        all_grad[i] = grad_loss_val
                        all_perceptual[i] = perceptual_loss_val
                        output_image=np.clip(output_image,0.0,1.0)
                        #print("output_image clipped")
                        output_image *= 255.0
                        #print("output_image scaled")
                        
                        if output_nc == 3:
                            cv2.imwrite("%s/%06d.png"%(debug_dir, i+1),np.uint8(output_image[0,:,:,:]))
                        else:
                            assert output_nc % 3 == 0
                            for img_id in range(output_nc // 3):
                                cv2.imwrite("%s/%06d%d.png"%(debug_dir, i+1, img_id),np.uint8(output_image[0,:,:,3*img_id:3*img_id+3]))
                                
                        python_time[i] = st_sum
                        if args.generate_timeline:
                            timeline_time[i-nburns] = timeline_sum
                        #print("output_image written")
                    open("%s/all_loss.txt"%debug_dir, 'w').write("%f, %f"%(np.mean(all_l2), np.mean(all_grad)))
                    numpy.save(os.path.join(debug_dir, 'python_time.npy'), python_time)
                    numpy.save(os.path.join(debug_dir, 'timeline_time.npy'), timeline_time)
                    open("%s/all_time.txt"%debug_dir, 'w').write("%f, %f"%(np.median(python_time), np.median(timeline_time)))
                    print("all times saved")
            test_dirname = debug_dir

            if args.collect_validate_loss:
                #assert not args.test_training
                dirs = sorted(os.listdir(args.name))
                train_writer = tf.summary.FileWriter(args.name, sess.graph)

                if args.test_training:
                    camera_pos_vals = np.load(os.path.join(args.dataroot, 'train.npy'))
                    time_vals = np.load(os.path.join(args.dataroot, 'train_time.npy'))
                    if args.motion_blur:
                        camera_pos_velocity_vals = np.load(os.path.join(args.dataroot, 'train_velocity.npy'))
                else:
                    if (not args.motion_blur) and (not args.temporal_texture_buffer):
                        camera_pos_vals = np.concatenate((
                                            np.load(os.path.join(args.dataroot, 'test_close.npy')),
                                            np.load(os.path.join(args.dataroot, 'test_far.npy')),
                                            np.load(os.path.join(args.dataroot, 'test_middle.npy'))
                                            ), axis=0)
                    else:
                        camera_pos_vals = np.load(os.path.join(args.dataroot, 'test.npy'))
                    time_vals = np.load(os.path.join(args.dataroot, 'test_time.npy'))
                    if args.motion_blur:
                        camera_pos_velocity_vals = np.load(os.path.join(args.dataroot, 'test_velocity.npy'))

                for dir in dirs:
                    success = False
                    try:
                        epoch = int(dir)
                        if epoch % args.validate_loss_freq == 0:
                            success = True
                    except:
                        pass
                    if not success:
                        continue


                    ckpt = tf.train.get_checkpoint_state(os.path.join(args.name, dir))
                    if ckpt is None:
                        continue
                    print('loaded '+ckpt.model_checkpoint_path)
                    saver.restore(sess,ckpt.model_checkpoint_path)

                    avg_loss = float(open(os.path.join(args.name, dir, 'score.txt')).read())
                    all_test = np.zeros(len(val_names), dtype=float)
                    all_grad = np.zeros(len(val_names), dtype=float)
                    for ind in range(len(val_names)):
                        if not args.use_queue:
                            output_image = np.expand_dims(read_name(val_img_names[ind], False, False), axis=0)
                            st=time.time()
                            #feed_dict = {camera_pos: camera_pos_vals[ind, :], shader_time: time_vals[ind: ind+1], output: output_image}
                            feed_dict[camera_pos] = np.expand_dims(camera_pos_vals[ind, :], axis=1)
                            feed_dict[shader_time] = time_vals[ind: ind+1]
                            feed_dict[output_pl] = output_image
                            feed_dict[h_start] = np.array([- padding_offset / 2])
                            feed_dict[w_start] = np.array([- padding_offset / 2])
                            if args.motion_blur:
                                feed_dict[camera_pos_velocity] = camera_pos_velocity_vals[ind, :]

                            if args.use_weight_map or args.gradient_loss_canny_weight:
                                feed_dict[weight_map] = np.expand_dims(read_name(val_map_names[ind], True), 0)
                            if args.gradient_loss:
                                grad_arr = read_name(val_grad_names[ind], True)
                                feed_dict[canny_edge] = grad_arr[:, :, :, 0]
                                if args.grayscale_grad:
                                    feed_dict[dx_ground] = grad_arr[:, :, :, 1:2]
                                    feed_dict[dy_ground] = grad_arr[:, :, :, 2:3]
                                else:
                                    feed_dict[dx_ground] = grad_arr[:, :, :, 1:4]
                                    feed_dict[dy_ground] = grad_arr[:, :, :, 4:]
                            current, l2_loss_val, gradient_loss_val = sess.run([loss, loss_l2, loss_add_term],feed_dict=feed_dict)
                            print("%d, %d, %.3f"%(epoch, ind, time.time()-st))
                        else:
                            st=time.time()
                            current, l2_loss_val = sess.run([loss, loss_l2],feed_dict={alpha: alpha_val})
                            print("%.3f"%(time.time()-st))
                        all_test[ind] = l2_loss_val
                        all_grad[ind] = gradient_loss_val

                    if not args.test_training:
                        avg_test_close = np.mean(all_test[:5])
                        avg_test_far = np.mean(all_test[5:10])
                        avg_test_middle = np.mean(all_test[10:])
                        avg_test_all = np.mean(all_test)
                        open(os.path.join(os.path.join(args.name, dir, 'test_score.txt')), 'w').write('%f, %f, %f, %f' % (avg_test_close, avg_test_far, avg_test_middle, avg_test_all))

                    if args.write_summary:
                        summary = tf.Summary()
                        summary.value.add(tag='avg_loss', simple_value=avg_loss)
                        summary.value.add(tag='gradient_loss', simple_value=np.mean(all_grad))
                        summary.value.add(tag='l2_loss', simple_value=np.mean(all_test))

                        if not args.test_training:
                            avg_test_close = np.mean(all_test[:5])
                            avg_test_far = np.mean(all_test[5:10])
                            avg_test_middle = np.mean(all_test[10:])
                            avg_test_all = np.mean(all_test)

                            summary.value.add(tag='avg_test_close', simple_value=avg_test_close)
                            summary.value.add(tag='avg_test_far', simple_value=avg_test_far)
                            summary.value.add(tag='avg_test_middle', simple_value=avg_test_middle)
                            summary.value.add(tag='avg_test_all', simple_value=avg_test_all)

                        train_writer.add_summary(summary, epoch)

        else:
            test_dirbase = 'train' if args.test_training else 'test'
            if args.clip_weights > 0:
                test_dirname = "%s/%s_abs%s"%(args.name, test_dirbase, str(args.clip_weights).replace('.', ''))
            elif args.clip_weights_percentage > 0:
                test_dirname = "%s/%s_pct%s"%(args.name, test_dirbase, str(args.clip_weights_percentage).replace('.', ''))
            elif args.clip_weights_percentage_after_normalize > 0:
                test_dirname = "%s/%s_pct_norm%s"%(args.name, test_dirbase, str(args.clip_weights_percentage_after_normalize).replace('.', ''))
            else:
                test_dirname = "%s/%s"%(args.name, test_dirbase)

            if read_from_epoch and not args.is_train:
                test_dirname += "_epoch_%04d"%args.which_epoch

            if not os.path.isdir(test_dirname):
                os.makedirs(test_dirname)

            if args.clip_weights > 0:
                var_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                count_zero = 0
                count_all = 0
                for var in var_all:
                    if args.encourage_sparse_features:
                        success = 'w0' in var.name and 'feature_reduction' in var.name
                    else:
                        success = 'weights' in var.name
                    success = success and 'Adam' not in var.name
                    if success:
                        weights_val = sess.run(var)
                        inds = abs(weights_val) < args.clip_weights
                        weights_val[inds] = 0
                        count_zero += np.sum(inds)
                        count_all += weights_val.size
                        sess.run(tf.assign(var, weights_val))
                clip_percent = count_zero / count_all * 100
                target = open(os.path.join(test_dirname, 'clip_info.txt'), 'w')
                target.write("""
        threshold: {args.clip_weights}
        clipped_weights: {count_zero} / {count_all}
        percentage: {clip_percent}%""".format(**locals()))
                target.close()
            elif args.clip_weights_percentage > 0:
                all_weights = np.empty(0)
                var_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                for var in var_all:
                    if args.encourage_sparse_features:
                        success = 'w0' in var.name and 'feature_reduction' in var.name
                    else:
                        success = 'weights' in var.name
                    success = success and 'Adam' not in var.name
                    if success:
                        weights_val = sess.run(var)
                        all_weights = np.concatenate((all_weights, np.absolute(weights_val.reshape(weights_val.size))))

                percentile = np.percentile(abs(all_weights), args.clip_weights_percentage)
                for var in var_all:
                    if args.encourage_sparse_features:
                        success = 'w0' in var.name and 'feature_reduction' in var.name
                    else:
                        success = 'weights' in var.name
                    success = success and 'Adam' not in var.name
                    if success:
                        weights_val = sess.run(var)
                        weights_val[abs(weights_val) < percentile] = 0
                        sess.run(tf.assign(var, weights_val))
                total_num_weights = all_weights.shape[0]
                target = open(os.path.join(test_dirname, 'clip_info.txt'), 'w')
                target.write("""
        percentage: {args.clip_weights_percentage}%
        percentile value: {percentile}
        total number of weights: {total_num_weights}""".format(**locals()))
                target.close()
            elif args.clip_weights_percentage_after_normalize > 0.0:
                    weights_val = sess.run(weights_to_input, feed_dict={replace_normalize_weights: False, normalize_weights: np.empty((1, 1, args.input_nc, actual_conv_channel))})
                    percentile = np.percentile(abs(weights_val), args.clip_weights_percentage_after_normalize)
                    weights_val[abs(weights_val) < percentile] = 0
                    elements_left_per_row = numpy.sum((weights_val != 0), (0, 1, 2))
                    numpy.savetxt(os.path.join(test_dirname, 'weights_statistic.txt'), elements_left_per_row, '%d', delimiter=',    ', newline=',    \n')
                    #numpy.save(os.path.join(test_dirname, 'weights_val.npy'), weights_val)
                    target = open(os.path.join(test_dirname, 'clip_info.txt'), 'w')
                    target.write("""
        percentage: {args.clip_weights_percentage_after_normalize}%
        percentile value after normalize: {percentile}""".format(**locals()))
                    target.close()

            all_test=np.zeros(len(val_names), dtype=float)
            for ind in range(len(val_names)):
                if args.generate_timeline:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                else:
                    run_options = None
                    run_metadata = None
                feed_dict = {alpha: alpha_val}
                if not args.use_queue:
                    if args.preload:
                        input_image = eval_images[ind]
                        output_image = eval_out_images[ind]
                    else:
                        input_image = np.expand_dims(read_name(val_names[ind], args.is_npy, args.is_bin), axis=0)
                        output_image = np.expand_dims(read_name(val_img_names[ind], False, False), axis=0)
                    if input_image is None:
                        continue
                    st=time.time()
                    feed_dict[input] = input_image
                    feed_dict[output_pl] = output_image
                    #output_image, current=sess.run([network, loss],feed_dict={input:input_image, output: output_image, alpha: alpha_val})
                else:
                    st=time.time()
                    #output_image, current=sess.run([network, loss],feed_dict={alpha: alpha_val})
                if args.clip_weights_percentage_after_normalize > 0:
                    feed_dict[replace_normalize_weights] = True
                    feed_dict[normalize_weights] = weights_val
                if args.use_weight_map or args.gradient_loss_canny_weight:
                    feed_dict[weight_map] = np.expand_dims(read_name(val_map_names[ind], True), 0)
                if args.gradient_loss:
                    grad_arr = read_name(val_grad_names[ind], True)
                    feed_dict[canny_edge] = grad_arr[:, :, :, 0]
                    if args.grayscale_grad:
                        feed_dict[dx_ground] = grad_arr[:, :, :, 1:2]
                        feed_dict[dy_ground] = grad_arr[:, :, :, 2:3]
                    else:
                        feed_dict[dx_ground] = grad_arr[:, :, :, 1:4]
                        feed_dict[dy_ground] = grad_arr[:, :, :, 4:]
                output_image, current=sess.run([network, loss_l2],feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                print("%.3f"%(time.time()-st))
                all_test[ind] = current
                output_image=np.minimum(np.maximum(output_image,0.0),1.0)*255.0
                if args.use_queue:
                    output_image = output_image[:, :, :, ::-1]
                cv2.imwrite("%s/%06d.png"%(test_dirname, ind+1),np.uint8(output_image[0,:,:,:]))
                if args.generate_timeline:
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open("%s/nn_%d.json"%(test_dirname, ind+1), 'w') as f:
                        f.write(chrome_trace)

        target=open(os.path.join(test_dirname, 'score.txt'),'w')
        target.write("%f"%np.mean(all_test[np.where(all_test)]))
        target.close()
        target=open(os.path.join(test_dirname, 'vgg.txt'),'w')
        target.write("%f"%np.mean(all_perceptual[np.where(all_perceptual)]))
        target.close()
        target=open(os.path.join(test_dirname, 'vgg_same_scale.txt'),'w')
        target.write("%f"%np.mean(all_perceptual[np.where(all_perceptual)]))
        target.close()
        if args.use_dataroot and len(val_img_names) == 30:
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

    first_layer_dict = {}
    for var in var_first_layer_only:
        first_layer_dict[var.name] = sess.run(var)
    save_obj(first_layer_dict, "%s/first_layer.pkl"%(args.name))

    if args.use_queue and not args.debug_mode:
        coord.request_stop()
        coord.join(threads)
    sess.close()

    #if not args.is_train:
    if False:
        check_command = 'source activate pytorch36 && CUDA_VISIBLE_DEVICES=2, python plot_clip_weights.py ' + test_dirname + ' ' + grounddir + ' && source activate tensorflow35'
        #check_command = 'python plot_clip_weights.py ' + test_dirname + ' ' + grounddir
        subprocess.check_output(check_command, shell=True)
        #os.system('source activate pytorch36 && CUDA_VISIBLE_DEVICES=2, python plot_clip_weights.py ' + test_dirname + ' ' + grounddir)

if __name__ == '__main__':
    main()
