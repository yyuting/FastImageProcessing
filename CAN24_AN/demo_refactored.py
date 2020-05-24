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

deprecated_options = ['feature_reduction_channel_by_samples', 'is_npy', 'orig_channel', 'nsamples', 'is_bin', 'upsample_scale', 'upsample_single', 'upsample_shrink_feature', 'clip_weights', 'deconv', 'share_weights', 'clip_weights_percentage', 'encourage_sparse_features', 'validate_loss_freq', 'collect_validate_while_training', 'clip_weights_percentage_after_normalize', 'normalize_weights', 'normalize_weights', 'rowwise_L2_normalize', 'Frobenius_normalize', 'bilinear_upsampling', 'full_resolution', 'unet', 'unet_base_channel', 'learn_scale', 'soft_scale', 'scale_ratio', 'use_sigmoid', 'orig_rgb', 'use_weight_map', 'render_camera_pos_velocity', 'gradient_loss', 'normalize_grad', 'grayscale_grad', 'cos_sim', 'gradient_loss_scale', 'gradient_loss_all_pix', 'gradient_loss_canny_weight', 'two_stage_training', 'new_minimizer', 'weight_map_add', 'sigmoid_scaling', 'visualize_scaling', 'visualize_ind', 'test_tiling', 'motion_blur', 'dynamic_training_samples', 'dynamic_training_mode', 'automatic_find_gpu', 'test_rotation', 'abs_normalize', 'feature_reduction_regularization_scale', 'learn_sigma', 'repeat_timing', 'name', 'specified_ind']

def get_tensors(dataroot, name, camera_pos, shader_time, output_type='remove_constant', nsamples=1, shader_name='zigzag', geometry='plane', feature_w=[], color_inds=[], intersection=True, manual_features_only=False, aux_plus_manual_features=False, efficient_trace=False, collect_loop_statistic=False, h_start=0, h_offset=height, w_start=0, w_offset=width, samples=None, fov='regular', camera_pos_velocity=None, t_sigma=1/60.0, first_last_only=False, last_only=False, subsample_loops=-1, last_n=-1, first_n=-1, first_n_no_last=-1, mean_var_only=False, zero_samples=False, render_fix_spatial_sample=False, render_fix_temporal_sample=False, render_zero_spatial_sample=False, spatial_samples=None, temporal_samples=None, every_nth=-1, every_nth_stratified=False, one_hop_parent=False, target_idx=[], use_manual_index=False, manual_index_file='', additional_features=True, ignore_last_n_scale=0, include_noise_feature=False, crop_h=-1, crop_w=-1, no_noise_feature=False, relax_clipping=False, render_sigma=None, same_sample_all_pix=False, stratified_sample_higher_res=False, samples_int=[None], texture_maps=[], partial_trace=1.0, use_lstm=False, lstm_nfeatures_per_group=1, rotate=0, flip=0, use_dataroot=True, automatic_subsample=False, automate_raymarching_def=False, chron_order=False, def_loop_log_last=False, temporal_texture_buffer=False, texture_inds=[], log_only_return_def_raymarching=True, debug=[], SELECT_FEATURE_THRE=200, n_boids=40, log_getitem=True, color_scale=[], parallel_stack=True, compiler_problem_idx=-1, input_feature_pl=[], input_to_shader=[], trace_features=[], input_feature_scale_bias=[], finite_diff=False, feature_normalize_lo_pct=20, get_col_aux_inds=False, specified_ind=None, write_file=True, alt_dir=''):
    # 2x_1sample on margo
    #camera_pos = np.load('/localtmp/yuting/out_2x1_manual_carft/train.npy')[0, :]

    #feature_scale = np.load('/localtmp/yuting/out_2x1_manual_carft/train/zigzag_plane_normal_spheres/datas_rescaled_25_75_2_153/feature_scale.npy')
    #feature_bias = np.load('/localtmp/yuting/out_2x1_manual_carft/train/zigzag_plane_normal_spheres/datas_rescaled_25_75_2_153/feature_bias.npy')

    manual_features_only = manual_features_only or aux_plus_manual_features

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
        elif shader_name == 'bricks_normal_texture':
            shader_args = ' render_bricks_normal_texture ' + geometry + ' none '
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
        elif shader_name in ['fluid_approximate', 'fluid_approximate_3pass_10f']:
            shader_args = ' render_fluid_approximate_3pass_10f ' + geometry + ' none'
        elif shader_name == 'fluid_approximate_3pass_no_mouse_perlin':
            shader_args = ' render_fluid_approximate_3pass_no_mouse_perlin ' + geometry + ' none'
        elif shader_name == 'fluid_approximate_sin':
            shader_args = ' render_fluid_approximate_sin ' + geometry + ' none'
        elif shader_name == 'boids':
            geometry = 'boids'
            shader_args = ' render_boids ' + geometry + ' none'
        elif shader_name == 'boids_coarse':
            geometry = 'boids_coarse'
            shader_args = ' render_boids_coarse ' + geometry + ' none'
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
        if not intersection:
            render_single_cmd = render_single_cmd + ' --log_intermediates_level 1'
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
        if log_only_return_def_raymarching:
            render_single_cmd = render_single_cmd + ' --log_only_return_def_raymarching'
        if shader_name.startswith('boids'):
            render_single_cmd = render_single_cmd + ' --n_boids %d' % n_boids
        if not log_getitem:
            render_single_cmd = render_single_cmd + ' --no_log_getitem'
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
    
    texture_map_size = []
    
    if len(input_feature_pl) > 0:
        feature_pl = [None] * len(input_feature_pl)
        for i in range(len(input_feature_pl)):
            if input_feature_pl[i].dtype != dtype:
                feature_pl[i] = tf.cast(input_feature_pl[i], dtype)
            else:
                feature_pl[i] = input_feature_pl[i]
    else:
        feature_pl = []

    features, vec_output, manual_features = get_render(camera_pos, shader_time, nsamples=nsamples, shader_name=shader_name, geometry=geometry, return_vec_output=True, compiler_module=compiler_module, manual_features_only=manual_features_only, aux_plus_manual_features=aux_plus_manual_features, h_start=h_start, h_offset=h_offset, w_start=w_start, w_offset=w_offset, samples=samples, fov=fov, camera_pos_velocity=camera_pos_velocity, t_sigma=t_sigma, zero_samples=zero_samples, render_fix_spatial_sample=render_fix_spatial_sample, render_fix_temporal_sample=render_fix_temporal_sample, render_zero_spatial_sample=render_zero_spatial_sample, spatial_samples=spatial_samples, temporal_samples=temporal_samples, additional_features=additional_features, include_noise_feature=include_noise_feature, no_noise_feature=no_noise_feature, render_sigma=render_sigma, same_sample_all_pix=same_sample_all_pix, stratified_sample_higher_res=stratified_sample_higher_res, samples_int=samples_int, texture_maps=texture_maps, temporal_texture_buffer=temporal_texture_buffer, n_boids=n_boids, texture_map_size=texture_map_size, debug=debug, input_feature_pl=feature_pl, input_to_shader=input_to_shader, phase=compiler_problem_idx, finite_diff=finite_diff)
    
    if len(input_feature_pl) > 0:
        # hack, should fix later to generalize better
        features = features + vec_output
        #for feat in features:
        #    trace_features.append(feat)
    
    # workaround if for some feature sparsification setup, RGB channels are not logged
    # also prevent aux feature from not being logged
    if efficient_trace:
        features = features + vec_output + manual_features
        
    if temporal_texture_buffer:
        out_textures = vec_output[:]
    
    #if len(vec_output) > 3:
    #    loop_statistic = vec_output[3:]
    #    vec_output = vec_output[:3]
    #    features = features + loop_statistic
        
    
        
    if temporal_texture_buffer:
        if shader_name.startswith('boids'):
            pass
        elif isinstance(texture_maps, list): 
            # hack: both features and manual_features contain texture input
            reshaped_texture_maps = []
            for i in range(len(texture_maps)):
                if w_offset == width and h_offset == height:
                    reshaped_texture_maps.append(tf.expand_dims(texture_maps[i], 0))
                else:
                    reshaped_texture_maps.append(tf.expand_dims(tf.pad(texture_maps[i], [[padding_offset // 2, padding_offset // 2], [padding_offset // 2, padding_offset // 2]], "CONSTANT"), 0))
                features.append(reshaped_texture_maps[-1])
                manual_features.append(reshaped_texture_maps[-1])
        else:
            # if texture_maps is a tensor generated from dataset pipeline, then assume it's not in tile mode
            assert w_offset == width and h_offset == height

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
                # hack for fluid approx mode: because we also add texture to aux features, should include their inds
                if temporal_texture_buffer and (not isinstance(texture_maps, list)):
                    for i in range(texture_map_size[0]):
                        manual_inds.append(len(valid_features)+i)
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
                
            if temporal_texture_buffer:
                if isinstance(texture_maps, list):
                    for i in range(3):
                        # hack, directly assume the first 3 channels of texture map is the starting color channel
                        # this is only useful when providing the input condition to discriminator
                        # will not be useful for mean estimator since it's just initial status of the fluid
                        actual_ind = out_features.index(reshaped_texture_maps[i])
                        color_inds.append(actual_ind)

                    for vec in out_textures:
                        actual_ind = out_features.index(vec)
                        texture_inds.append(actual_ind)
                else:
                    # hack, because in pipeline mode, directly concat features with texture_maps
                    # can safely assume the last 7 channels are provided texture_maps
                    current_len = len(out_features)
                    #color_inds = current_len + np.array([0, 1, 2])
                    if not shader_name.startswith('boids'):
                        for i in range(3):
                            color_inds.append(current_len+i)
                    texture_inds = current_len + np.arange(texture_map_size[0])

            if output_type not in ['rgb', 'bgr'] and use_dataroot and (not geometry.startswith('boids')):
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
            
            if len(feature_pl) > 0:
                for var in feature_pl:
                    if var in out_features:
                        idx = out_features.index(var)
                        input_feature_scale_bias.append((feature_scale[idx], feature_bias[idx]))
                    else:
                        input_feature_scale_bias.append(None)
            
            if output_type == 'remove_constant':
                if parallel_stack:
                    features = tf.parallel_stack(out_features)

                    if not shader_name.startswith('boids'):
                        features = tf.transpose(features, [1, 2, 3, 0])

                        if temporal_texture_buffer and (not isinstance(texture_maps, list)):
                            # TODO: one caveat here is we're assuming slices of texture is never part of the trace
                            # this is true for the fluid approx example
                            # however, it may not be always guaranteed if we write other shaders
                            features = tf.concat((features, tf.expand_dims(tf.transpose(texture_maps, [1, 2, 0]), 0)), -1)
                    else:
                        features = tf.transpose(features, [1, 2, 0])
                        features = tf.concat((features, texture_maps), -1)
                else:
                    features = tf.stack(out_features, -1)

            elif output_type == 'all':
                features = tf.cast(tf.stack(features, axis=-1), tf.float32)
            elif output_type in ['rgb', 'bgr']:
                features = tf.cast(tf.stack(vec_output, axis=-1), tf.float32)
                if output_type == 'bgr':
                    features = features[..., ::-1]
            else:
                raise
            
            if use_dataroot and output_type not in ['rgb', 'bgr']:
                color_scale.append(feature_bias[color_inds])
                color_scale.append(feature_scale[color_inds])

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
                if shader_name.startswith('boids'):
                    # return early to avoid further clipping
                    return features

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
    if crop_h > 0:
        features = features[:, :crop_h, :, :]
    if crop_w > 0:
        features = features[:, :, :crop_w, :]
    
    if isinstance(rotate, tf.Tensor):
        # TODO: it's a hack that rotate is a scalar
        # it should be a seperate value for each sample in a batch, and they could either be rotated or not
        # but it's much easeir to code if treat rotate as a scalar for the whole batch
        features = tf.cond(rotate > 0, lambda: tf.image.rot90(features, rotate), lambda: features)
    if isinstance(flip, tf.Tensor):
        # as long as rotate and flip are 2 seperate generated random integers, these 2 values combined can generate all 8 different random permutation in the image space
        features = tf.cond(flip > 0, lambda: tf.image.flip_left_right(features), lambda: features)
    return features

    #numpy.save('valid_inds.npy', valid_inds)
    #return features

def get_render(camera_pos, shader_time, samples=None, nsamples=1, shader_name='zigzag', color_inds=None, return_vec_output=False, render_size=None, render_sigma=None, compiler_module=None, geometry='plane', zero_samples=False, debug=[], extra_args=[None], render_g=False, manual_features_only=False, aux_plus_manual_features=False, fov='regular', h_start=0, h_offset=height, w_start=0, w_offset=width, camera_pos_velocity=None, t_sigma=1/60.0, render_fix_spatial_sample=False, render_fix_temporal_sample=False, render_zero_spatial_sample=False, spatial_samples=None, temporal_samples=None, additional_features=True, include_noise_feature=False, no_noise_feature=False, same_sample_all_pix=False, stratified_sample_higher_res=False, samples_int=[None], texture_maps=[], temporal_texture_buffer=False, n_boids=40, texture_map_size=[], input_feature_pl=[], input_to_shader=[], phase=-1, finite_diff=False):

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

    if camera_pos_velocity is not None:
        features_len_add += 7
        if fov == 'regular':
            print("creating dx/dt, dy/dt")
            # marble case
            features_len_add += 2
            
    if geometry in ['boids', 'boids_coarse']:
        features_len_add = 0

    features_len = compiler_module.f_log_intermediate_len + features_len_add

    vec_output_len = compiler_module.vec_output_len

    manual_features_len = compiler_module.f_log_intermediate_subset_len
    manual_depth_offset = 0
    if geometry not in ['none', 'texture', 'texture_approximate_10f', 'boids', 'boids_coarse']:
        manual_features_len += 1
        manual_depth_offset = 1
    if aux_plus_manual_features:
        manual_features_len += features_len_add
    if camera_pos_velocity is not None:
        manual_features_len += 6
        motion_start = compiler_module.f_log_intermediate_subset_len
    f_log_intermediate_subset = [None] * manual_features_len

        
    if render_size is not None:
        global width
        global height
        width = render_size[0]
        height = render_size[1]
        
    texture_map_size.append(compiler_module.vec_output_len)
    if temporal_texture_buffer and (texture_maps == []):
        for i in range(texture_map_size[0]):
            texture_maps.append(tf.placeholder(tf.float32, [height, width]))

    f_log_intermediate = [None] * features_len
    vec_output = [None] * vec_output_len
    #vec_output[0] = {camera_pos: np.expand_dims(np.array([-0.07611784,  3.81202799, -1.45621813,  5.82435499,  4.37503681, 2.70497378]), 1), shader_time: np.array([1.7]), h_start:np.array([-16.]), w_start:np.array([-16.])}
    
    if not geometry.startswith('boids'):
        # 2D shader case

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
    else:
        vector3 = [texture_maps, shader_time, tf.tile(tf.expand_dims(tf.cast(tf.range(n_boids), dtype), 0), [nsamples, 1])]
    get_shader(vector3, f_log_intermediate, f_log_intermediate_subset, camera_pos, features_len, manual_features_len, shader_name=shader_name, color_inds=color_inds, vec_output=vec_output, compiler_module=compiler_module, geometry=geometry, debug=debug, extra_args=extra_args, render_g=render_g, manual_features_only=manual_features_only, aux_plus_manual_features=aux_plus_manual_features, fov=fov, camera_pos_velocity=camera_pos_velocity, features_len_add=features_len_add, manual_depth_offset=manual_depth_offset, additional_features=additional_features, texture_maps=texture_maps, n_boids=n_boids, input_feature_pl=input_feature_pl, input_to_shader=input_to_shader, phase=phase)

    # TODO: potential bug here
    # what to put if zero_samples = True
    
    if finite_diff:
        sample1 = tf.tile(sample1, [int(camera_pos.shape[1]), 1, 1])
        sample2 = tf.tile(sample2, [int(camera_pos.shape[1]), 1, 1])
    
    if (not no_noise_feature) and not geometry.startswith('boids'):
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

def get_shader(x, f_log_intermediate, f_log_intermediate_subset, camera_pos, features_len, manual_features_len, shader_name='zigzag', color_inds=None, vec_output=None, compiler_module=None, geometry='plane', debug=[], extra_args=[None], render_g=False, manual_features_only=False, aux_plus_manual_features=False, fov='regular', camera_pos_velocity=None, features_len_add=7, manual_depth_offset=1, additional_features=True, texture_maps=[], n_boids=40, input_feature_pl=[], input_to_shader=[], phase=-1):
    assert compiler_module is not None
    features_dt = []
    
    if phase in [29, 291, 31]:
        input_pl_to_features = input_feature_pl
    else:
        input_pl_to_features = []
    
    features = get_features(x, camera_pos, geometry=geometry, debug=debug, extra_args=extra_args, fov=fov, camera_pos_velocity=camera_pos_velocity, features_dt=features_dt, n_boids=n_boids, input_feature_pl=input_pl_to_features, phase=phase)
    
    if vec_output is None:
        vec_output = [None] * 3

    # adding depth
    if geometry == 'plane':
        f_log_intermediate_subset[-1] = features[7]
    elif geometry in ['hyperboloid1', 'paraboloid']:
        f_log_intermediate_subset[-1] = extra_args[0]
    elif geometry not in ['none', 'texture', 'texture_approximate_10f', 'boids', 'boids_coarse']:
        raise

    with tf.variable_scope("auxiliary"):

        if geometry not in ['none', 'texture', 'texture_approximate_10f', 'boids', 'boids_coarse'] and additional_features:
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

    for feat in features:
        input_to_shader.append(feat)
            
    if len(debug) > 0:
        vec_output[0] = debug[0]
    if not render_g:
        if texture_maps != []:
            compiler_module.f(features, f_log_intermediate, vec_output, f_log_intermediate_subset, texture_maps=texture_maps)
        else:
            if len(input_feature_pl) > 0:
                compiler_module.f(features, f_log_intermediate, vec_output, f_log_intermediate_subset, trace_pl=input_feature_pl)
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

def get_features(x, camera_pos, geometry='plane', debug=[], extra_args=[None], fov='regular', camera_pos_velocity=None, features_dt=[], n_boids=40, input_feature_pl=[], phase=-1):
    if geometry.startswith('boids'):
        # 1D simulation case
        if geometry == 'boids':
            features = [None] * 6
        else:
            features = [None] * 7
        features[0] = x[2]
        features[1] = x[0][..., 0]
        features[2] = x[0][..., 1]
        features[3] = x[0][..., 2]
        features[4] = x[0][..., 3]
        features[5] = tf.transpose(x[0], [1, 2, 0])
        if geometry == 'boids_coarse':
            features[6] = tf.expand_dims(x[1], 1) * tf.ones_like(features[0])
        return features
    if fov.startswith('regular'):
        ray_dir = [x[0] - width / 2, x[1] + 1, width / 2]
    elif fov.startswith('small'):
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
        if phase == 29:
            t_ray = input_feature_pl[7]
        elif phase == 291:
            t_ray = -ray_origin[2] * input_feature_pl[7]
        elif phase == 31:
            t_ray = -ray_origin[2] * input_feature_pl[0]
        else:
            t_ray = -ray_origin[2] / (ray_dir_p[2])
        features[0] = x[2]
        if phase in [29, 291]:
            features[1] = ray_origin[0] - t_ray * input_feature_pl[4]
            features[2] = ray_origin[1] - t_ray * input_feature_pl[5]
            features[3] = ray_origin[2] - t_ray * input_feature_pl[6]
        else:
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
    parser.add_argument('--dataroot', dest='dataroot', default='../data', help='directory to store training and testing data')
    parser.add_argument('--is_train', dest='is_train', action='store_true', help='state whether this is training or testing')
    parser.add_argument('--input_nc', dest='input_nc', type=int, default=-1, help='number of channels for input')
    parser.add_argument('--use_batch', dest='use_batch', action='store_true', help='whether to use batches in training')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='size of batches')
    parser.add_argument('--finetune', dest='finetune', action='store_true', help='fine tune on a previously tuned network')
    parser.add_argument('--orig_name', dest='orig_name', default='', help='name of original task that is fine tuned on')
    parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='number of epochs to train, seperated by comma')
    parser.add_argument('--debug_mode', dest='debug_mode', action='store_true', help='debug mode')
    parser.add_argument('--use_queue', dest='use_queue', action='store_true', help='whether to use queue instead of feed_dict ( when inputs are too large)')
    parser.add_argument('--no_preload', dest='preload', action='store_false', help='whether to preload data')
    parser.add_argument('--L1_regularizer_scale', dest='regularizer_scale', type=float, default=0.0, help='scale for L1 regularizer')
    parser.add_argument('--L2_regularizer_scale', dest='L2_regularizer_scale', type=float, default=0.0, help='scale for L2 regularizer')
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
    parser.add_argument('--update_bn', dest='update_bn', action='store_true', help='accurately update batch normalization')
    parser.add_argument('--conv_channel_no', dest='conv_channel_no', type=int, default=-1, help='directly specify number of channels for dilated conv layers')
    parser.add_argument('--mean_estimator', dest='mean_estimator', action='store_true', help='if true, use mean estimator instead of neural network')
    parser.add_argument('--estimator_samples', dest='estimator_samples', type=int, default=1, help='number of samples used in mean estimator')
    parser.add_argument('--accurate_timing', dest='accurate_timing', action='store_true', help='if true, do not calculate loss for more accurate timing')
    parser.add_argument('--shader_name', dest='shader_name', default='zigzag', help='shader name used to generate shader input in GPU')
    parser.add_argument('--batch_norm_only', dest='batch_norm_only', action='store_true', help='if specified, use batch norm only (no adaptive normalization)')
    parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false', help='if specified, do not apply batch norm')
    parser.add_argument('--data_from_gpu', dest='data_from_gpu', action='store_true', help='if specified input data is generated from gpu on the fly')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.0001, help='learning rate for adam optimizer')
    parser.add_argument('--identity_initialize', dest='identity_initialize', action='store_true', help='if specified, initialize weights such that output is 1 sample RGB')
    parser.add_argument('--nonzero_ini', dest='allow_nonzero', action='store_true', help='if specified, use xavier for all those supposed to be 0 entries in identity_initializer')
    parser.add_argument('--no_identity_output_layer', dest='identity_output_layer', action='store_false', help='if specified, do not use identity mapping for output layer')
    parser.add_argument('--less_aggresive_ini', dest='less_aggresive_ini', action='store_true', help='if specified, use a less aggresive way to initialize RGB weights (multiples of the original xavier weights)')
    parser.add_argument('--render_only', dest='render_only', action='store_true', help='if specified, render using given camera pos, does not calculate loss')
    parser.add_argument('--render_camera_pos', dest='render_camera_pos', default='camera_pos.npy', help='used to render result')
    parser.add_argument('--render_t', dest='render_t', default='render_t.npy', help='used to render output')
    parser.add_argument('--render_temporal_texture', dest='render_temporal_texture', default='', help='used to provide initial temporal texture buffer')
    parser.add_argument('--train_res', dest='train_res', action='store_true', help='if specified, out_img = in_noisy_img + out_network')
    parser.add_argument('--no_intersection', dest='intersection', action='store_false', help='if specified, do not include geometry intersection computation in intermediate variables')
    parser.add_argument('--geometry', dest='geometry', default='plane', help='geometry of shader')
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
    parser.add_argument('--sparsity_target_channel', dest='sparsity_target_channel', type=int, default=100, help='specifies the target nonzero channel entries for sparsity vec')
    parser.add_argument('--random_target_channel', dest='random_target_channel', action='store_true', help='if specified, randomly select target number of channels, used as a baseline for sparsity experiments')
    parser.add_argument('--schedule_scale_after_freeze', dest='schedule_scale_after_freeze', action='store_true', help='if specified, weight scale is changed only after phase 1 (changing nonconvexity) is finished and phase 2 (maximum nonconvexity fixed) is started')
    parser.add_argument('--finetune_epoch', dest='finetune_epoch', type=int, default=-1, help='if specified, use checkpoint from specified epoch for finetune start point')
    parser.add_argument('--every_nth_stratified', dest='every_nth_stratified', action='store_true', help='if specified, do stratified sampling for every nth traces')
    parser.add_argument('--aux_plus_manual_features', dest='aux_plus_manual_features', action='store_true', help='if specified, use RGB+aux+manual features')
    parser.add_argument('--use_manual_index', dest='use_manual_index', action='store_true', help='if true, only use trace indexed in dataroot for training')
    parser.add_argument('--manual_index_file', dest='manual_index_file', default='index.npy', help='specifies file that stores index of trace used for training')
    parser.add_argument('--no_additional_features', dest='additional_features', action='store_false', help='if specified, do not use additional features during training')
    parser.add_argument('--ignore_last_n_scale', dest='ignore_last_n_scale', type=int, default=0, help='if nonzero, ignore the last n entries of stored feature_bias and feature_scale')
    parser.add_argument('--include_noise_feature', dest='include_noise_feature', action='store_true', help='if specified, include noise as additional features during trianing')
    parser.add_argument('--crop_w', dest='crop_w', type=int, default=-1, help='if specified, crop features / imgs on width dimension upon specified ind')
    parser.add_argument('--crop_h', dest='crop_h', type=int, default=-1, help='if specified, crop features / imgs on height dimension upon specified ind')
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
    parser.add_argument('--stratified_sample_higher_res', dest='stratified_sample_higher_res', action='store_true', help='in this mode, learn from a gt that is stratified sample from a higher res gt')
    parser.add_argument('--save_intermediate_epoch', dest='save_intermediate_epoch', action='store_true', help='if true, save all intermediate epochs')
    parser.add_argument('--texture_maps', dest='texture_maps', default='', help='if not empty, retrieve texture map from the file')
    parser.add_argument('--additional_input', dest='additional_input', action='store_true', help='if true, find additional input features from train/test_add')
    parser.add_argument('--partial_trace', dest='partial_trace', type=float, default=1.0, help='if less than 1, only record the first 100x percent of the trace wehre x = partial_trace')
    parser.add_argument('--use_lstm', dest='use_lstm', action='store_true', help='if specified, use lstm to process raw feature')
    parser.add_argument('--lstm_nfeatures_per_group', dest='lstm_nfeatures_per_group', type=int, default=1, help='specifies how many features processed per timestamp by LSTM')
    parser.add_argument('--patch_gan_loss', dest='patch_gan_loss', action='store_true', help='if specified, use a patch gan loss together with existing loss')
    parser.add_argument('--no_spatial_GAN', dest='spatial_GAN', action='store_false', help='if sepcified, do not include spatial GAN. This option is only valid when patch_gan_loss is true, and train_temporal_seq is also true.')
    parser.add_argument('--ndf', dest='ndf', type=int, default=32, help='number of discriminator filters on first layer if using patch GAN loss')
    parser.add_argument('--ndf_temporal', dest='ndf_temporal', type=int, default=32, help='number of temporal discriminator filters on first layer')
    parser.add_argument('--gan_loss_scale', dest='gan_loss_scale', type=float, default=1.0, help='the scale multiplied to GAN loss before adding to regular loss')
    parser.add_argument('--discrim_nlayers', dest='discrim_nlayers', type=int, default=2, help='number of layers of discriminator')
    parser.add_argument('--discrim_use_trace', dest='discrim_use_trace', action='store_true', help='if specified, use trace info for discriminator')
    parser.add_argument('--discrim_trace_shared_weights', dest='discrim_trace_shared_weights', action='store_true', help='if specified, when discriminator is using trace, it directly uses the 48 dimension used by the generator, and the feature reduction layer weights will be shared with the generator')
    parser.add_argument('--discrim_paired_input', dest='discrim_paired_input', action='store_true', help='if specified, use paired input (gt, generated) or (generated, gt) to discriminator and predict which order is correct')
    parser.add_argument('--save_frequency', dest='save_frequency', type=int, default=100, help='specifies the frequency to save a checkpoint')
    parser.add_argument('--discrim_train_steps', dest='discrim_train_steps', type=int, default=1, help='specified how often to update discrim')
    parser.add_argument('--gan_loss_style', dest='gan_loss_style', default='cross_entropy', help='specifies what GAN loss to use')
    parser.add_argument('--train_with_random_rotation', dest='train_with_random_rotation', action='store_true', help='if specified, during training, will randomly rotate data in image space')
    parser.add_argument('--temporal_texture_buffer', dest='temporal_texture_buffer', action='store_true', help='if specified, render temporal correlated sequences, each frame is using texture rendered from previous frame')
    parser.add_argument('--no_dataroot', dest='use_dataroot', action='store_false', help='if specified, do not need a dataroot (used for check runtime)')
    parser.add_argument('--camera_pos_file', dest='camera_pos_file', default='', help='if specified, use for no_dataroot mode')
    parser.add_argument('--camera_pos_len', dest='camera_pos_len', type=int, default=50, help='specifies the number of camera pos used in no_dataroot mode')
    parser.add_argument('--feature_size_only', dest='feature_size_only', action='store_true', help='if specified, do not further create neural network, return after collecting the feature size')
    parser.add_argument('--automatic_subsample', dest='automatic_subsample', action='store_true', help='if specified, automatically decide program subsample rate (and raymarching and function def)')
    parser.add_argument('--automate_raymarching_def', dest='automate_raymarching_def', action='store_true', help='if specified, automatically choose schedule for raymarching and function def (but not subsampling rate')
    parser.add_argument('--chron_order', dest='chron_order', action='store_true', help='if specified, log trace in their execution order')
    parser.add_argument('--def_loop_log_last', dest='def_loop_log_last', action='store_true', help='if true, log the last execution of function, else, log first execution')
    parser.add_argument('--no_log_only_return_def_raymarching', dest='log_only_return_def_raymarching', action='store_false', help='if set, do not use default log_only_return_def_raymarching mode')
    parser.add_argument('--train_temporal_seq', dest='train_temporal_seq', action='store_true', help='if set, training on temporal sequences instead of on single frames')
    parser.add_argument('--temporal_seq_length', dest='temporal_seq_length', type=int, default=6, help='length of generated temporal seq during training')
    parser.add_argument('--nframes_temporal_gen', dest='nframes_temporal_gen', type=int, default=3, help='number of frames generator considers in temporal seq mode')
    parser.add_argument('--nframes_temporal_discrim', dest='nframes_temporal_discrim', type=int, default=3, help='number of frames temporal discrim considers in temporal seq mode')
    parser.add_argument('--SELECT_FEATURE_THRE', dest='SELECT_FEATURE_THRE', type=int, default=200, help='when automatically decide subsample rate, this will decide the trace budget')
    parser.add_argument('--temporal_discrim_only', dest='temporal_discrim_only', action='store_true', help='if set, do not use single frame discriminator')
    parser.add_argument('--n_boids', dest='n_boids', type=int, default=40, help='number of boids in boids app')
    parser.add_argument('--no_log_getitem', dest='log_getitem', action='store_false', help='if true, do not log getitem node in DSL')
    parser.add_argument('--niters', dest='niters', type=int, default=200, help='in examples e.g. boids, we will enumerate the entire training data, but rather, we would randomly sample from training dataset during each epoch, niters shows how many samples / steps trained per epoch')
    parser.add_argument('--repeat_timing', dest='repeat_timing', type=int, default=1, help='if > 1, repeat inference multiple times to get stable timing')
    parser.add_argument('--interval_sample_square', dest='interval_sample_square', action='store_true', help='if true, sample by 1/t^2')
    parser.add_argument('--interval_sample_geometric', dest='interval_sample_geometric', type=float, default=0.0, help='if >1, sample by 1/r^t')
    parser.add_argument('--fc_nlayer', dest='fc_nlayer', type=int, default=3, help='specifies the number of hidden layers (before output layer) for fully connected network')
    parser.add_argument('--min_sample_time', dest='min_sample_time', type=int, default=2, help='specifies the lower bound of time interval sampling for simulations')
    parser.add_argument('--max_sample_time', dest='max_sample_time', type=int, default=64, help='specifies the upper bound of time interval sampling for simulations')
    parser.add_argument('--use_validation', dest='use_validation', action='store_true', help='if set, terminate training hwen validation loss increases')
    parser.add_argument('--random_switch_label', dest='random_switch_label', action='store_true', help='a form of data augmentation, randomly switch label of boids')
    parser.add_argument('--inference_seq_len', dest='inference_seq_len', type=int, default=8, help='sequence length for inference')
    parser.add_argument('--boids_single_step_metric', dest='boids_single_step_metric', action='store_true', help='if true, also compute single step l2 error on test set')
    parser.add_argument('--boids_seq_metric', dest='boids_seq_metric', action='store_true', help='if true, synthesize long sequences and compute error')
    parser.add_argument('--analyze_nn_discontinuity', dest='analyze_nn_discontinuity', action='store_true', help='if true, analyze the discontinuity in each conv layer (after lrelu activation)')
    parser.add_argument('--optimize_input', dest='optimize_input', action='store_true', help='if true, do not use placeholder for input, use tf.Variable instead, also return session, optimizable varialbe and output node before rendering/training')
    parser.add_argument('--boost_channels', dest='boost_channels', default='', help='if having indices seperated by comma, create a regularizer on the rest of the indices to decrease their weights (therefore boost the importance of listed indices)')
    parser.add_argument('--no_boost_ch_from_finetune', dest='boost_ch_from_finetune', action='store_false', help='if specified, remove sparsity_vec from saver')
    parser.add_argument('--learn_loss_proxy', dest='learn_loss_proxy', action='store_true', help='if specified, instead of learning target gt image, learn the loss between the gt image to another input image instead')
    parser.add_argument('--optimize_target_img', dest='optimize_target_img', action='store_true', help='if specified and if in learn_loss_proxy mode, optimize input parameters to approximate the target image')
    parser.add_argument('--target_img', dest='target_img', default='', help='name of target image file for optimization')
    parser.add_argument('--list_parameter_idx', dest='list_parameter_idx', action='store_true', help='if specified, list the names of all placeholder parameters and their corresponding indices (which can be used in command line to specify)')
    parser.add_argument('--loss_proxy_encourage_0', dest='loss_proxy_encourage_0', action='store_true', help='if specified, set a quater of the training target to be the same as reference, so that the model can correctly learn the loss should be 0 in those cases')
    parser.add_argument('--proxy_loss_type', dest='proxy_loss_type', default='smoothed_l2', help='type of loss the proxy learns')
    parser.add_argument('--augment_camera_periodic_range', dest='augment_camera_periodic_range', default='', help='if specified, during training will randomly augment camera position by either incrementing or decrementing the range')
    parser.add_argument('--model_arch', dest='model_arch', default='dilated', help='specified the architeture model uses')
    parser.add_argument('--collect_inference_tensor', dest='collect_inference_tensor', action='store_true', help='if specified, return pl and tensor that can be used for inference')
    parser.add_argument('--optimize_nn_input_feature_space', dest='optimize_nn_input_feature_space', type=int, default=0, help='experiment mode')
    parser.add_argument('--compiler_problem_idx', dest='compiler_problem_idx', type=int, default=-1, help='if nonnegative, use this idx to find appropriate compiler problem')
    parser.add_argument('--render_no_video', dest='render_no_video', action='store_true', help='in this mode, render images only, do not generate video')
    parser.add_argument('--render_dirname', dest='render_dirname', default='render', help='directory used to store render result')
    parser.add_argument('--render_tile_start', dest='render_tile_start', default='', help='specifies the tile start for each rendering if render in test_training mode')
    parser.add_argument('--vgg_channels', dest='vgg_channels', default='', help='if specified, use these channel number to replace default vgg network')
    parser.add_argument('--feature_reduction_ch', dest='feature_reduction_ch', type=int, default=-1, help='specifies dimensionality after feature reduction channel. By default it should be the same as following initial layer or dilation layers, but we might want to change the dimensionality larger for fair RGBx comparison')
    parser.add_argument('--feed_dict_optimize_input', dest='feed_dict_optimize_input', action='store_true', help='if specified, create pl for optimize input mode, instead of variable')
    parser.add_argument('--finite_diff', dest='finite_diff', action='store_true', help='if specified, create 2 extra copy of camera_pos and shader_time in the batch dimension that can be used to compute finite difference later')
    parser.add_argument('--collect_validate_loss', dest='collect_validate_loss', action='store_true', help='if true, collect validation loss (and training score) and write to tensorboard')
    parser.add_argument('--read_from_best_validation', dest='read_from_best_validation', action='store_true', help='if true, read from the best validation checkpoint')
    parser.add_argument('--feature_normalize_lo_pct', dest='feature_normalize_lo_pct', type=int, default=25, help='used to find feature_bias file')
    parser.add_argument('--get_col_aux_inds', dest='get_col_aux_inds', action='store_true', help='if true, write the inds for color and aux channels and do nothing else')
    parser.add_argument('--specified_ind', dest='specified_ind', default='', help='if specified, using the specified ind to define a subset of the trace for learning')
    parser.add_argument('--test_output_dir', dest='test_output_dir', default='', help='if specified, write output to this directory instead')
    parser.add_argument('--no_overwrite_option_file', dest='overwrite_option_file', action='store_false', help='if specified, do not overwrite option file even if the old one is outdated')
    
    parser.set_defaults(is_train=False)
    parser.set_defaults(use_batch=False)
    parser.set_defaults(finetune=False)
    parser.set_defaults(debug_mode=False)
    parser.set_defaults(use_queue=False)
    parser.set_defaults(preload=True)
    parser.set_defaults(test_training=False)
    parser.set_defaults(generate_timeline=False)
    parser.set_defaults(add_initial_layers=False)
    parser.set_defaults(add_final_layers=False)
    parser.set_defaults(dilation_remove_large=False)
    parser.set_defaults(dilation_clamp_large=False)
    parser.set_defaults(dilation_remove_layer=False)
    parser.set_defaults(update_bn=False)
    parser.set_defaults(mean_estimator=False)
    parser.set_defaults(accurate_timing=False)
    parser.set_defaults(batch_norm_only=False)
    parser.set_defaults(batch_norm=True)
    parser.set_defaults(data_from_gpu=False)
    parser.set_defaults(identity_initialize=False)
    parser.set_defaults(allow_nonzero=False)
    parser.set_defaults(identity_output_layer=True)
    parser.set_defaults(less_aggresive_ini=False)
    parser.set_defaults(train_res=False)
    parser.set_defaults(intersection=True)
    parser.set_defaults(mean_estimator_memory_efficient=False)
    parser.set_defaults(efficient_trace=False)
    parser.set_defaults(collect_loop_statistic=False)
    parser.set_defaults(tiled_training=False)
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
    parser.set_defaults(random_target_channel=False)
    parser.set_defaults(schedule_scale_after_freeze=False)
    parser.set_defaults(every_nth_stratified=False)
    parser.set_defaults(aux_plus_manual_features=False)
    parser.set_defaults(use_manual_index=False)
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
    parser.set_defaults(train_with_random_rotation=False)
    parser.set_defaults(temporal_texture_buffer=False)
    parser.set_defaults(use_dataroot=True)
    parser.set_defaults(feature_size_only=False)
    parser.set_defaults(automatic_subsample=False)
    parser.set_defaults(automate_raymarching_def=False)
    parser.set_defaults(chron_order=False)
    parser.set_defaults(def_loop_log_last=False)
    parser.set_defaults(log_only_return_def_raymarching=True)
    parser.set_defaults(train_temporal_seq=False)
    parser.set_defaults(temporal_discrim_only=False)
    parser.set_defaults(log_getitem=True)
    parser.set_defaults(spatial_GAN=True)
    parser.set_defaults(interval_sample_square=False)
    parser.set_defaults(use_validation=False)
    parser.set_defaults(random_switch_label=False)
    parser.set_defaults(boids_single_step_metric=False)
    parser.set_defaults(boids_seq_metric=False)
    parser.set_defaults(analyze_nn_discontinuity=False)
    parser.set_defaults(optimize_input=False)
    parser.set_defaults(boost_ch_from_finetune=True)
    parser.set_defaults(learn_loss_proxy=False)
    parser.set_defaults(optimize_target_img=False)
    parser.set_defaults(list_parameter_idx=False)
    parser.set_defaults(loss_proxy_encourage_0=False)
    parser.set_defaults(collect_inference_tensor=False)
    parser.set_defaults(render_no_video=False)
    parser.set_defaults(feed_dict_optimize_input=False)
    parser.set_defaults(finite_diff=False)
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
    delattr(new_args, 'dataroot')
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
    delattr(new_args, 'render_fix_temporal_sample')
    delattr(new_args, 'render_zero_spatial_sample')
    delattr(new_args, 'render_fov')
    delattr(new_args, 'zero_out_sparsity_vec')
    delattr(new_args, 'sparsity_vec_histogram')
    delattr(new_args, 'write_summary')
    delattr(new_args, 'analyze_channel')
    delattr(new_args, 'bad_example_base_dir')
    delattr(new_args, 'analyze_current_only')
    delattr(new_args, 'save_intermediate_epoch')
    delattr(new_args, 'render_temporal_texture')
    delattr(new_args, 'inference_seq_len')
    delattr(new_args, 'repeat_timing')
    delattr(new_args, 'boids_single_step_metric')
    delattr(new_args, 'analyze_nn_discontinuity')
    delattr(new_args, 'optimize_input')
    delattr(new_args, 'optimize_target_img')
    delattr(new_args, 'target_img')
    delattr(new_args, 'list_parameter_idx')
    delattr(new_args, 'collect_inference_tensor')
    delattr(new_args, 'optimize_nn_input_feature_space')
    delattr(new_args, 'compiler_problem_idx')
    delattr(new_args, 'render_no_video')
    delattr(new_args, 'render_dirname')
    delattr(new_args, 'render_tile_start')
    delattr(new_args, 'feed_dict_optimize_input')
    delattr(new_args, 'finite_diff')
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
            for key in deprecated_options:
                if key in option_str:
                    idx = option_str.index(key)
                    try:
                        next_sep = option_str.index(',', idx)
                    except ValueError:
                        next_sep = option_str.index(')', idx) - 2
                        idx -= 2
                    option_str = option_str.replace(option_str[idx:next_sep+2], '')
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
        
    global dilation_threshold
    dilation_threshold = args.dilation_threshold
    global padding_offset
    if args.model_arch == 'dilated':
        padding_offset = 4 * args.dilation_threshold
    else:
        padding_offset = 0
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
    
        
    if args.train_with_random_rotation:
        rotate = tf.placeholder(tf.int32)
        flip = tf.placeholder(tf.int32)
    else:
        rotate = 0
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
        if args.model_arch == 'dilated':
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
        
    if args.specified_ind != '':
        my_specified_ind_file = os.path.join(args.name, 'specified_ind.npy')
        specified_ind = np.load(args.specified_ind)
        if not os.path.exists(my_specified_ind_file):
            shutil.copyfile(args.specified_ind, my_specified_ind_file)
        else:
            my_ind = np.load(my_specified_ind_file)
            assert np.allclose(my_ind, specified_ind)
    else:
        specified_ind = None

    if args.use_dataroot:
        if not args.shader_name.startswith('boids'):
            output_names, val_img_names, map_names, val_map_names, grad_names, val_grad_names, add_names, val_add_names, validate_img_names = prepare_data_root(args.dataroot, additional_input=args.additional_input)
            if args.test_training:
                val_img_names = output_names
                val_map_names = map_names
                val_grad_names = grad_names
                val_add_names = add_names
    else:
        args.write_summary = False

    
    render_sigma = [args.render_sigma, args.render_sigma, 0]
    
    if args.augment_camera_periodic_range != '':
        augment_camera_pos = True
        camera_periodic_range = []
        for val in args.augment_camera_periodic_range.split(','):
            if val == 'pi':
                camera_periodic_range.append(np.pi)
            elif val == '2pi':
                camera_periodic_range.append(2 * np.pi)
            else:
                camera_periodic_range.append(float(val))
        camera_periodic_range = np.array(camera_periodic_range)
    else:
        augment_camera_pos = False

    if args.geometry == 'texture_approximate_10f':
        output_nc = 34
    else:
        output_nc = 3
    target_nc = output_nc
        
    

    if (args.tiled_training or args.tile_only) and (not inference_entire_img_valid):
        output_pl_w = args.tiled_w
        output_pl_h = args.tiled_h
    else:
        output_pl_w = args.input_w
        output_pl_h = args.input_h
    if args.stratified_sample_higher_res and args.is_train:
        output_pl_w *= 2
        output_pl_h *= 2
        
    target_pl_w = output_pl_w
    target_pl_h = output_pl_h
        
    if args.learn_loss_proxy and args.proxy_loss_type == 'from_data':
        if args.is_train:
            gt_loss_train = np.load(os.path.join(args.dataroot, 'train_loss.npy'))
            gt_loss_val = np.load(os.path.join(args.dataroot, 'validate_loss.npy'))
            
        else:
            if args.test_training:
                # try not to use test_training mode since enumerate all training combination can be super slow...
                gt_loss_train = np.load(os.path.join(args.dataroot, 'train_loss.npy'))
            else:
                gt_loss_train = np.load(os.path.join(args.dataroot, 'test_loss.npy'))
        
        output_pl_w = gt_loss_train.shape[3]
        output_pl_h = gt_loss_train.shape[2]
        args.use_validation = True
        output_nc = 1
        
    if args.render_only:
        args.use_queue = False
        print('setting use_queue to False')
        
    if args.train_temporal_seq:
        if (args.is_train or args.collect_validate_loss) and not args.geometry.startswith('boids'):
            # only require queue for 2D case
            # can release this assertion if we complete the logic of using feed_dict
            assert args.use_queue
        else:
            # at inference, we're probably inferencing on full res img, so not efficient to build the seq generator into a giant graph, is better to inference one img a step
            
            args.use_queue = False
        
    if args.use_dataroot:
        if not args.shader_name.startswith('boids'):
            if args.collect_validate_loss:
                assert args.test_training
                camera_pos_vals = np.load(os.path.join(args.dataroot, 'validate.npy'))
                time_vals = np.load(os.path.join(args.dataroot, 'validate_time.npy'))
                if args.tile_only:
                    tile_start_vals = np.load(os.path.join(args.dataroot, 'validate_start.npy'))
            elif args.is_train or args.test_training:
                camera_pos_vals = np.load(os.path.join(args.dataroot, 'train.npy'))
                time_vals = np.load(os.path.join(args.dataroot, 'train_time.npy'))
                if args.tile_only:
                    tile_start_vals = np.load(os.path.join(args.dataroot, 'train_start.npy'))
            else:
                if not args.temporal_texture_buffer:
                    camera_pos_vals = np.concatenate((
                                        np.load(os.path.join(args.dataroot, 'test_close.npy')),
                                        np.load(os.path.join(args.dataroot, 'test_far.npy')),
                                        np.load(os.path.join(args.dataroot, 'test_middle.npy'))
                                        ), axis=0)
                else:
                    camera_pos_vals = np.load(os.path.join(args.dataroot, 'test.npy'))
                time_vals = np.load(os.path.join(args.dataroot, 'test_time.npy'))
                
            if args.use_validation:
                validate_camera_vals = np.load(os.path.join(args.dataroot, 'validate.npy'))
                validate_time_vals = np.load(os.path.join(args.dataroot, 'validate_time.npy'))
        else:
            if args.is_train or args.test_training:
                training_datas = glob.glob(os.path.join(args.dataroot, 'train_ground*.npy'))
                camera_pos_vals_pool = [None] * len(training_datas)
                for i in range(len(training_datas)):
                    camera_pos_vals_pool[i] = np.load(training_datas[i])
                camera_pos_vals = camera_pos_vals_pool[0]
                if args.use_validation:
                    validate_camera_pos = np.load(os.path.join(args.dataroot, 'validate_ground.npy'))
                #camera_pos_vals = np.load(os.path.join(args.dataroot, 'train_ground.npy'))
            else:
                camera_pos_vals = np.load(os.path.join(args.dataroot, 'test_ground.npy'))
    else:
        if len(args.camera_pos_file):
            camera_pos_vals = np.load(args.camera_pos_file)[:args.camera_pos_len]
        else:
            camera_pos_vals = np.random.random([args.camera_pos_len, 6])
        time_vals = np.zeros(camera_pos_vals.shape[0])
        
    current_ind = tf.constant(0)
    
    ncameras = None
    if not args.shader_name.startswith('boids'):
        if args.learn_loss_proxy and args.proxy_loss_type == 'from_data':
            # if loss proxy gt is read from file, we will enumearate each possible pair for one epoch
            nexamples = time_vals.shape[0] ** 2
            ncameras = time_vals.shape[0]
        else:
            nexamples = time_vals.shape[0]
    else:
        
        #min_sample_time = 2
        #max_sample_time = 64
        min_sample_time = args.min_sample_time
        max_sample_time = args.max_sample_time
        if args.is_train:
            nexamples = args.niters
        else:
            #nexamples = camera_pos_vals.shape[0] // (max_sample_time + 1)
            #nexamples *= (max_sample_time - min_sample_time)
            nexamples = camera_pos_vals.shape[0]
            
        sample_range = np.arange(min_sample_time, max_sample_time+1)
        
        if args.interval_sample_square:
            sample_p = 1 / sample_range ** 2
        elif args.interval_sample_geometric > 1:
            sample_p  = 1 / args.interval_sample_geometric ** sample_range
        else:
            sample_p = 1 / sample_range
        sample_p /= np.sum(sample_p)
        
    texture_maps = []
    
    if args.shader_name.startswith('fluid') and args.use_queue:
        # the only case when texture map also needs to be read from queue
        pass
    else:
        if args.shader_name.startswith('boids'):
            if args.train_temporal_seq and args.is_train:
                texture_maps = None
            else:
                texture_maps = tf.placeholder(dtype, (args.batch_size, args.n_boids, 4))
        elif args.texture_maps != '':
            combined_texture_maps = np.load(args.texture_maps)
            if not args.temporal_texture_buffer:
                texture_maps = []
                for i in range(combined_texture_maps.shape[0]):
                    texture_maps.append(tf.convert_to_tensor(combined_texture_maps[i], dtype=dtype))
         
    ngts = args.temporal_seq_length + args.nframes_temporal_gen - 1
        
    target_pl = None
    if not args.use_queue:
        
        if args.shader_name.startswith('boids'):
            # if train_temporal seq
            # output_pl has size batch_size * (nframes_temporal_gen + 1)
            # the first batch_size * nframes_temporal_gen are used to bootstrap the temporal sequence
            # the last batch_size is used to regulate inference when starting point is a gt texture
            if args.train_temporal_seq:
                if args.is_train:
                    output_pl = tf.placeholder(tf.float32, shape=[None, ngts + 1, args.n_boids, 4])
                else:
                    output_pl = tf.placeholder(tf.float32, shape=[None, args.n_boids, 4])
            else:
                output_pl = tf.placeholder(tf.float32, shape=[None, args.n_boids, 4])
        else:
            output_pl = tf.placeholder(tf.float32, shape=[None, output_pl_h, output_pl_w, output_nc])
            
        if args.learn_loss_proxy:
            if args.optimize_input:
                target_pl = np.expand_dims(np.float32(cv2.imread(args.target_img, -1)) / 255.0, 0)
                if args.finite_diff:
                    target_pl = np.tile(target_pl, [8, 1, 1, 1])
            else:
                target_pl = tf.placeholder(tf.float32, shape=[None, target_pl_h, target_pl_w, target_nc])
            output_nc = 1
        
        if args.train_temporal_seq:
            if not args.geometry.startswith('boids'):
                # 2D case
                input_pl_h = output_pl_h
                input_pl_w = output_pl_w
                if conv_padding == "VALID":
                    input_pl_h += padding_offset
                    input_pl_w += padding_offset
                previous_input = tf.placeholder(tf.float32, shape=[None, input_pl_h, input_pl_w, 6 * (args.nframes_temporal_gen - 1)])
            else:
                previous_input = tf.placeholder(tf.float32, shape=[None, args.n_boids, 8*(args.nframes_temporal_gen-1)])
    else:
        # TODO: currently use_queue is only valid for the fluid approx shader
        # and for trippy temporal
        # should generatlize the code if need to use the same pipeline on more shaders
        assert args.temporal_texture_buffer or args.train_temporal_seq
        #assert args.geometry == 'texture_approximate_10f'
        if args.collect_validate_loss:
            queue_dir_prefix = 'validate'
        elif args.is_train or args.test_training:
            queue_dir_prefix = 'train'
        else:
            queue_dir_prefix = 'test'
        queue_img_dir = os.path.join(args.dataroot, queue_dir_prefix + '_img')
        queue_npy_dir = os.path.join(args.dataroot, queue_dir_prefix + '_label')
        nexamples = time_vals.shape[0]
        if args.temporal_texture_buffer:
            if not args.is_train:
                nexamples -= 10
                dataset_range = np.arange(0, nexamples, 10).astype('i')
            else:
                if args.shader_name == 'fluid_approximate_3pass_10f':
                    # for training examples, check camera_pos_val, and only inlucde examples with a valid mouse movement
                    # and 1/5 of the following no mouse frames to a mouse movement sequence
                    # this lowers the fraction of no mouse examples and might better train mouse movement
                    # plus, this can reduce the number of uninteresting dark frames
                    seq_start = None
                    seq_end = None
                    seq_len = None
                    no_mouse_frac = 0.2
                    dataset_range = np.arange(0)
                    for frame_idx in range(camera_pos_vals.shape[0]):
                        if seq_start is None and camera_pos_vals[frame_idx, 2] > 1 and camera_pos_vals[frame_idx, 5] > 1:
                            seq_start = frame_idx
                            if seq_len is not None and seq_end is not None:
                                if frame_idx - seq_end <= no_mouse_frac * seq_len:
                                    dataset_range = np.concatenate((dataset_range, np.arange(seq_end-seq_len, frame_idx)))
                                else:
                                    dataset_range = np.concatenate((dataset_range, np.arange(seq_end-seq_len, seq_end+int(no_mouse_frac * seq_len))))
                                seq_len = None
                                seq_end = None
                        if seq_start is not None and (camera_pos_vals[frame_idx, 2] <= 1 or camera_pos_vals[frame_idx, 5] <= 1):
                            seq_end = frame_idx
                            seq_len = seq_end - seq_start
                            seq_start = None

                    if seq_len is not None and seq_end is not None:
                        if camera_pos_vals.shape[0] - seq_end - 10 <= no_mouse_frac * seq_len:
                            dataset_range = np.concatenate((dataset_range, np.arange(seq_end-seq_len, camera_pos_vals.shape[0] - 10)))
                        else:
                            dataset_range = np.concatenate((dataset_range, np.arange(seq_end-seq_len, seq_end+int(no_mouse_frac * seq_len))))

                    dataset_range = dataset_range.astype('i')
                    nexamples = dataset_range.shape[0]
                else:
                    dataset_range = np.arange(600, nexamples-12).astype('i')
                    nexamples -= 612

        else:
            dataset_range = np.arange(nexamples).astype('i')
            ngts = args.temporal_seq_length + args.nframes_temporal_gen - 1

        dataset = tf.data.Dataset.from_tensor_slices(dataset_range)
        if args.is_train:
            dataset = dataset.shuffle(nexamples)
        
        def load_image(ind):
            # TODO: specific to fluid approx and trippy, generalize if we need it on other shaders
            if args.temporal_texture_buffer:
                if args.is_train:
                    base_idx = 0
                else:
                    base_idx = 9000
            
                output_arr = np.empty([640, 960, 34])
                for i in range(10):
                    imgfile = os.path.join(queue_img_dir, 'test_mouse_seq%05d.png' % (ind + i + 1 + base_idx))
                    #img = skimage.io.imread(imgfile)
                    img = np.float32(cv2.imread(imgfile, -1)) / 255.0
                    output_arr[:, :, i*3:i*3+3] = img
                init_npy = np.transpose(np.load(os.path.join(queue_npy_dir, 'test_mouse_seq%05d.npy' % (ind + base_idx))), (2, 0, 1))
                final_npy = np.load(os.path.join(queue_npy_dir, 'test_mouse_seq%05d.npy' % (ind + 10 + base_idx)))
                # TODO: final_npy also contains NON-CLIPPED RGB color for last frame
                # should use the CLIPPED RGB color in png to keep the same format as previous frames, or use the more accurate RGB from npy file?
                output_arr[:, :, -4:] = final_npy[:, :, 3:]

                # e.g. if idx = 1
                # camera_pos_val[1] is last mouse for 1st frame
                # and we collect 10 next mouse pos
                camera_val = np.reshape(camera_pos_vals[ind:ind+11, 3:], 33)

                if args.shader_name in ['fluid_approximate_3pass_no_mouse_perlin', 'fluid_approximate_sin']:
                    # also load previous 1 frame and frame 11 for temporal seq discrim
                    add_gt = np.empty([640, 960, 6])
                    imgfile = os.path.join(queue_img_dir, 'test_mouse_seq%05d.png' % (ind + base_idx))
                    img = np.float32(cv2.imread(imgfile, -1)) / 255.0
                    add_gt[:, :, :3] = img
                    imgfile = os.path.join(queue_img_dir, 'test_mouse_seq%05d.png' % (ind + base_idx + 11))
                    img = np.float32(cv2.imread(imgfile, -1)) / 255.0
                    add_gt[:, :, 3:] = img
                    return ind, output_arr, init_npy, camera_val, time_vals[ind+1], add_gt
                else:
                    return ind, output_arr, init_npy, camera_val, time_vals[ind+1]
            elif args.train_temporal_seq:
                # case for train_temporal_seq
                # currently supporting trippy only
                base_idx = 0
                output_arr = np.empty([output_pl_h, output_pl_w, output_nc * ngts])
                
                if args.collect_validate_loss:
                    img_prefix = 'validate'
                elif args.is_train or args.test_training:
                    img_prefix = 'train'
                else:
                    img_prefix = 'test'
                
                for i in range(ngts):
                    imgfile = os.path.join(queue_img_dir, '%s_ground%d%05d.png' % (img_prefix, i, ind))
                    img = np.float32(cv2.imread(imgfile, -1)) / 255.0
                    output_arr[:, :, i*3:i*3+3] = img
                time_val_seq = np.empty(ngts)
                # generate trace for all frames in the seq
                # NOT just the frames that will be generated by the generator
                for i in range(ngts):
                    time_val_seq[i] = time_vals[ind] + float(i) / 30
                if args.is_train or args.collect_validate_loss:
                    return ind, output_arr, camera_pos_vals[ind], time_val_seq, tile_start_vals[ind]
                else:
                    return ind, output_arr, camera_pos_vals[ind], time_val_seq, np.array([-16., -16.])
            else:
                raise
        
        if args.shader_name in ['fluid_approximate_3pass_10f', 'fluid_approximate']:
            dataset = dataset.map(lambda x: tf.contrib.eager.py_func(func=load_image, inp=[x], Tout=(tf.int32, dtype, dtype, dtype, dtype)), num_parallel_calls=8)
            # TODO: total hack, should generalize later
            shapes = (tf.TensorShape([]), tf.TensorShape([output_pl_h, output_pl_w, 34]), tf.TensorShape([7, output_pl_h, output_pl_w]), tf.TensorShape([33]), tf.TensorShape([]))
        elif args.shader_name in ['fluid_approximate_3pass_no_mouse_perlin', 'fluid_approximate_sin']:
            dataset = dataset.map(lambda x: tf.contrib.eager.py_func(func=load_image, inp=[x], Tout=(tf.int32, dtype, dtype, dtype, dtype, dtype)), num_parallel_calls=8)
            # TODO: total hack, should generalize later
            shapes = (tf.TensorShape([]), tf.TensorShape([output_pl_h, output_pl_w, 34]), tf.TensorShape([7, output_pl_h, output_pl_w]), tf.TensorShape([33]), tf.TensorShape([]), tf.TensorShape([output_pl_h, output_pl_w, 6]))
        #elif args.shader_name.startswith('trippy') or args.shader_name.startswith('mandelbulb'):
        else:
            dataset = dataset.map(lambda x: tf.contrib.eager.py_func(func=load_image, inp=[x], Tout=(tf.int32, dtype, dtype, dtype, dtype)), num_parallel_calls=8)
            shapes = (tf.TensorShape([]), tf.TensorShape([output_pl_h, output_pl_w, output_nc * ngts]), tf.TensorShape([6]), tf.TensorShape([ngts]), tf.TensorShape([2]))
        #else:
        #    raise
            
        dataset = dataset.apply(tf.contrib.data.assert_element_shape(shapes))
            
        # TODO: for simplificy, only allow batchsize=1 for now
        assert args.batch_size == 1
        assert not args.use_batch
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(buffer_size=8)
        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        train_iterator = iterator.make_initializer(dataset)

        if args.shader_name in ['fluid_approximate_3pass_10f', 'fluid_approximate', 'fluid_approximate_3pass_no_mouse_perlin', 'fluid_approximate_sin']:
            if args.shader_name in ['fluid_approximate_3pass_10f', 'fluid_approximate']:
                current_ind, output_pl, texture_maps, camera_pos, shader_time = iterator.get_next()
            else:
                current_ind, output_pl, texture_maps, camera_pos, shader_time, additional_gt_frames = iterator.get_next()
            texture_maps = texture_maps[0]
            camera_pos = tf.expand_dims(camera_pos[0], axis=1)
            shader_time = tf.expand_dims(shader_time[0], axis=0)
        #elif args.shader_name.startswith('trippy') or args.shader_name.startswith('mandelbulb'):
        else:
            current_ind, output_pl, camera_pos, shader_time, tile_start = iterator.get_next()
            camera_pos = tf.tile(tf.expand_dims(camera_pos[0], axis=1), [1, ngts])
            output_pl = tf.expand_dims(output_pl[0], 0)
            shader_time = shader_time[0]
            h_start = tf.tile(tf.expand_dims(tile_start[0, 0] - padding_offset / 2, axis=0), (ngts,))
            w_start = tf.tile(tf.expand_dims(tile_start[0, 1] - padding_offset / 2, axis=0), (ngts,))
        #else:
        #    raise
        
        

    camera_pos_var = None
    shader_time_var = None
    
    if not args.use_queue:
        camera_pos = None
        if not args.shader_name.startswith('boids'):
            if args.geometry != 'texture_approximate_10f':
                #camera_pos = tf.placeholder(dtype, shape=[6, args.batch_size])
                camera_pos_len = 6
            else:
                #camera_pos = tf.placeholder(dtype, shape=[33, args.batch_size])
                camera_pos_len = 33
        if args.optimize_input:
            assert args.use_dataroot
            # normalize camera pos and shader time according to training data
            camera_pos_training_pool = numpy.load(os.path.join(args.dataroot, 'train.npy'))
            camera_pos_bias = np.expand_dims(-np.min(camera_pos_training_pool, 0), 1)
            
            camera_pos_scale_raw = np.max(camera_pos_training_pool, 0) - np.min(camera_pos_training_pool, 0)
            camera_pos_scale = numpy.ones(camera_pos_scale_raw.shape)
            camera_pos_scale[camera_pos_scale_raw.nonzero()] = camera_pos_scale_raw[camera_pos_scale_raw.nonzero()]
            camera_pos_scale = np.expand_dims(camera_pos_scale, 1)
            
            shader_time_training_pool = numpy.load(os.path.join(args.dataroot, 'train_time.npy'))
            shader_time_bias = -numpy.min(shader_time_training_pool)
            shader_time_scale = np.max(shader_time_training_pool) + shader_time_bias
            if shader_time_scale == 0:
                shader_time_scale = 1
            
            #camera_pos_var = tf.Variable(np.zeros([camera_pos_len, args.batch_size]), dtype=dtype)
            if args.feed_dict_optimize_input:
                camera_pos_var = tf.placeholder(dtype, shape=[camera_pos_len, args.batch_size])
                shader_time_var = tf.placeholder(dtype, shape=args.batch_size)
            else:
                camera_pos_var = tf.Variable((camera_pos_training_pool[:args.batch_size].transpose() + camera_pos_bias) / camera_pos_scale, dtype=dtype)
                shader_time_var = tf.Variable(np.zeros(args.batch_size), dtype=dtype)
                
                
            if args.finite_diff:
                finite_h = 1e-2
                # for computational efficientcy we only add 7 more examples, finite diff will be computed as
                # (f(x + h) - f(x)) / h
                
                shader_time_finite = tf.concat((tf.tile(shader_time_var, [7]), shader_time_var + finite_h), 0)
                
                camera_stacks = [camera_pos_var]                            
                current_finite = np.zeros(6)    
                for i in range(6):
                    current_finite[:] = 0
                    current_finite[i] = finite_h
                    camera_stacks.append(camera_pos_var + np.expand_dims(current_finite, 1))
                camera_stacks.append(camera_pos_var)
                
                camera_pos_finite = tf.concat(camera_stacks, 1)
            else:
                shader_time_finite = shader_time_var
                camera_pos_finite = camera_pos_var
                
            camera_pos = camera_pos_finite * camera_pos_scale - camera_pos_bias
            shader_time = shader_time_finite * shader_time_scale - shader_time_bias
        else:
            camera_pos = tf.placeholder(dtype, shape=[camera_pos_len, args.batch_size])
            shader_time = tf.placeholder(dtype, shape=args.batch_size)
    if args.additional_input:
        additional_input_pl = tf.placeholder(dtype, shape=[None, output_pl_h, output_pl_w, 1])
    camera_pos_velocity = None

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
        elif args.train_temporal_seq:
            if args.is_train or args.collect_validate_loss:
                if not args.geometry.startswith('boids'):
                    shader_samples = ngts
                else:
                    shader_samples = args.batch_size
            else:
                shader_samples = args.batch_size
        else:
            shader_samples = args.batch_size

        #print("sample count", shader_samples)
        feature_w = []
        color_inds = []
        texture_inds = []
        color_scale = []
        if args.tiled_training or args.tile_only:
            if args.optimize_input or args.collect_inference_tensor:
                h_start = tf.constant([- padding_offset / 2])
                w_start = tf.constant([- padding_offset / 2])
            elif not args.use_queue:
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
        
        input_feature_pl = []
        input_to_shader = []
        trace_features = []
        input_feature_scale_bias = []
        if args.optimize_nn_input_feature_space > 0:
            for i in range(args.optimize_nn_input_feature_space):
                input_feature_pl.append(tf.Variable(np.random.rand(args.batch_size, h_offset, w_offset), dtype=dtype))

        #if args.geometry.startswith('boids'):
        if False:
            debug = [{texture_maps: camera_pos_vals[:args.batch_size], shader_time: np.ones(args.batch_size)}]
        else:
            debug = [[camera_pos_var, shader_time_var] + input_feature_pl]
            
        


        def generate_input_to_network_wrapper():
            def func(texture_maps_input):
            
                return get_tensors(args.dataroot, args.name, camera_pos, shader_time, output_type, shader_samples, shader_name=args.shader_name, geometry=args.geometry, feature_w=feature_w, color_inds=color_inds, intersection=args.intersection, manual_features_only=args.manual_features_only, aux_plus_manual_features=args.aux_plus_manual_features, efficient_trace=args.efficient_trace, collect_loop_statistic=args.collect_loop_statistic, h_start=h_start, h_offset=h_offset, w_start=w_start, w_offset=w_offset, samples=feed_samples, fov=args.fov, camera_pos_velocity=camera_pos_velocity, first_last_only=args.first_last_only, last_only=args.last_only, subsample_loops=args.subsample_loops, last_n=args.last_n, first_n=args.first_n, first_n_no_last=args.first_n_no_last, mean_var_only=args.mean_var_only, zero_samples=zero_samples, render_fix_spatial_sample=args.render_fix_spatial_sample, render_fix_temporal_sample=args.render_fix_temporal_sample, render_zero_spatial_sample=args.render_zero_spatial_sample, spatial_samples=spatial_samples, temporal_samples=temporal_samples, every_nth=args.every_nth, every_nth_stratified=args.every_nth_stratified, one_hop_parent=args.one_hop_parent, target_idx=target_idx, use_manual_index=args.use_manual_index, manual_index_file=args.manual_index_file, additional_features=args.additional_features, ignore_last_n_scale=args.ignore_last_n_scale, include_noise_feature=args.include_noise_feature, crop_h=args.crop_h, crop_w=args.crop_w, no_noise_feature=args.no_noise_feature, relax_clipping=args.relax_clipping, render_sigma=render_sigma, same_sample_all_pix=args.same_sample_all_pix, stratified_sample_higher_res=args.stratified_sample_higher_res, samples_int=samples_int, texture_maps=texture_maps_input, partial_trace=args.partial_trace, use_lstm=args.use_lstm, lstm_nfeatures_per_group=args.lstm_nfeatures_per_group, rotate=rotate, flip=flip, use_dataroot=args.use_dataroot, automatic_subsample=args.automatic_subsample, automate_raymarching_def=args.automate_raymarching_def, chron_order=args.chron_order, def_loop_log_last=args.def_loop_log_last, temporal_texture_buffer=args.temporal_texture_buffer, texture_inds=texture_inds, log_only_return_def_raymarching=args.log_only_return_def_raymarching, SELECT_FEATURE_THRE=args.SELECT_FEATURE_THRE, n_boids=args.n_boids, log_getitem=args.log_getitem, debug=debug, color_scale=color_scale, parallel_stack=not args.optimize_input, compiler_problem_idx=args.compiler_problem_idx, input_feature_pl=input_feature_pl, input_to_shader=input_to_shader, trace_features=trace_features, input_feature_scale_bias=input_feature_scale_bias, finite_diff=args.finite_diff, feature_normalize_lo_pct=args.feature_normalize_lo_pct, get_col_aux_inds=args.get_col_aux_inds, specified_ind=specified_ind, write_file=args.overwrite_option_file, alt_dir=args.test_output_dir)
            
            return func
            
        generate_input_to_network = generate_input_to_network_wrapper()
        if not (args.geometry.startswith('boids') and args.train_temporal_seq):
            input_to_network = generate_input_to_network(texture_maps)
        else:
            input_to_network = None
            
        if args.get_col_aux_inds:
            return

        if args.feature_size_only:
            print('feature size: ', int(input_to_network.shape[-1]))
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
        elif args.geometry.startswith('boids'):
            output = output_pl
            if not args.mean_estimator:
                np.save('color_scale.npy', color_scale)
            else:
                color_scale = np.load('color_scale.npy')
                input_to_network += color_scale[0]
                input_to_network *= color_scale[1]
            #feature_scale = np.load(os.path.join(args.dataroot, 'feature_scale.npy'))
            #feature_bias = np.load(os.path.join(args.dataroot, 'feature_bias.npy'))
            output += color_scale[0]
            output *= color_scale[1]
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
            print('concatenating additional input')
            input_to_network = tf.concat((input_to_network, additional_input), axis=3)

        
        if len(color_inds) == 3:
            color_inds = color_inds[::-1]
        
        if input_to_network is not None:
            args.input_nc = int(input_to_network.shape[-1])
            debug_input = input_to_network
        
            
    def feature_reduction_layer(input_to_network, _replace_normalize_weights=None, ndims=2):
        with tf.variable_scope("feature_reduction"):
            if (args.regularizer_scale > 0 or args.L2_regularizer_scale > 0):
                regularizer = slim.l1_l2_regularizer(scale_l1=args.regularizer_scale, scale_l2=args.L2_regularizer_scale)
            else:
                regularizer = None

            actual_nfeatures = args.input_nc
            
            if args.feature_reduction_ch > 0:
                actual_feature_reduction_ch = args.feature_reduction_ch
            else:
                actual_feature_reduction_ch = args.initial_layer_channels

            if ndims == 2:
                w_shape = [1, 1, actual_nfeatures, actual_feature_reduction_ch]
                conv = tf.nn.conv2d
                strides = [1, 1, 1, 1]
            else:
                w_shape = [1, actual_nfeatures, actual_feature_reduction_ch]
                conv = tf.nn.conv1d
                strides = 1
            weights = tf.get_variable('w0', w_shape, initializer=tf.contrib.layers.xavier_initializer() if not args.identity_initialize else identity_initializer(color_inds, ndims=ndims), regularizer=regularizer)

            weights_to_input = weights

            reduced_feat = conv(input_to_network, weights_to_input, strides, "SAME")

            if args.initial_layer_channels <= actual_conv_channel:
                ini_id = True
            else:
                ini_id = False
            
        return reduced_feat
    
    if args.geometry.startswith('boids'):
        ndims = 1
    else:
        ndims = 2

    #with tf.control_dependencies([input_to_network]):
    with tf.variable_scope("generator"):
        if args.debug_mode and args.mean_estimator:
            with tf.variable_scope("shader"):
                network = tf.reduce_mean(input_to_network, axis=0, keep_dims=True)
            regularizer_loss = 0
            sparsity_loss = 0
            sparsity_schedule = None
            target_channel_schedule = None
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
            sparsity_schedule = None
            target_channel_schedule = None
            if args.feature_sparsity_vec:
                target_channel = args.sparsity_target_channel
                target_channel_max = target_channel
                if args.input_nc > target_channel_max:
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

            elif args.analyze_channel:
                sparsity_vec = tf.ones(args.input_nc, dtype=dtype)
                input_to_network = input_to_network * sparsity_vec
            
            elif args.boost_channels != '':
                
                mask_vec = np.ones(args.input_nc)
                
                selected_channels = [int(k) for k in args.boost_channels.split(',')]
                valid_inds = np.load(os.path.join(args.name, 'valid_inds.npy')).tolist()
                
                for ch in selected_channels:
                    if ch in valid_inds:
                        mask_vec[valid_inds.index(ch)] = 0
                        
                sparsity_vec = tf.Variable(mask_vec, dtype=dtype)
                boost_mask = sparsity_vec * mask_vec + (1 - mask_vec)
                input_to_network = input_to_network  * boost_mask
                sparsity_loss = tf.reduce_mean(tf.abs(sparsity_vec))
                
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

            regularizer_loss = tf.constant(0.0)
            actual_initial_layer_channels = args.initial_layer_channels
            regularizer = None
            
            feature_reduction_tensor = None
            if not args.train_temporal_seq:
                if not args.use_lstm:
                    input_to_network = feature_reduction_layer(input_to_network, _replace_normalize_weights=replace_normalize_weights, ndims=ndims)
                    feature_reduction_tensor = input_to_network
                else:
                    ngroups = input_to_network.shape[-1] // args.lstm_nfeatures_per_group
                    orig_shape = input_to_network.shape
                    input_to_network = tf.reshape(input_to_network, [-1, ngroups, args.lstm_nfeatures_per_group])
                    rnn_cell = tf.nn.rnn_cell.LSTMCell(args.initial_layer_channels)
                    #layer = tf.keras.layers.RNN(rnn_cell)
                    #input_to_network = layer(input_to_network)
                    _, states = tf.nn.dynamic_rnn(rnn_cell, input_to_network, dtype=dtype)
                    input_to_network = tf.reshape(states.h, [orig_shape[0], orig_shape[1], orig_shape[2], args.initial_layer_channels])

                
                
                if args.learn_loss_proxy:
                    if padding_offset > 0:
                        target_input = tf.pad(target_pl, [[0, 0], [padding_offset // 2, padding_offset // 2], [padding_offset // 2, padding_offset // 2], [0, 0]], 'REFLECT')
                    else:
                        target_input = target_pl
                    input_to_network = tf.concat((input_to_network, target_input), 3)

                reduced_dim_feature = input_to_network
                
                if args.add_initial_layers:
                    for nlayer in range(3):
                        input_to_network = slim.conv2d(input_to_network, actual_initial_layer_channels, [1, 1], rate=1, activation_fn=lrelu, normalizer_fn=nm, weights_initializer=identity_initializer(allow_map_to_less=True), scope='initial_'+str(nlayer), weights_regularizer=regularizer, padding=conv_padding)            

                if args.geometry.startswith('boids'):
                    # 1D case
                    input_to_network = tf.reshape(input_to_network, [args.batch_size, -1])
                    stacked_channels = [256] * args.fc_nlayer + [args.n_boids * 4]
                    network = slim.stack(input_to_network, slim.fully_connected, stacked_channels, activation_fn=lrelu)
                    network = tf.reshape(network, [args.batch_size, args.n_boids, 4])
                    
                    #network /= feature_scale[color_inds]
                    #network -= feature_bias[color_inds]
                    
                else:
                    if args.model_arch == 'dilated':
                        network=build(input_to_network, ini_id, regularizer_scale=args.regularizer_scale, final_layer_channels=args.final_layer_channels, identity_initialize=args.identity_initialize, output_nc=output_nc)
                    elif args.model_arch == 'vgg':
                        if args.vgg_channels != '':
                            vgg_channels = [int(ch) for ch in args.vgg_channels.split(',')]
                            assert len(vgg_channels) == 5
                        else:
                            vgg_channels = []
                        network = build_vgg(input_to_network, output_nc=output_nc, channels=vgg_channels)
                    else:
                        raise
            else:
                assert not args.use_lstm
                if not args.geometry.startswith('boids'):
                    # 2D case
                    assert args.add_initial_layers
                    if args.is_train or args.collect_validate_loss:
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

                else:
                    # 1D case
                    init_textures = []
                    input_labels = []
                    current_input_ls = []
                    generated_seq = []
                    stacked_channels = [256] * args.fc_nlayer + [args.n_boids * 4]
                    
                    if not args.is_train:
                        with tf.variable_scope("single_frame_generator", reuse=tf.AUTO_REUSE):
                            # at inference, only build compute graph for 1 step, so it can easily generalize to long sequences
                            input_trace = generate_input_to_network(texture_maps)
                            debug_input = input_trace
                            input_label = tf.stack([input_trace[:, :, col] for col in color_inds], 2)
                            
                            args.input_nc = int(input_trace.shape[-1])
                            current_input = feature_reduction_layer(tf.stop_gradient(input_trace), _replace_normalize_weights=replace_normalize_weights, ndims=1)
                            current_input = tf.concat((current_input, previous_input), 2)
                            current_input = tf.reshape(current_input, [args.batch_size, -1])

                            current_output = slim.stack(current_input, slim.fully_connected, stacked_channels, activation_fn=lrelu)
                            network = tf.reshape(current_output, [args.batch_size, args.n_boids, 4])
                    else:
                    
                        output = None

                        for i in range(args.nframes_temporal_gen - 1):

                            # input texture is not normalized, raw data range
                            color_inds = []
                            input_trace = generate_input_to_network(output_pl[:, i])

                            if output is None:
                                output = (output_pl + color_scale[0]) * color_scale[1]                        

                            init_textures.append(output[:, i])

                            # already normalized by feature_scale and feature_bias
                            input_labels.append(tf.stack([input_trace[:, :, col] for col in color_inds], 2))

                            current_input_ls.append(init_textures[-1])
                            current_input_ls.append(input_labels[-1])

                        args.input_nc = int(input_trace.shape[-1])

                        for i in range(args.nframes_temporal_gen - 1, args.temporal_seq_length + args.nframes_temporal_gen - 1):
                            current_input_prev = tf.concat(current_input_ls, 2)

                            with tf.variable_scope("single_frame_generator", reuse=tf.AUTO_REUSE):

                                # init_texture is normalized, so can't directly input to generate_input_to_network
                                if i == args.nframes_temporal_gen - 1:
                                    init_textures.append(output[:, i])
                                    color_inds = []
                                    current_input = generate_input_to_network(output_pl[:, i])
                                else:
                                    init_textures.append(generated_seq[-1])
                                    raw_range_texture = (init_textures[-1] / color_scale[1]) - color_scale[0]
                                    color_inds = []
                                    current_input = generate_input_to_network(tf.stop_gradient(raw_range_texture))

                                input_labels.append(tf.stack([current_input[:, :, col] for col in color_inds], 2))

                                current_input = feature_reduction_layer(tf.stop_gradient(current_input), _replace_normalize_weights=replace_normalize_weights, ndims=1)
                                current_input = tf.concat((current_input, tf.stop_gradient(current_input_prev)), 2)
                                current_input = tf.reshape(current_input, [args.batch_size, -1])

                                current_output = slim.stack(current_input, slim.fully_connected, stacked_channels, activation_fn=lrelu)
                                generated_seq.append(tf.reshape(current_output, [args.batch_size, args.n_boids, 4]))

                                current_input_ls.append(init_textures[-1])
                                current_input_ls.append(input_labels[-1])
                                current_input_ls = current_input_ls[2:]
                        
                            
    if args.optimize_input:
        if padding_offset > 0:
            raw_color = tf.stack([debug_input[:, padding_offset // 2 : -padding_offset // 2, padding_offset // 2 : -padding_offset // 2, ind] for ind in color_inds[::-1]], 3)
        else:
            raw_color = tf.stack([debug_input[:, :, :, ind] for ind in color_inds[::-1]], 3)
            
        return camera_pos_var, shader_time_var, network, raw_color, input_feature_pl, input_to_shader, trace_features, input_feature_scale_bias, feature_reduction_tensor, debug_input
    
    if args.collect_inference_tensor:
        if args.learn_loss_proxy:
            return camera_pos, target_pl, network
        else:
            return camera_pos, network
    
    weight_map = tf.placeholder(tf.float32,shape=[None,None,None])

    if args.learn_loss_proxy:
        if args.proxy_loss_type == 'smoothed_l2':
            # currently using smoothed l2 loss
            # smoothing kernel size equal to network's receptive field
            # preserve dimensionality with reflect padding
            kernel_dim = ((padding_offset - 1) // 2) * 2 + 1
            kernel_activation = np.zeros((kernel_dim, kernel_dim))
            kernel_center = kernel_dim // 2
            kernel_activation[kernel_center, kernel_center] = 1

            kernel_sigma = padding_offset // 2
            kernel = scipy.ndimage.gaussian_filter(kernel_activation, kernel_sigma)
            kernel /= np.sum(kernel)
            kernel = numpy.expand_dims(numpy.expand_dims(kernel, 2), 3)

            diff = (target_pl - output_pl) ** 2
            diff = tf.pad(diff, [[0, 0], [kernel_center, kernel_center], [kernel_center, kernel_center], [0, 0]], 'REFLECT')
            diff = tf.expand_dims(tf.reduce_mean(diff, 3), 3)

            smoothed_predict = tf.nn.conv2d(diff, kernel, [1, 1, 1, 1], 'VALID')
            loss = tf.reduce_mean((network - smoothed_predict) ** 2)
        elif args.proxy_loss_type == 'lpips':
            # using the lpips from elpips directory because custom modification on spatial has been made there
            sys.path += ['../../elpips']
            import elpips
            metric = elpips.Metric(elpips.lpips_vgg(args.batch_size))
            tf_distance = metric.forward(output_pl, target_pl, spatial=True)
            loss = tf.reduce_mean((network - tf_distance) ** 2)
        elif args.proxy_loss_type == 'from_data':
            # simply l2 loss between network generated loss and gt loss in output_pl
            loss = tf.reduce_mean((network - output_pl)) ** 2
        else:
            raise
        
    elif args.l2_loss:
        if args.geometry.startswith('boids') and args.train_temporal_seq:
            # only regulate the first prediction
            if args.is_train:
                loss = tf.reduce_mean((generated_seq[0] - output[:, args.nframes_temporal_gen]) ** 2)
            else:
                loss = tf.reduce_mean((network - output_pl) ** 2)
        elif ((not args.is_train) and (not args.collect_validate_loss)) or (not args.train_temporal_seq):
            if (not args.train_res) or (args.debug_mode and args.mean_estimator):
                diff = network - output
            else:
                input_color = tf.stack([debug_input[..., ind] for ind in color_inds], axis=-1)
                diff = network + input_color - output
                network += input_color

            if args.RGB_norm % 2 != 0:
                diff = tf.abs(diff)
            powered_diff = diff ** args.RGB_norm
            
            if args.temporal_texture_buffer and args.shader_name.startswith('fluid'):
                loss = tf.reduce_mean(powered_diff, (0, 1, 2))
                loss0 = tf.reduce_mean(loss[:-4])
                loss1 = tf.reduce_mean(loss[-4:])
                loss = loss0 + loss1
            else:
                if ndims == 2:
                    loss_per_sample = tf.reduce_mean(powered_diff, (1, 2, 3))
                elif ndims == 1:
                    loss_per_sample = tf.reduce_mean(powered_diff, (1, 2))
                else:
                    raise
                loss = tf.reduce_mean(loss_per_sample)
                #loss = tf.reduce_mean(powered_diff)

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
        if args.train_temporal_seq and (args.is_train or args.collect_validate_loss):
            loss_lpips = 0.0
            for i in range(len(generated_seq)):
                start_ind = args.nframes_temporal_gen - 1 + i
                loss_lpips += lpips_tf.lpips(generated_seq[i], output[:, :, :, 3*start_ind:3*start_ind+3], model='net-lin', net=args.lpips_net)
            loss_lpips /= len(generated_seq)
        elif output_nc == 3:
            loss_lpips = lpips_tf.lpips(network, output, model='net-lin', net=args.lpips_net)
        else:
            loss_lpips = 0
            # a hack tailored for fluid approx app, but should be generalized if needed later
            assert (output_nc - 4) % 3 == 0
            nimages = (output_nc - 4) // 3
            # perceptual error not in last 4 channels, which should be velocity field (only l2 will take care on them)
            for i in range(nimages):
                loss_lpips += lpips_tf.lpips(network[:, :, :, 3*i:3*i+3], output[:, :, :, 3*i:3*i+3], model='net-lin', net=args.lpips_net)
            loss_lpips /= nimages
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
        loss_to_opt = loss + regularizer_loss + sparsity_loss
        gen_loss_GAN = tf.constant(0.0)
        discrim_loss = tf.constant(0.0)
        savers = []
        save_names = []
    elif args.patch_gan_loss:
        loss = loss + regularizer_loss + sparsity_loss
        # descriminator adapted from
        # https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
        def create_discriminator(discrim_inputs, discrim_target, sliced_feat=None, other_target=None, is_temporal=False, ndims=2):
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
            if ndims == 1:
                d_network = tf.reshape(d_network, [args.batch_size, -1])
            #n_ch = 32
            if is_temporal:
                n_ch = args.ndf_temporal
            else:
                n_ch = args.ndf
            
            if ndims == 2:
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
            else:
                assert ndims == 1
                stacked_channels = [n_ch] * n_layers + [1]
                d_network = slim.stack(d_network, slim.fully_connected, stacked_channels, activation_fn=lrelu)

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
            if not args.shader_name.startswith('fluid'):
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
                    # use unconditioned discrim for 1D boids
                    # because gt frames may not correlate with condition
                    # e.g. condition can be computed from a initial pose that is the inference from a previous step
                    # so the actual gt will be different from what we have precomptued based on gt initial pose
                    if ndims == 1:
                        condition_input = [None] * len(condition_input)
                    if args.spatial_GAN:
                        for i in range(len(generated_seq)):
                            start_ind = args.nframes_temporal_gen - 1 + i
                            with tf.name_scope("discriminator_real"):
                                with tf.variable_scope("discriminator_single", reuse=tf.AUTO_REUSE):
                                    if ndims == 2:
                                        predict_real_single.append(create_discriminator(condition_input[i], output[:, :, :, 3*start_ind:3*start_ind+3]))
                                    else:
                                        predict_real_single.append(create_discriminator(condition_input[i], output[:, start_ind+1], ndims=1))
                            with tf.name_scope("discriminator_fake"):
                                with tf.variable_scope("discriminator_single", reuse=tf.AUTO_REUSE):
                                    predict_fake_single.append(create_discriminator(condition_input[i], generated_seq[i], ndims=ndims))
                    
                    for i in range(0, len(generated_seq), args.nframes_temporal_discrim):
                        start_ind = args.nframes_temporal_gen - 1 + i
                        with tf.name_scope("discriminator_real"):
                            with tf.variable_scope("discriminator_seq", reuse=tf.AUTO_REUSE):
                                if ndims == 2:
                                    predict_real_seq.append(create_discriminator(condition_input[i:i+args.nframes_temporal_discrim], output[:, :, :, 3*start_ind:3*start_ind+3*args.nframes_temporal_discrim], is_temporal=True))
                                else:
                                    predict_real_seq.append(create_discriminator(condition_input[i:i+args.nframes_temporal_discrim], output[:, start_ind+1:start_ind+1+args.nframes_temporal_discrim], is_temporal=True, ndims=1))
                        with tf.name_scope("discriminator_fake"):
                            with tf.variable_scope("discriminator_seq", reuse=tf.AUTO_REUSE):
                                predict_fake_seq.append(create_discriminator(condition_input[i:i+args.nframes_temporal_discrim], generated_seq[i:i+args.nframes_temporal_discrim], is_temporal=True, ndims=ndims))
                                
                else:
                    with tf.name_scope("discriminator_real"):
                        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
                            predict_real = create_discriminator(condition_input, output, sliced_feat, network)

                    with tf.name_scope("discriminator_fake"):
                        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
                            predict_fake = create_discriminator(condition_input, network, sliced_feat, output)
            else:
                if args.shader_name in ['fluid_approximate_3pass_10f', 'fluid_approximate']:
                    with tf.name_scope("discriminator_real"):
                        with tf.variable_scope("discriminator"):
                            predict_real = create_discriminator(condition_input, output, sliced_feat, network)

                    with tf.name_scope("discriminator_fake"):
                        with tf.variable_scope("discriminator", reuse=True):
                            predict_fake = create_discriminator(condition_input, network, sliced_feat, output)

                    with tf.name_scope("discriminator_fake_still"):
                        with tf.variable_scope("discriminator", reuse=True):
                            # TODO: just a hack that only works for fluid appxo
                            # another discriminator loss saying that still images are false
                            # use first 3 channels because at some point earlier in the code, color_inds are reversed
                            predict_fake_still = create_discriminator(condition_input, tf.concat((tf.tile(condition_input[:, :, :, :3], (1, 1, 1, 10)), network[:, :,:, -4:]), 3), sliced_feat, output)
                        
                elif args.shader_name in ['fluid_approximate_3pass_no_mouse_perlin', 'fluid_approximate_sin']:
                    # unconditioned single frame discrim
                    nimages = (output_nc - 4) // 3
                    predict_real_single = []
                    predict_fake_single = []
                    if not args.temporal_discrim_only:
                        with tf.name_scope("discriminator_real"):
                            with tf.variable_scope("discriminator_spatial", reuse=tf.AUTO_REUSE):
                                for i in range(nimages):
                                    predict_real_single.append(create_discriminator(None, output[:, :, :, 3*i:3*i+3]))
                        with tf.name_scope("discriminator_fake"):
                            with tf.variable_scope("discriminator_spatial", reuse=tf.AUTO_REUSE):
                                for i in range(nimages):
                                    predict_fake_single.append(create_discriminator(None, network[:, :, :, 3*i:3*i+3]))
                                
                    # unconditions seq discrim (L=3)
                    predict_real_seq = []
                    predict_fake_seq = []
                    if True:
                        with tf.name_scope("discriminator_real"):
                            with tf.variable_scope("discriminator_temporal", reuse=tf.AUTO_REUSE):
                                if False:
                                    # old code that computes temporal discrim for overlapping seq
                                    for i in range(-1, nimages + 1):
                                        if i == -1:
                                            current_seq = tf.concat((additional_gt_frames[:, :, :, :6], output[:, :, :, :3]), 3)
                                        elif i == 0:
                                            current_seq = tf.concat((additional_gt_frames[:, :, :, 3:6], output[:, :, :, :6]), 3)
                                        elif i == nimages - 1:
                                            current_seq = tf.concat((output[:, :, :, i*3-3:i*3+3], additional_gt_frames[:, :, :, 6:9]), 3)
                                        elif i == nimages:
                                            current_seq = tf.concat((output[:, :, :, i*3-3:i*3], additional_gt_frames[:, :, :, 6:]), 3)
                                        else:
                                            current_seq = output[:, :, :, i*3-3:i*3+6]
                                        predict_real_seq.append(create_discriminator(None, current_seq))
                                for i in range(-1, nimages, 3):
                                    if i == -1:
                                        current_seq = tf.concat((additional_gt_frames[:, :, :, :3], output[:, :, :, :6]), 3)
                                    elif i + 3 >= nimages:
                                        current_seq = tf.concat((output[:, :, :, i*3:i*3+6], additional_gt_frames[:, :, :, 3:]), 3)
                                    else:
                                        current_seq = output[:, :, :, i*3:i*3+9]
                                    predict_real_seq.append(create_discriminator(None, current_seq))
                        with tf.name_scope("discriminator_fake"):
                            with tf.variable_scope("discriminator_temporal", reuse=tf.AUTO_REUSE):
                                for i in range(-1, nimages + 1, 3):
                                    if i == -1:
                                        current_seq = tf.concat((additional_gt_frames[:, :, :, :3], network[:, :, :, :6]), 3)
                                    elif i + 3 >= nimages:
                                        current_seq = tf.concat((network[:, :, :, i*3:i*3+6], additional_gt_frames[:, :, :, 3:]), 3)
                                    else:
                                        current_seq = network[:, :, :, i*3:i*3+9]
                                    predict_fake_seq.append(create_discriminator(None, current_seq))
                                          
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
                    if args.temporal_texture_buffer:
                        loss_fake = loss_fake + gan_loss_func(predict_fake_still, False)
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
        
        gen_saver = tf.train.Saver(gen_tvars, max_to_keep=1000)
        
        if args.is_train or args.collect_validate_loss:
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

        adam_before = adam_optimizer
        if args.update_bn:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                opt=adam_optimizer.minimize(loss_to_opt,var_list=var_list)
        else:
            opt=adam_optimizer.minimize(loss_to_opt,var_list=var_list)

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
    l2_loss = 0
    tf.summary.scalar('l2_loss', l2_loss)
    perceptual_loss = 0
    tf.summary.scalar('perceptual_loss', perceptual_loss)
    
    validate = 0
    tf.summary.scalar('validate', validate)

    #print("start sess")
    #sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=10, intra_op_parallelism_threads=3))
    sess = tf.Session()
    #print("after start sess")
    merged = tf.summary.merge_all()

    #print("initialize local vars")
    sess.run(tf.local_variables_initializer())
    #print("initialize global vars")
    sess.run(tf.global_variables_initializer())

    exclude_prefix = 'feature_reduction'

    var_all = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    read_from_epoch = False

    if not (args.debug_mode and args.mean_estimator):
        
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
            read_from_epoch = True
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
            
            if args.patch_gan_loss and args.is_train:
                if not read_from_epoch:
                    ckpt_time = os.path.getmtime(os.path.join(args.name, 'model_gen/checkpoint'))
                    reset_dir = False
                    for i in range(args.epoch, 0, -1):
                        epoch_dir = os.path.join(args.name, '%04d' % i)
                        if os.path.isdir(epoch_dir):
                            if os.path.exists(os.path.join(epoch_dir, 'score.txt')):
                                this_epoch_time = os.path.getmtime(os.path.join(epoch_dir, 'score.txt'))
                                if this_epoch_time <= ckpt_time:
                                    read_from_epoch = True
                                    args.which_epoch = i
                                    reset_dir = True
                                    break
                    if reset_dir:
                        for i in range(args.epoch, args.which_epoch, -1):
                            epoch_dir = os.path.join(args.name, '%04d' % i)
                            if os.path.isdir(epoch_dir):
                                shutil.rmtree(epoch_dir)
                                    
                                
        elif args.finetune:
            ckpt_origs = [None] * len(savers)
            for c_i in range(len(savers)):
                if args.finetune_epoch > 0:
                    ckpt_origs[c_i] = tf.train.get_checkpoint_state(os.path.join(args.orig_name, "%04d"%int(args.finetune_epoch), save_names[c_i]))
                else:
                    ckpt_origs[c_i] = tf.train.get_checkpoint_state(os.path.join(args.orig_name, save_names[c_i]))
            
            if None not in ckpt_origs:
                if args.boost_channels != '' and (not args.boost_ch_from_finetune):
                    if sparsity_vec in gen_tvars and gen_saver in savers:
                        old_gen_tvars = [tvars for tvars in gen_tvars if tvars != sparsity_vec]
                        old_gen_saver = tf.train.Saver(old_gen_tvars)
                        
                        old_savers = []
                        for saver in savers:
                            if saver == gen_saver:
                                old_savers.append(old_gen_saver)
                            else:
                                old_savers.append(saver)
                        
                for c_i in range(len(ckpt_origs)):
                    old_savers[c_i].restore(sess, ckpt_origs[c_i].model_checkpoint_path)
                    print('loaded '+ckpt_origs[c_i].model_checkpoint_path)


    
    
    save_frequency = 1
    num_epoch = args.epoch
    #assert num_epoch % save_frequency == 0

    

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

    if args.preload and args.is_train and (not args.use_queue) and (not args.geometry.startswith('boids')):
        output_images = np.empty([camera_pos_vals.shape[0], target_pl_h, target_pl_w, 3])
        all_grads = [None] * camera_pos_vals.shape[0]
        all_adds = np.empty([camera_pos_vals.shape[0], target_pl_h, target_pl_w, 1])
        for id in range(camera_pos_vals.shape[0]):
            output_images[id, :, :, :] = read_name(output_names[id], False)
            print(id)
            if args.additional_input:
                all_adds[id, :, :, 0] = read_name(add_names[id], True)
        
        if args.use_validation:
            validate_images = np.empty([validate_camera_vals.shape[0], target_pl_h, target_pl_w, 3])
            for id in range(validate_camera_vals.shape[0]):
                validate_images[id, :, :, :] = read_name(validate_img_names[id], False)

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
            if not args.use_queue:
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
            train_writer = tf.summary.FileWriter(args.name, sess.graph)
        # shader specific hack: if in fluid approximate mode (or temporal_texture_buffer=True), then sample number is 11 less than number of png / npy files in dataroot dir
        
        if args.geometry.startswith('boids'):
            rec_arr_len = args.niters
        elif args.learn_loss_proxy and args.proxy_loss_type == 'from_data':
            rec_arr_len = time_vals.shape[0] ** 2
        else:
            rec_arr_len = time_vals.shape[0]
            if args.temporal_texture_buffer:
                rec_arr_len -= 10
                
        all=np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
        all_l2=np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
        all_sparsity=np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
        all_regularization = np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
        all_training_loss = np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
        all_perceptual = np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
        all_gen_gan_loss = np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)
        all_discrim_loss = np.zeros(int(rec_arr_len * ntiles_w * ntiles_h), dtype=float)

        alpha_start = -3
        alpha_end = 0
        #num_transition_epoch = 50
        num_transition_epoch = 20
        alpha_schedule = np.logspace(alpha_start, alpha_end, num_transition_epoch)
        printval = False
        
        total_step_count = 0

        min_avg_loss = 1e20
        old_val_loss = 1e20
        
        if args.use_validation:
            if args.learn_loss_proxy:
                nvalidation_cameras = validate_time_vals.shape[0]
                assert nvalidation_cameras % args.batch_size == 0
            elif args.geometry.startswith('boids'):
                # deterministic sampling, so every iter samples the same validation examples
                v_nsamples = (max_sample_time - min_sample_time + 1)
                ntex = validate_camera_pos.shape[0] // (max_sample_time + 1 )

                nvalidations = v_nsamples * ntex

                nbatches = v_nsamples * ntex // args.batch_size

                start_texture = np.tile(np.expand_dims(np.arange(ntex) * (max_sample_time + 1), 1), (1, v_nsamples)).reshape(-1)
                v_sample_time = np.tile(np.arange(v_nsamples) + min_sample_time, ntex)
                end_texture = start_texture + v_sample_time


        for epoch in range(1, num_epoch+1):
            if args.use_queue:
                sess.run(train_iterator)

            if read_from_epoch:
                if epoch <= args.which_epoch:
                    continue
            else:
                next_save_point = epoch
                # for patchGAN loss, the checkpoint in the root directory may not be very up to date (if discrim is too strong it is very possible that the model saves a best l2/perceptual error, which is a lot of epochs ago)
                if os.path.isdir("%s/%04d"%(args.name,next_save_point)) and not args.patch_gan_loss:
                    continue

            cnt=0

            permutation = np.random.permutation(int(nexamples * ntiles_h * ntiles_w))
            nupdates = permutation.shape[0] if not args.use_batch else int(np.ceil(float(permutation.shape[0]) / args.batch_size))
            sub_epochs = 1
            
            # if loss_proxy is gt value is directly read from data, we will enumerate each possible pair for each epoch
            if args.learn_loss_proxy and (not args.proxy_loss_type == 'from_data'):
                permutation_target_pl = np.random.permutation(int(nexamples * ntiles_h * ntiles_w))
                if args.loss_proxy_encourage_0:
                    # for a quater of the training iterations, actually use target as the same output_ground, to ensure the loss can be 0
                    same_target_idx = np.random.choice(int(nexamples * ntiles_h * ntiles_w), size=int(nexamples * ntiles_h * ntiles_w) // 4, replace=False)
                    permutation_target_pl[same_target_idx] = permutation[same_target_idx]
            else:
                permutation_target_pl = None
            
            if args.finetune and epoch <= num_transition_epoch:
                alpha_val = alpha_schedule[epoch-1]

            feed_dict={}
            if sparsity_schedule is not None:
                feed_dict[sparsity_scale] = sparsity_schedule[epoch-1]
            if target_channel_schedule is not None:
                print("using target channel schedule")
                feed_dict[target_channel] = target_channel_schedule[epoch-1]

            for sub_epoch in range(sub_epochs):

                for i in range(nupdates):


                    st=time.time()
                    start_id = i * args.batch_size
                    end_id = min(permutation.shape[0], (i+1)*args.batch_size)

                    frame_idx = (permutation[start_id:end_id] / (ntiles_w * ntiles_h)).astype('i')
                    tile_idx = (permutation[start_id:end_id] % (ntiles_w * ntiles_h)).astype('i')
                    run_options = None
                    run_metadata = None
                    
                    if args.learn_loss_proxy:
                        if args.proxy_loss_type == 'from_data':
                            # if read from data, nexamples = # of camera pos ** 2
                            # for each idx, source_idx = idx % # of camera pos
                            # for each idx, target_idx = idx // # of camera pos
                            target_pl_idx = frame_idx // ncameras
                        else:
                            # using the same reference img pool as target image
                            # but randomly choose from the pool so that it doesn't has to be exactly same as current reference image
                            target_pl_idx = (permutation_target_pl[start_id:end_id] / (ntiles_w * ntiles_h)).astype('i')
                            
                    if args.discrim_train_steps > 1:
                        feed_dict[step_count] = total_step_count
                        total_step_count += 1

                    if args.geometry.startswith('boids'):
                        # 1D case
                        if not args.train_temporal_seq:
                            camera_pos_vals = camera_pos_vals_pool[np.random.choice(len(camera_pos_vals_pool))]
                            
                            sample_time = np.random.choice(sample_range, args.batch_size, p=sample_p)
                            sample_start = numpy.random.choice(camera_pos_vals.shape[0] - max_sample_time, args.batch_size, replace=False)
                            feed_dict[shader_time] = sample_time
                            feed_dict[texture_maps] = camera_pos_vals[sample_start]
                            feed_dict[output_pl] = camera_pos_vals[sample_start+sample_time]
                            
                            if args.random_switch_label:
                                new_label = np.random.permutation(args.n_boids)
                                feed_dict[texture_maps] = feed_dict[texture_maps][:, new_label]
                                feed_dict[output_pl] = feed_dict[output_pl][:, new_label]
                        else:
                            # for simplicity, use same sample_time in the same batch
                            sample_time = np.random.choice(sample_range, p=sample_p)
                            sample_start = numpy.random.choice(np.arange(camera_pos_vals.shape[0] - max_sample_time * ngts), args.batch_size, replace=False)
                            feed_dict[shader_time] = sample_time * np.ones(args.batch_size)
                            output_indices = sample_time * np.arange(ngts + 1) + np.tile(np.expand_dims(sample_start, 1), ngts+1)
                            feed_dict[output_pl] = camera_pos_vals[output_indices]
                        
                    elif args.learn_loss_proxy and args.proxy_loss_type == 'from_data':
                        source_idx = frame_idx % ncameras
                        output_arr = gt_loss_train[target_pl_idx, source_idx]
                        feed_dict[output_pl] = np.expand_dims(output_arr, 3)
                        
                        feed_dict[shader_time] = time_vals[source_idx]
                        feed_dict[camera_pos] = camera_pos_vals[source_idx, :].transpose()
                        
                        if args.preload:
                            target_arr = output_images[target_pl_idx]
                        else:
                            target_arr = np.empty([args.batch_size, target_pl_h, target_pl_w, 3])
                            for i in range(target_pl_idx.shape[0]):
                                target_arr[i] = read_name(output_names[target_pl_idx[i]], False)
                        feed_dict[target_pl] = target_arr
                                
                    else:
                    
                        if not args.use_queue:
                            if not args.preload:
                                #if args.tile_only:
                                #    output_arr = np.empty([args.batch_size, args.tiled_h, args.tiled_w, 3])
                                #else:
                                #    output_arr = np.empty([args.batch_size, args.input_h, args.input_w, 3])
                                output_arr = np.empty([args.batch_size, target_pl_h, target_pl_w, output_nc])

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
                                
                                if args.learn_loss_proxy:
                                    target_arr = np.empty(output_arr.shape)
                                    for img_idx in range(target_pl_idx.shape[0]):
                                        target_arr[img_idx, :, :, :] = read_name(output_names[target_pl_idx[img_idx]], False)
                            else:
                                output_arr = output_images[frame_idx]
                                if args.additional_input:
                                    additional_arr = all_adds[frame_idx]
                                if args.learn_loss_proxy:
                                    target_arr = output_images[target_pl_idx]
                            #output_arr = numpy.ones([1, args.input_h, args.input_w, 3])
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
                            if args.learn_loss_proxy:
                                feed_dict[target_pl] = target_arr

                        if not args.use_queue:
                            if not args.temporal_texture_buffer:
                                camera_val = camera_pos_vals[frame_idx, :].transpose()
                                feed_dict[camera_pos] = camera_val
                                feed_dict[shader_time] = time_vals[frame_idx]
                                
                                if augment_camera_pos:
                                    augment_random_seed = np.random.choice(3, size=frame_idx.shape).astype('f') - 1
                                    feed_dict[camera_pos] += np.expand_dims(camera_periodic_range, 1) * np.expand_dims(augment_random_seed, 0)
                            else:
                                assert args.batch_size == 1
                                camera_val = np.empty([33, 1])
                                # e.g. if idx = 1
                                # camera_pos_val[1] is last mouse for 1st frame
                                # and we collect 10 next mouse pos
                                camera_val[:, 0] = np.reshape(camera_pos_vals[frame_idx[0]:frame_idx[0]+11, 3:], 33)
                                feed_dict[camera_pos] = camera_val
                                feed_dict[shader_time] = time_vals[frame_idx]
                                current_texture_maps = np.transpose(np.load(output_names[frame_idx[0]]), (2, 0, 1))
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    if not (current_texture_maps.shape[1] == height and current_texture_maps.shape[2] == width):
                                        current_texture_maps = skimage.transform.resize(current_texture_maps, (current_texture_maps.shape[0], height, width))
                                for k in range(len(texture_maps)):
                                    feed_dict[texture_maps[k]] = current_texture_maps[k]
                                    
                            

                    st1 = time.time()
                    
                    _,current, current_l2, current_sparsity, current_regularization, current_training, current_perceptual, current_gen_loss_GAN, current_discrim_loss, current_ind_val =sess.run([opt,loss, loss_l2, sparsity_loss, regularizer_loss, loss_to_opt, perceptual_loss_add, gen_loss_GAN, discrim_loss, current_ind],feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
                    
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


                    if run_metadata is not None and args.generate_timeline:
                        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open("%s/epoch%04d_step%d.json"%(args.name,epoch, i), 'w') as f:
                            f.write(chrome_trace)
                    #_,current=sess.run([opt,loss],feed_dict={input:input_image,output:output_image, alpha: alpha_val})
                    if args.use_queue:
                        current_slice = current_ind_val
                    else:
                        current_slice = permutation[start_id:end_id]
                    all[current_slice]=current
                    all_l2[current_slice]=current_l2
                    all_sparsity[current_slice]=current_sparsity
                    all_regularization[current_slice] = current_regularization
                    all_training_loss[current_slice] = current_training
                    all_perceptual[current_slice] = current_perceptual
                    all_gen_gan_loss[current_slice] = current_gen_loss_GAN
                    all_discrim_loss[current_slice] = current_discrim_loss
                    cnt += args.batch_size if args.use_batch else 1
                    print("%d %d %.5f %.5f %.2f %.2f %s %d"%(epoch,cnt,current,np.mean(all[np.where(all)]),time.time()-st, st2-st1,os.getcwd().split('/')[-2], current_slice[0]))

            avg_loss = np.mean(all[np.where(all)])
            avg_loss_l2 = np.mean(all_l2[np.where(all_l2)])
            #avg_loss_sparsity = np.mean(all_sparsity[np.where(all_sparsity)])
            avg_loss_sparsity = np.mean(all_sparsity)
            avg_loss_regularization = np.mean(all_regularization)
            avg_training_loss = np.mean(all_training_loss)
            avg_perceptual = np.mean(all_perceptual)
            avg_gen_gan = np.mean(all_gen_gan_loss)
            avg_discrim = np.mean(all_discrim_loss)
            
            if args.use_validation:
                
                current_val_loss = 0.0
                
                if args.learn_loss_proxy and args.proxy_loss_type == 'from_data':
                    
                    for ind_target in range(nvalidation_cameras):
                        for ind_source in range(0, nvalidation_cameras, args.batch_size):
                            feed_dict[output_pl] = np.expand_dims(gt_loss_val[ind_target, ind_source:ind_source+args.batch_size], 3)
                            feed_dict[shader_time] = validate_time_vals[ind_source:ind_source+args.batch_size]
                            feed_dict[camera_pos] = validate_camera_vals[ind_source:ind_source+args.batch_size].transpose()
                            
                            if args.preload:
                                target_arr = np.tile(validate_images[ind_target], [args.batch_size, 1, 1, 1])
                            else:
                                target_arr = np.tile(read_name(validate_img_names[ind_target], False), [args.batch_size, 1, 1, 1])
                            feed_dict[target_pl] = target_arr
                            
                            current_val_loss += sess.run(loss_to_opt, feed_dict=feed_dict)
                        print('validation,', ind_target)
                            
                else:
                    for batch_ind in range(nbatches):
                        start_ind = args.batch_size * batch_ind
                        end_ind = start_ind + args.batch_size
                        feed_dict[texture_maps] = validate_camera_pos[start_texture[start_ind:end_ind]]
                        feed_dict[shader_time] = v_sample_time[start_ind:end_ind].astype('f')
                        feed_dict[output_pl] = validate_camera_pos[end_texture[start_ind:end_ind]]
                        loss_per_sample_val = sess.run(loss_per_sample, options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)

                        current_val_loss += np.sum(loss_per_sample_val * sample_p[v_sample_time[start_ind:end_ind] - min_sample_time])

                        
                if current_val_loss > old_val_loss:
                    pass
                else:
                    old_val_loss = current_val_loss
                

            if min_avg_loss > avg_training_loss:
                min_avg_loss = avg_training_loss
                
            update_current_epoch = False
            if args.use_validation:
                if current_val_loss == old_val_loss:
                    update_current_epoch = True
            else:
                if min_avg_loss == avg_training_loss:
                    update_current_epoch = True

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
                if args.use_validation:
                    summary.value.add(tag='validate', simple_value=current_val_loss)
                train_writer.add_summary(summary, epoch)

            os.makedirs("%s/%04d"%(args.name,epoch))
            target=open("%s/%04d/score.txt"%(args.name,epoch),'w')
            target.write("%f"%np.mean(all[np.where(all)]))
            target.close()

            #target = open("%s/%04d/score_breakdown.txt"%(args.name,epoch),'w')
            #target.write("%f, %f, %f, %f"%(avg_test_close, avg_test_far, avg_test_middle, avg_test_all))
            #target.close()

            if update_current_epoch:
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
        
        if args.test_output_dir != '':
            args.name = args.test_output_dir
        
        if args.collect_validate_loss:
            assert args.test_training
            dirs = sorted(os.listdir(args.name))
            
            if not args.use_queue:
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
                        
            all_example_vals = np.empty([time_vals.shape[0], 6])
                
            for dir in dirs:
                success = False
                try:
                    epoch = int(dir)
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
                    
                # DOGE: debug only
                if epoch > 2:
                    break
                    
                if args.use_queue:
                    sess.run(train_iterator)
                                
                for ind in range(time_vals.shape[0]):
                    
                    if not args.use_queue:
                        feed_dict = {camera_pos: np.expand_dims(camera_pos_vals[ind], 1),
                                     shader_time: time_vals[ind:ind+1],
                                     output_pl: validate_imgs[ind]}

                        if args.tile_only:
                            feed_dict[h_start] = tile_start_vals[ind:ind+1, 0] - padding_offset / 2
                            feed_dict[w_start] = tile_start_vals[ind:ind+1, 1] - padding_offset / 2
                        else:
                            feed_dict[h_start] = np.array([- padding_offset / 2])
                            feed_dict[w_start] = np.array([- padding_offset / 2])
                    else:
                        feed_dict = {}
                    

                    current, current_l2, current_perceptual, current_gen_loss_GAN, current_discrim_loss = sess.run([loss, loss_l2, perceptual_loss_add, gen_loss_GAN, discrim_loss], feed_dict=feed_dict)
                    all_example_vals[ind] = np.array([epoch, current, current_l2, current_perceptual, current_gen_loss_GAN, current_discrim_loss])
                    
                all_vals.append(np.mean(all_example_vals, 0))
            
            all_vals = np.array(all_vals)
            np.save(os.path.join(args.name, 'validation.npy'), all_vals)
            open(os.path.join(args.name, 'validation.txt'), 'w').write('validation dataset: %s\n raw data stored in: %s\n' % (args.dataroot, os.uname().nodename))
            
            min_idx = np.argsort(all_vals[:, 1])
            min_epoch = all_vals[min_idx[0], 0]
            
            val_dir = os.path.join(args.name, 'best_val')
            if os.path.isdir(val_dir):
                shutil.rmtree(val_dir)
            os.mkdir(val_dir)
                
            for c_i in range(len(savers)):
                saver_dir = os.path.join(val_dir, save_names[c_i])
                if os.path.isdir(saver_dir):
                    shutil.rmtree(os.path.abspath(saver_dir))
                shutil.copytree(os.path.join(args.name, '%04d' % (int(min_epoch)), save_names[c_i]), saver_dir)
            
            open(os.path.join(args.name, 'best_val_epoch.txt'), 'w').write(str(int(min_epoch)))
            
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
            
            plt.savefig(os.path.join(args.name, 'validation.png'))
            plt.close(figure)
                                
        else:
            if args.use_queue:
                sess.run(train_iterator)

            if args.sparsity_vec_histogram:
                sparsity_vec_vals = numpy.abs(sess.run(sparsity_vec))
                figure = pyplot.figure()
                pyplot.hist(sparsity_vec_vals, bins=10)
                pyplot.title('entries > 0.2: %d' % numpy.sum(sparsity_vec_vals > 0.2))
                figure.savefig(os.path.join(args.name, 'sparsity_hist' + ("_epoch_%04d"%args.which_epoch if read_from_epoch else '') + '.png'))
                return

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



            debug_dir += '_zero_out_sparsity_vec' if args.zero_out_sparsity_vec else ''

            if read_from_epoch:
                debug_dir += "_epoch_%04d"%args.which_epoch

            if not os.path.isdir(debug_dir):
                os.makedirs(debug_dir)

            if args.render_only and os.path.exists(os.path.join(debug_dir, 'video.mp4')):
                os.remove(os.path.join(debug_dir, 'video.mp4'))

            if args.render_only:
                shutil.copyfile(args.render_t, os.path.join(debug_dir, 'render_t.npy'))
                if args.geometry.startswith('boids'):
                    np.save(os.path.join(debug_dir, 'init_texture.npy'), camera_pos_vals[0:1])
                else:
                    shutil.copyfile(args.render_camera_pos, os.path.join(debug_dir, 'camera_pos.npy'))
                    if args.temporal_texture_buffer or args.train_temporal_seq:
                        shutil.copyfile(args.render_temporal_texture, os.path.join(debug_dir, 'init_texture.npy'))

            nburns = 10

            if args.repeat_timing > 1:
                nburns = 20
                if args.train_temporal_seq:
                    time_stats = numpy.zeros(time_vals.shape[0] * args.repeat_timing * (args.inference_seq_len - args.nframes_temporal_gen + 1))
                else:
                    time_stats = numpy.zeros(time_vals.shape[0] * args.repeat_timing)
                time_count = 0

            if not args.geometry.startswith('boids'):
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
                if args.input_nc > target_channel_max:
                    # workaround due to an incorrect tensorflow dependency on placeholders...
                    #feed_dict = {camera_pos: camera_pos_vals[0, :], shader_time: time_vals[0:1]}
                    if not args.use_queue:
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


            if args.render_only:
                if h_start.op.type == 'Placeholder':
                    feed_dict[h_start] = np.array([- padding_offset / 2])
                if w_start.op.type == 'Placeholder':
                    feed_dict[w_start] = np.array([- padding_offset / 2])
                if args.train_with_random_rotation:
                    feed_dict[rotate] = 0
                    feed_dict[flip] = 0
                if args.temporal_texture_buffer:
                    if args.geometry.startswith('boids'):
                        feed_dict[texture_maps] = camera_pos_vals[0:1]
                        nexamples = time_vals.shape[0]
                        all_output = np.empty([nexamples, args.n_boids, 4])
                    else:
                        init_texture = np.transpose(np.load(args.render_temporal_texture), (2, 0, 1))
                        for k in range(len(texture_maps)):
                            feed_dict[texture_maps[k]] = init_texture[k]
                        nexamples = time_vals.shape[0] // 10 - 1
                else:
                    nexamples = time_vals.shape[0]

                if args.train_temporal_seq:
                    all_previous_inputs = []
                    if not args.geometry.startswith('boids'):
                        # 2D case
                        output_grounds = np.load(args.render_temporal_texture)
                        previous_buffer = np.zeros([1, input_pl_h, input_pl_w, 6*(args.nframes_temporal_gen-1)])

                for i in range(nexamples):
                    #feed_dict = {camera_pos: camera_pos_vals[i:i+1, :].transpose(), shader_time: time_vals[i:i+1]}

                    if not inference_entire_img_valid:
                        feed_dict[h_start] = tile_start_vals[i:i+1, 0] - padding_offset / 2
                        feed_dict[w_start] = tile_start_vals[i:i+1, 1] - padding_offset / 2

                    if args.temporal_texture_buffer:
                        if args.geometry.startswith('boids'):
                            feed_dict[shader_time] = time_vals[i:i+1]
                        else:
                            # TODO: this only works for fluid approx app
                            camera_val = np.empty([33, 1])
                            camera_val[:, 0] = np.reshape(camera_pos_vals[i*10:i*10+11, 3:], 33)
                            feed_dict[camera_pos] = camera_val
                            feed_dict[shader_time] = time_vals[i*10:i*10+1]
                    else:
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
                        if not args.geometry.startswith('boids'):
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
                            if i < args.nframes_temporal_gen - 1:
                                all_previous_inputs.append((camera_pos_vals[i:i+1] + color_scale[0]) * color_scale[1])
                                feed_dict[texture_maps] = camera_pos_vals[i:i+1]
                                all_previous_inputs.append(sess.run(input_label, feed_dict=feed_dict))
                                output_image = (camera_pos_vals[i+1:i+2] + color_scale[0]) * color_scale[1]
                            else:
                                previous_buffer = np.concatenate(all_previous_inputs, 2)
                                feed_dict[previous_input] = previous_buffer

                                if i == args.nframes_temporal_gen - 1:
                                    feed_dict[texture_maps] = camera_pos_vals[i:i+1]
                                else:
                                    feed_dict[texture_maps] = output_image
                                all_previous_inputs.append((feed_dict[texture_maps] + color_scale[0]) * color_scale[1])

                                output_image, generated_label = sess.run([network, input_label], feed_dict=feed_dict)
                                all_previous_inputs.append(generated_label)
                                all_previous_inputs = all_previous_inputs[2:]

                    else:
                        for _ in range(nruns):
                            output_image = sess.run(network, feed_dict=feed_dict)
                            if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                output_buffer += output_image[:, :, :, ::-1]



                    if args.geometry.startswith('boids'):
                        output_image /= color_scale[1]
                        output_image -= color_scale[0]
                        all_output[i] = output_image[0]
                        feed_dict[texture_maps] = output_image
                    else:
                        if args.mean_estimator:
                            output_image = output_image[:, :, :, ::-1]
                        if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                            output_buffer /= args.estimator_samples
                            output_image[:] = output_buffer[:]

                        if output_nc == 3:
                            output_image = np.clip(output_image,0.0,1.0)
                            output_image *= 255.0
                            cv2.imwrite("%s/%06d.png"%(debug_dir, i+1),np.uint8(output_image[0,:,:,:]))
                        else:
                            assert (output_nc - 4) % 3 == 0
                            nimages = (output_nc - 4) // 3
                            output_image[:, :, :, :3*nimages] = np.clip(output_image[:, :, :, :3*nimages],0.0,1.0)
                            output_image[:, :, :, :3*nimages] *= 255.0
                            for img_id in range(nimages):
                                cv2.imwrite("%s/%05d%d.png"%(debug_dir, i+1, img_id),np.uint8(output_image[0,:,:,3*img_id:3*img_id+3]))
                            texture_map_start = output_nc - len(texture_maps)
                            for k in range(len(texture_maps)):
                                feed_dict[texture_maps[k]] = output_image[0, :, :, texture_map_start+k]

                    print('finished', i)
                if args.geometry.startswith('boids'):
                    np.save(os.path.join(debug_dir, 'all_output.npy'), all_output)
                else:
                    if not args.render_no_video:
                        os.system('ffmpeg %s -i %s -r 30 -c:v libx264 -preset slow -crf 0 -r 30 %s'%('-start_number 10' if args.temporal_texture_buffer else '', os.path.join(debug_dir, '%06d.png'), os.path.join(debug_dir, 'video.mp4')))
                        open(os.path.join(debug_dir, 'index.html'), 'w+').write("""
        <html>
        <body>
        <br><video controls><source src="video.mp4" type="video/mp4"></video><br>
        </body>
        </html>""")
                return
            else:
                if args.geometry.startswith('boids'):


                    #nsamples = (max_sample_time - min_sample_time + 1)
                    #ntex = nexamples // (max_sample_time + 1 )
                    nsamples = 128
                    ntex = 100
                    tex_spacing = (nexamples - nsamples - 1) // ntex

                    all_l2 = np.zeros([nsamples, ntex], dtype=float)
                    all_time = np.zeros([nsamples, ntex])

                    all_output_pos = np.empty([nsamples, ntex, args.n_boids, 4])
                    all_error = np.empty([nsamples, ntex, args.n_boids, 4])

                    if args.boids_seq_metric:
                        simulation_step = 20

                        nframes = 150
                        nrepeat = 200

                        spacing = (nexamples - 1 - nframes * simulation_step) // nrepeat

                        print('spacing', spacing)

                        #nframes = (nexamples - 1) // simulation_step
                        #nrepeat = min((nexamples - 1) % simulation_step, 10)
                        #if nrepeat < 10:
                        #    nframes -= 1
                        #    nrepeat = 10

                        all_seq_pos = np.empty([nrepeat, nframes, args.n_boids, 4])
                        all_seq_err = np.empty([nrepeat, nframes])

                        feed_dict[shader_time] = np.array([simulation_step]).astype('f')
                        for repeat in range(nrepeat):
                            current_idx = repeat * spacing
                            print(repeat)
                            feed_dict[texture_maps] = camera_pos_vals[[current_idx]]
                            for frame in range(nframes):
                                feed_dict[output_pl] = camera_pos_vals[[current_idx + simulation_step]]
                                output_pos, l2_loss_val = sess.run([network, loss_l2], options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
                                output_pos /= color_scale[1]
                                output_pos -= color_scale[0]

                                all_seq_pos[repeat, frame] = output_pos[0]
                                all_seq_err[repeat, frame] = l2_loss_val

                                feed_dict[texture_maps] = output_pos

                                current_idx += simulation_step
                        np.save(os.path.join(debug_dir, 'all_seq_pos.npy'), all_seq_pos)
                        np.save(os.path.join(debug_dir, 'all_seq_err.npy'), all_seq_err)


                    if args.boids_single_step_metric:
                        for tex_ind in range(ntex):
                            #current_ind = tex_ind * (max_sample_time + 1)
                            current_ind = tex_ind * tex_spacing
                            feed_dict[texture_maps] = camera_pos_vals[[current_ind]]
                            print(tex_ind)
                            for time_ind in range(nsamples):
                                #current_time = time_ind + min_sample_time
                                current_time = time_ind + 1
                                feed_dict[shader_time] = np.array([current_time]).astype('f')
                                feed_dict[output_pl] = camera_pos_vals[[current_ind+current_time]]
                                st_before = time.time()
                                if not args.accurate_timing:
                                    output_pos, l2_loss_val = sess.run([network, loss_l2], options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
                                    st_after = time.time()
                                else:
                                    sess.run(network, feed_dict=feed_dict)
                                    st_after = time.time()
                                    output_pos, l2_loss_val = sess.run([network, loss_l2], options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
                                st_sum = (st_after - st_before)

                                output_pos /= color_scale[1]
                                output_pos -= color_scale[0]

                                all_l2[time_ind, tex_ind] = l2_loss_val
                                all_output_pos[time_ind, tex_ind] = output_pos[0]
                                all_time[time_ind, tex_ind] = st_sum

                                all_error[time_ind, tex_ind] = output_pos[0] - feed_dict[output_pl][0]

                        all_time = all_time[:, nburns:]        

                        numpy.save(os.path.join(debug_dir, 'all_l2.npy'), all_l2)
                        numpy.save(os.path.join(debug_dir, 'all_output.npy'), all_output_pos)
                        numpy.save(os.path.join(debug_dir, 'all_time.npy'), all_time)
                        numpy.save(os.path.join(debug_dir, 'all_error.npy'), all_error)
                        #loss_prob = np.sum(np.mean(all_l2, 1) * sample_p)
                        #open("%s/all_loss.txt"%debug_dir, 'w').write("%f, %f"%(np.mean(all_l2), loss_prob))
                        #open("%s/all_time.txt"%debug_dir, 'w').write("%f"%(np.median(all_time)))
                        print("all times saved")

                else:
                    if args.analyze_nn_discontinuity:
                        all_discontinuities = []
                        all_ops = tf.get_default_graph().get_operations()
                        all_generator_ops = [n for n in all_ops if n.name.startswith('generator/')]
                        all_lrelu = [n for n in all_generator_ops if n.name.endswith('Maximum')]
                        for node in all_lrelu:
                            op = tf.get_default_graph().get_operation_by_name(node.name.replace('Maximum', 'BiasAdd'))
                            assert len(op.outputs) == 1
                            all_discontinuities.append(op.outputs[0])
                        args.repeat_timing = 1

                    nexamples = time_vals.shape[0]
                    ncameras = nexamples
                    if args.temporal_texture_buffer:
                        # for regular test, only inference once
                        # start from a texture buffer rendered from gt
                        # then inference the next 10 frames
                        # for efficiency, do not render overlapping frames
                        nexamples -= 1
                        nexamples = nexamples // 10
                    elif args.learn_loss_proxy and args.proxy_loss_type == 'from_data':
                        nexamples = time_vals.shape[0] ** 2

                    all_test = np.zeros(nexamples, dtype=float)
                    all_grad = np.zeros(nexamples, dtype=float)
                    all_l2 = np.zeros(nexamples, dtype=float)
                    all_perceptual = np.zeros(nexamples, dtype=float)
                    python_time = numpy.zeros(nexamples)

                    if args.learn_loss_proxy and (not args.proxy_loss_type == 'from_data'):
                        all_loss_proxy = np.zeros((nexamples, 2), dtype=float)
                    if args.train_temporal_seq:
                        previous_buffer = np.empty([1, input_pl_h, input_pl_w, 6*(args.nframes_temporal_gen-1)])

                    for i in range(nexamples):
                        print(i)
                        if args.train_with_random_rotation:
                            feed_dict[rotate] = 0
                            feed_dict[flip] = 0
                        if not args.use_queue:

                            if args.learn_loss_proxy and args.proxy_loss_type == 'from_data':
                                target_pl_idx = i // ncameras
                                source_idx = i % ncameras
                                feed_dict[output_pl] = np.expand_dims(np.expand_dims(gt_loss_train[target_pl_idx, source_idx], 0), 3)
                                feed_dict[shader_time] = [time_vals[source_idx]]
                                feed_dict[camera_pos] = np.expand_dims(camera_pos_vals[source_idx, :], axis=1)
                                feed_dict[target_pl] = np.expand_dims(read_name(val_img_names[target_pl_idx], False, False), 0)
                            else:    
                                camera_val = np.expand_dims(camera_pos_vals[i, :], axis=1)
                                #feed_dict = {camera_pos: camera_val, shader_time: time_vals[i:i+1]}
                                feed_dict[camera_pos] = camera_val
                                feed_dict[shader_time] = time_vals[i:i+1]

                                if args.use_dataroot and args.temporal_texture_buffer:
                                    # TODO: this only works for fluid approx app
                                    camera_val = np.empty([33, 1])
                                    camera_val[:, 0] = np.reshape(camera_pos_vals[i*10:i*10+11, 3:], 33)
                                    feed_dict[camera_pos] = camera_val
                                    feed_dict[shader_time] = time_vals[i*10:i*10+1]


                                if args.use_dataroot:
                                    if args.temporal_texture_buffer:
                                        output_ground = np.empty([1, output_pl.shape[1].value, output_pl.shape[2].value, output_nc])
                                        # a hack to read 10 frames after selected idx in fluid approx mode
                                        # at first we only test inference on input with every 10 frames
                                        # so that output frame will be non overlapping
                                        # for index i, output gt is i+1 to i+10
                                        for seq_id in range(10):
                                            output_ground[0, :, :, seq_id*3:seq_id*3+3] = read_name(val_img_names[i*10+seq_id+1], False)


                                        current_texture_maps = np.transpose(np.load(val_img_names[i*10]), (2, 0, 1))
                                        with warnings.catch_warnings():
                                            warnings.simplefilter("ignore")
                                            if not (current_texture_maps.shape[1] == height and current_texture_maps.shape[2] == width):
                                                current_texture_maps = skimage.transform.resize(current_texture_maps, (current_texture_maps.shape[0], height, width))
                                        for k in range(len(texture_maps)):
                                            feed_dict[texture_maps[k]] = current_texture_maps[k]
                                    elif args.train_temporal_seq:
                                        output_ground = None
                                    else:
                                        if args.learn_loss_proxy:

                                            # placeholder sampling stragety for target output
                                            # for each example, compute the loss twoce
                                            # once with the target that is exactly the same as the expected output
                                            # another time use the next wrapped len(dataset) // 2 index

                                            output_ground = np.expand_dims(read_name(val_img_names[i], False, False), 0)

                                            target_idx = (i + len(val_img_names) // 2) % (len(val_img_names))
                                            target_ground = np.expand_dims(read_name(val_img_names[target_idx], False, False), 0)

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
                            output_buffer = np.zeros([1, args.input_h, args.input_w, output_nc])
                        else:
                            output_buffer = np.zeros([1, args.input_h, args.input_w, output_nc*(args.inference_seq_len - args.nframes_temporal_gen + 1)])

                        st = time.time()
                        if args.tiled_training:
                            assert not args.use_queue
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
                            elif args.learn_loss_proxy and (not args.proxy_loss_type == 'from_data'):
                                nruns = 2
                                assert output_nc == 1
                                output_images = []
                                l2_loss_vals = 0
                            else:
                                nruns = args.repeat_timing
                            st_sum = 0
                            timeline_sum = 0
                            for k in range(nruns):
                                if args.learn_loss_proxy and (not args.proxy_loss_type == 'from_data'):
                                    if k == 0:
                                        feed_dict[target_pl] = output_ground
                                    else:
                                        feed_dict[target_pl] = target_ground

                                st_before = time.time()
                                if not args.accurate_timing:
                                    output_image, l2_loss_val, grad_loss_val, perceptual_loss_val, current_ind_val = sess.run([network, loss_l2, loss_add_term, perceptual_loss_add, current_ind], options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
                                    st_after = time.time()
                                else:
                                    sess.run(network, feed_dict=feed_dict)
                                    st_after = time.time()
                                    output_image, l2_loss_val, grad_loss_val, perceptual_loss_val, current_ind_val = sess.run([network, loss_l2, loss_add_term, perceptual_loss_add, current_ind], options=run_options, run_metadata=run_metadata, feed_dict=feed_dict)
                                st_sum += (st_after - st_before)
                                if args.repeat_timing > 1:
                                    time_stats[time_count] = st_after - st_before
                                    time_count += 1
                                if args.debug_mode and args.mean_estimator and args.mean_estimator_memory_efficient:
                                    output_buffer += output_image[:, :, :, ::-1]
                                if args.learn_loss_proxy and (not args.proxy_loss_type == 'from_data'):
                                    output_images.append(output_image)
                                    l2_loss_vals += l2_loss_val
                                    all_loss_proxy[i, k] = np.mean(output_image)
                                else:
                                    output_images = [output_image]
                            st2 = time.time()
                            if args.learn_loss_proxy and (not args.proxy_loss_type == 'from_data'):
                                l2_loss_val = l2_loss_vals / nruns
                            #print("rough time estimate:", st2 - st)
                            print(current_ind_val, "rough time estimate:", st_sum)

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
                        #if not args.stratified_sample_higher_res:
                        if (not args.use_queue) and (not args.train_temporal_seq) and (not args.learn_loss_proxy):
                            loss_val = np.mean((output_image - output_ground) ** 2)
                        else:
                            loss_val = l2_loss_val
                        #print("loss", loss_val, l2_loss_val * 255.0 * 255.0)
                        all_test[i] = loss_val
                        all_l2[i] = l2_loss_val
                        all_grad[i] = grad_loss_val
                        all_perceptual[i] = perceptual_loss_val

                        if args.learn_loss_proxy:
                            for k in range(len(output_images)):
                                output_image = output_images[k]
                                output_image=np.clip(output_image,0.0,1.0)
                                output_image *= 255.0
                                cv2.imwrite("%s/%06d%d.png"%(debug_dir, i+1, k),np.uint8(output_image[0,:,:,:]))
                        elif output_image.shape[3] == 3:
                            output_image=np.clip(output_image,0.0,1.0)
                            output_image *= 255.0
                            cv2.imwrite("%s/%06d.png"%(debug_dir, i+1),np.uint8(output_image[0,:,:,:]))
                        else:
                            assert args.temporal_texture_buffer or args.train_temporal_seq
                            #assert args.geometry == 'texture_approximate_10f'
                            if args.temporal_texture_buffer:
                                assert (output_nc - 4) % 3 == 0
                                nimages = (output_nc - 4) // 3
                            else:
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
                    if args.learn_loss_proxy:
                        if args.proxy_loss_type == 'from_data':
                            all_l2 = all_l2.reshape((ncameras, ncameras))
                            # indexed by [target_idx, source_idx]
                            np.save(os.path.join(debug_dir, 'all_loss_proxy.npy'), all_l2)
                        else:
                            numpy.save(os.path.join(debug_dir, 'all_loss_proxy.npy'), all_loss_proxy)
                    print("all times saved")

            test_dirname = debug_dir

            if not args.geometry.startswith('boids'):
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


    sess.close()

if __name__ == '__main__':
    main()
