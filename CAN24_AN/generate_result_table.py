import os
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import colorsys
import skimage
import skimage.io
import shutil
import sys
from matplotlib.patches import BoxStyle

hue_shaders = {
    'bricks': [0.1,1,1],
    'mandelbrot': [0.2, 1, 1],
    'mandelbulb': [0.3, 1, 1],
    'marble': [0.0, 0.6, 1],
    'oceanic': [0.5, 1, 1],
    'primitives': [0.6, 0.9, 1],
    'trippy': [0.8, 0.9, 1],
    'venice': [0.9, 0.8, 1],
    'boids': [0.7, 0.8, 1]
}

teaser_app = 'simplified'
teaser_shader = 'venice'

model_parent_dir = '/mnt/shadermlnfs1/shadermlvm/playground/models/'
dataset_parent_dir = '/mnt/shadermlnfs1/shadermlvm/playground/datasets/'

tex_prefix = 'new_submission/'

score_pl = 0.0
img_pl = np.zeros((640, 960, 3))
allow_missing_MSAA = False
allow_missing_temporal = True
allow_missing_simulation = True
use_gamma_corrected = False

app_shader_dir_200 = {
'denoising': {
    'bricks': {'dir': ['1x_1sample_bricks_with_bg_all/test/',
                       '1x_1sample_bricks_with_bg_aux_largest_capacity/test/',
                       '1x_1sample_bricks_with_bg_all/mean1_test/'],
               'img_idx': 9,
               'img_zoom_bbox': [400, 480, -140, -80],
               'gt_dir': 'datas_bricks_normal_texture/test_img',
               'msaa_sample': 1,
               'print': 'Bricks'
              },
    'mandelbrot': {'dir': ['1x_1sample_mandelbrot_with_bg_stratified_subsample_2/test/',
                           '1x_1sample_mandelbrot_with_bg_aux_largest_capacity/test/',
                           '1x_1sample_mandelbrot_with_bg_all/mean5_test/'],
                   'img_idx': 30,
                   'img_zoom_bbox': [250, 250+80, 570, 570+60],
                   'gt_dir': 'datas_mandelbrot/test_img',
                   'msaa_sample': 5,
                   'print': 'Mandelbrot'
                  },
    'mandelbulb': {'dir': ['1x_1sample_mandelbulb_with_bg_all/test/',
                           '1x_1sample_mandelbulb_with_bg_aux_largest_capacity/test/',
                           '1x_1sample_mandelbulb_with_bg_all/mean2_test/'],
                   'img_idx': 20,
                   'img_zoom_bbox': [250, 250+80, 325, 325+60],
                   'gt_dir': 'datas_mandelbulb/test_img',
                   'msaa_sample': 2,
                   'print': 'Mandelbulb'
                  },
    'primitives': {'dir': ['1x_1sample_primitives_all/test_new/',
                           '1x_1sample_primitives_aux_largest_capacity/test_new/',
                           '1x_1sample_primitives_all/mean1_test/'],
                   'img_idx': 15,
                   'img_zoom_bbox': [420, 420+80, 600, 600+60],
                   'gt_dir': 'datas_primitives_correct_test_range/test_img',
                   'msaa_sample': 1,
                   #'crop_box': [80, -180, 115, -275],
                   'print': 'Gear'
                  },
    'trippy': {'dir': ['1x_1sample_trippy_stratified_subsample_8/test_new/',
               '1x_1sample_trippy_aux_largest_capacity/test_new/',
               '1x_1sample_trippy_subsample_2/mean9_test/'],
               'img_idx': 30,
               'img_zoom_bbox': [550, 550+80, 65, 65+60],
               'gt_dir': 'datas_trippy_new_extrapolation_subsample_2/test_img',
               'msaa_sample': 9,
               'print': 'Trippy Heart'
              },
    'oceanic': {'dir': ['1x_1sample_oceanic_all/test/',
                '1x_1sample_oceanic_aux_largest_capacity/test/',
                '1x_1sample_oceanic_all/mean1_test/'],
                'img_idx': 11,
                'img_zoom_bbox': [475, 475+120, -120, -30],
                'gt_dir': 'datas_oceanic/test_img',
                'msaa_sample': 1,
                'print': 'Oceanic'
               },
    'venice': {'dir': ['1x_1sample_venice_stratified_subsample_3/test_new/',
               '1x_1sample_venice_aux_largest_capacity/test_new/',
               '1x_1sample_venice_all/mean1_test/'],
               'img_idx': 3,
               'img_zoom_bbox': [170, 170+80, 40, 40+60],
               'gt_dir': 'datas_venice_new_extrapolation/test_img',
               'msaa_sample': 1,
               'print': 'Venice'
              }
    },
'simplified': {
    'bricks': {'dir': ['1x_1sample_bricks_simplified_with_bg_all/test/',
               '1x_1sample_bricks_simplified_with_bg_aux_largest_capacity/test/',
               '1x_1sample_bricks_simplified_with_bg_all/mean1_test'],
               'img_idx': 15,
               'gt_dir': 'datas_bricks_normal_texture/test_img',
               'print': 'Bricks'
              },
    'mandelbrot': {'dir': ['1x_1sample_mandelbrot_simplified_with_bg_all/test/',
                   '1x_1sample_mandelbrot_simplified_with_bg_aux_largest_capacity/test/',
                   '1x_1sample_mandelbrot_simplified_all/mean1_test/'],
                   'img_idx': 30,
                   'gt_dir': 'datas_mandelbrot/test_img',
                   'print': 'Mandelbrot'
                  },
    'mandelbulb': {'dir': ['1x_1sample_mandelbulb_with_bg_siimplified_all/test/',
                   '1x_1sample_mandelbulb_simplified_with_bg_aux_largest_capacity/test/',
                   '1x_1sample_mandelbulb_with_bg_siimplified_all/mean1_test/'],
                   'img_idx': 12,
                   'gt_dir': 'datas_mandelbulb/test_img',
                   'print': 'Mandelbulb'
                  },
    'trippy': {'dir': ['1x_1sample_trippy_simplified_stratified_subsample_4/test_new/',
               '1x_1sample_trippy_simplified_aux_largest_capacity/test_new/',
               '1x_1sample_trippy_simplified_all/mean1_test'],
               'img_idx': 30,
               'gt_dir': 'datas_trippy_new_extrapolation_subsample_2/test_img',
               'print': 'Trippy Heart'
              },
    'venice': {'dir': ['1x_1sample_venice_simplified_20_100_stratified_subsample_3/test_new/',
                       '1x_1sample_venice_simplified_20_100_aux_largest_capacity/test_new/',
                       '1x_1sample_venice_simplified_20_100_all/mean1_test/'],
               'img_idx': 18,
               'img_zoom_bbox': [160, 160+120, 150, 150+80],
               'gt_dir': 'datas_venice_new_extrapolation/test_img',
               'input_time_frag': 0.61,
               'print': 'Venice'
              }
    },
'temporal': {
    'mandelbrot_simplified': {'dir': ['1x_1sample_mandelbrot_simplified_proxy_temporal/test',
                              '1x_1sample_mandelbrot_simplified_proxy_temporal_aux/test',
                                     '1x_1sample_mandelbrot_simplified_proxy_all_trace_patchGAN_scale_005_update_8_long/mean1_test/'],
                              'other_view': ['mandelbrot_simplified_temporal_gt_gamma_corrected.png',
                                             '1x_1sample_mandelbrot_simplified_proxy_temporal/test_gamma_corrected/00000827.png',
                                             '1x_1sample_mandelbrot_simplified_proxy_temporal_aux/test_gamma_corrected/00000827.png',
                                             'mandelbrot_simplified_temporal_input_gamma_corrected.png'],
                              'print': 'Simplified Mandelbrot'
                             },
    'mandelbrot': {'dir': ['1x_1sample_mandelbrot_full_temporal_automatic_200_correct_alpha/test',
                   '1x_1sample_mandelbrot_full_temporal_correct_alpha_aux/test',
                          '1x_1sample_mandelbrot_tile_automatic_200_repeat/mean1_test'],
                   'other_view': ['home/global_opt/proj/apps/out/mandelbrot_tile_radius_plane_normal_none/video_gt_gamma_corrected/video_gt00119.png',
                       '1x_1sample_mandelbrot_full_temporal_automatic_200_correct_alpha/render_gamma_corrected/000120.png',
                                  '1x_1sample_mandelbrot_full_temporal_correct_alpha_aux/render_gamma_corrected/000120.png',
                                  'mandelbrot_temporal_inpug_gamma_corrected.png'
                   ],
                   'print': 'Mandelbrot'
                  },
    'mandelbulb_simplified': {'dir': ['1x_1sample_mandelbulb_simplified_temporal_corrected_alpha/test',
                              '1x_1sample_mandelbulb_simplified_temporal_corrected_alpha_aux/test',
                                     '1x_1sample_mandelbulb_simplified_automatic_200/mean1_test/'],
                              'other_view': ['mandelbulb_simplified_temporal_gt.png',
                                             '1x_1sample_mandelbulb_simplified_temporal_corrected/render/000030.png',
                                             '1x_1sample_mandelbulb_simplified_temporal_corrected_aux/render/000030.png',
                                             'mandelbulb_simplified_temporal_input.png'],
                              'print': 'Simplified Mandelbulb'
                             },
    'mandelbulb': {'dir': ['1x_1sample_mandelbulb_full_temporal_larger_cap_correct_alpha/test',
                   '1x_1sample_mandelbulb_full_temporal_larger_cap_correct_alpha_aux/test',
                          '1x_1sample_mandelbulb_automatic_200/mean1_test/'],
                   'other_view': ['/n/fs/shaderml/datas_mandelbulb_full_temporal/test_img/test_ground2900007.png',
                                  '1x_1sample_mandelbulb_full_temporal_larger_cap_correct_alpha/test/00000827.png',
                                  '1x_1sample_mandelbulb_full_temporal_larger_cap_correct_alpha_aux/test/00000827.png',
                                  'mandelbulb_full_temporal_input.png'],
                   'print': 'Mandelbulb'
                  },
    'trippy_simplified': {'dir': ['1x_1sample_trippy_simplified_temporal_corrected_alpha/test_new',
                                  '1x_1sample_trippy_simplified_temporal_corrected_alpha_aux/test_new',
                                  '1x_1sample_trippy_heart_simplified_proxy_dfs_automatic_200/mean1_test'],
                          'other_view': ['/n/fs/shaderml/datas_trippy_temporal/test_img/test_ground2900010.png',
                                         '1x_1sample_trippy_simplified_temporal_corrected_alpha/test/00001127.png',
                                         '1x_1sample_trippy_simplified_temporal_corrected_alpha_aux/test/00001127.png',
                                         'trippy_simplified_temporal_input.png'],
                          'print': 'Simplified Trippy Heart'
                         }
    },
'post_processing': {
    'mandelbulb_blur': {'dir': ['1x_1sample_mandelbulb_with_bg_defocus_blur/test',
                                '1x_1sample_mandelbulb_with_bg_defocus_blur_aux_largest_capacity/test'],
                        'img_idx': 21,
                        'img_zoom_bbox': [320, 320+80, 450, 450+60],
                        'gt_dir': 'datas_mandelbulb_defocus_blur/test_img',
                        'crop_box': [76, 524, 207, -207],
                        'print': 'Mandelbulb Blur'
                       },
    'trippy_sharpen': {'dir': ['1x_1sample_trippy_local_laplacian_stratified_subsample_8/test_new',
                          '1x_1sample_trippy_local_laplacian_aux_largest_capacity/test_new'],
                       'img_idx': 30,
                       'img_zoom_bbox': [240, 240+80, 705, 705+60],
                       'gt_dir': 'datas_trippy_new_extrapolation_local_laplacian_subsample_2/test_img',
                       'print': 'Trippy Heart Sharpen'
                      },
    'trippy_simplified_sharpen': {'dir': ['1x_1sample_trippy_simplified_local_laplacian_stratified_subsample_4/test_new',
                                     '1x_1sample_trippy_simplified_local_laplacian_aux_largest_capacity/test_new'],
                                  'img_idx': 30,
                                  'img_zoom_bbox': [440, 440+80, 550, 550+60],
                                  'gt_dir': 'datas_trippy_new_extrapolation_local_laplacian_subsample_2/test_img',
                                  'print': 'Simplified Trippy Heart Sharpen'
                                 }
    },
'simulation': {
    'boids': {'dir': ['/n/fs/visualai-scr/yutingy/boids_res_20_64_validate_switch_label/test',
              '/n/fs/visualai-scr/yutingy/boids_res_20_64_validate_switch_label_aux/test']}
    }
}

app_names = ['denoising',
             'simplified',
             'post_processing',
             'temporal',
             'simulation']

# DOGE, when data is NOT ready for temporal and simulation
app_names = app_names[:3]
del app_shader_dir_200['temporal']
del app_shader_dir_200['simulation']

max_shader_per_fig = 5

def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'all'
    
    # barplot summary over all apps
    
    bar_x_ticks = []
    bar_avg = []
    bar_dif = []
    bar_sim = []
    bar_col = []
    bar_edge_width = []

    full_shader_idx = []
    simplified_shader_idx = []

    

    slice_start = np.zeros(len(app_names))
    slice_end = np.zeros(len(app_names))

    slice_start[0] = -0.5

    for k in range(len(app_names)):
        app_name = app_names[k]
        if k > 0:
            slice_start[k] = slice_end[k-1] + 1

        app_data = app_shader_dir_200[app_name]
        for shader_name in sorted(app_data.keys()):


            if 'simplified' in shader_name or 'simplified' in app_name:
                bar_edge_width.append(1)
                simplified_shader_idx.append(len(bar_x_ticks))
            else:
                bar_edge_width.append(0)
                full_shader_idx.append(len(bar_x_ticks))

            for name in hue_shaders.keys():
                if shader_name.startswith(name):
                    current_hue = hue_shaders[name]
                    current_col = colorsys.hsv_to_rgb(*hue_shaders[name])
                    bar_x_ticks.append(name)
                    break

            #current_col = colorsys.hsv_to_rgb(current_hue, 1, 1)
            bar_col.append(current_col)

            if app_name == 'post_processing':
                if 'blur' in shader_name:
                    bar_x_ticks[-1] += '_blur'
                elif 'sharpen' in shader_name:
                    bar_x_ticks[-1] += '_sharpen'
                else:
                    raise

            neval = len(app_data[shader_name]['dir'])
            if app_name == 'denoising':
                assert neval == 3
            else:
                assert neval >= 2
                #neval = 2

            score = -np.ones((neval, 3))
            l2_score = -np.ones((neval, 3))

            for i in range(neval):
                
                if app_name in ['denoising', 'simplified'] and i == neval - 1:
                    additional_dir = 'out_inference_time'
                else:
                    additional_dir = ''

                dir = app_data[shader_name]['dir'][i]

                if app_name != 'temporal':
                    perceptual_breakdown_file = os.path.join(model_parent_dir, additional_dir, dir, 'perceptual_tf_breakdown.txt')
                else:
                    perceptual_breakdown_file = os.path.join(model_parent_dir, additional_dir, dir, 'vgg_breakdown.txt') 
                
                perceptual_single_file = os.path.join(model_parent_dir, additional_dir, dir, 'vgg.txt')
                l2_single_file = os.path.join(model_parent_dir, additional_dir, dir, 'all_loss.txt')

                if app_name in ['denoising', 'simplified', 'post_processing', 'temporal']:
                    
                    if os.path.exists(perceptual_breakdown_file):
                        perceptual_scores = open(perceptual_breakdown_file).read()
                    else:
                        assert allow_missing_MSAA and app_name in ['denoising', 'simplified'] and i == neval - 1, perceptual_breakdown_file
                        perceptual_scores = '1,1,1'
                    
                    perceptual_scores.replace('\n', '')
                    perceptual_scores.replace(' ', '')
                    perceptual_scores = perceptual_scores.split(',')
                    perceptual_scores = [float(score) for score in perceptual_scores]
                    score[i][0] = perceptual_scores[2]
                    score[i][1] = (perceptual_scores[0] + perceptual_scores[1]) / 2.0
                    score[i][2] = (perceptual_scores[0] * 5 + perceptual_scores[1] * 5 + perceptual_scores[2] * 20) / 30
                #elif app_name == 'temporal':
                #    perceptual_scores = open(perceptual_single_file).read()
                #    score[i][2] = float(perceptual_scores)
                elif app_name == 'simulation':
                    l2_scores = open(l2_single_file).read()
                    l2_scores.replace('\n', '')
                    l2_scores.replace(' ', '')
                    l2_scores = l2_scores.split(',')
                    l2_scores = [float(score) for score in l2_scores]
                    score[i][0] = l2_scores[1]
                    score[i][1] = l2_scores[0]
                    score[i][2] = l2_scores[1]
                else:
                    raise

                if app_name not in ['simulation']:
                    
                    l2_breakdown_file = os.path.join(model_parent_dir, additional_dir, dir, 'score_breakdown.txt')
                    
                    if os.path.exists(l2_breakdown_file):
                        l2_scores = open(l2_breakdown_file).read()
                    else:
                        assert allow_missing_MSAA and app_name in ['denoising', 'simplified'] and i == neval - 1
                        l2_scores = '1,1,1'
                        
                    l2_scores.replace('\n', '')
                    l2_scores.replace(' ', '')
                    l2_scores = l2_scores.split(',')
                    l2_scores = [float(score) for score in l2_scores]
                    l2_score[i][0] = l2_scores[2]
                    l2_score[i][1] = (l2_scores[0] + l2_scores[1]) / 2.0
                    l2_score[i][2] = (l2_scores[0] * 5 + l2_scores[1] * 5 + l2_scores[2] * 20) / 30

            if app_name == 'temporal':
                # scale for temporal perceptual needs to be adjusted
                score /= 0.04
            
            bar_avg.append(score[0, 2] / score[1, 2])
            if score[0, 1] > 0:
                bar_dif.append(score[0, 1] / score[1, 1])
                bar_sim.append(score[0, 0] / score[1, 0])
            else:
                bar_dif.append(None)
                bar_sim.append(None)

            app_data[shader_name]['perceptual'] = score
            if app_name not in ['simulation']:
                app_data[shader_name]['l2'] = l2_score

        slice_end[k] = len(bar_x_ticks) - 1 + 0.5
        bar_x_ticks.append('')
        bar_avg.append(0)
        bar_dif.append(0)
        bar_sim.append(0)
        bar_col.append((0, 0, 0))
        bar_edge_width.append(0)

    if 'bar' in mode or mode == 'all':
        
        fontsize = 14
        
        bar_x = np.arange(len(bar_x_ticks))
        fig = plt.figure()
        fig.set_size_inches(9, 4.5)

        ax = plt.subplot(111)
        #ax.set_aspect(1.0)
        plt.bar(bar_x[full_shader_idx], [bar_avg[i] for i in full_shader_idx], color=[bar_col[i] for i in full_shader_idx], edgecolor='k', linestyle='-')
        plt.bar(bar_x[simplified_shader_idx], [bar_avg[i] for i in simplified_shader_idx], color=[bar_col[i] for i in simplified_shader_idx], edgecolor='k', linestyle='--')
        plt.xticks(bar_x, bar_x_ticks, rotation=90, fontsize=fontsize)
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(fontsize)
        

        plt.text(2.5, 1.05, 'denoising', fontsize=fontsize)
        plt.text(10.5, 1.05, 'simplified', fontsize=fontsize)
        plt.text(16.7, 1.05, 'temporal', fontsize=fontsize)
        plt.text(22.3, 1.05, 'post', fontsize=fontsize)
        plt.text(25.2, 1.05, 'sim', fontsize=fontsize)


        #plt.text(0, 0.9, 'baseline', withdash=True, color='b')

        for k in range(len(app_names)):
            #col = colorsys.hsv_to_rgb(0.0, k / len(app_names), k / len(app_names))
            col = [0, 0, 0]
            plt.axvspan(slice_start[k], slice_end[k], facecolor=col, alpha=0.3)
            
        plt.xlim(slice_start[0], slice_end[-1])

        plt.plot(np.arange(-1, len(bar_x_ticks)+1), np.ones(len(bar_x_ticks)+2), 'k-', label='full shader')
        plt.plot(np.arange(-1, len(bar_x_ticks)+1), np.ones(len(bar_x_ticks)+2), 'k--', label='simplified shader')

        plt.plot(np.arange(-1, len(bar_x_ticks)+1), np.ones(len(bar_x_ticks)+2), label='baseline')

        plt.legend(loc='upper right', prop={'size': fontsize}, framealpha=0.9, bbox_to_anchor=(0.8, 0.5))

        plt.ylim(0, 1.15)
        plt.ylabel('relative error', fontsize=fontsize)

        plt.tight_layout()
        plt.savefig('result_figs/bar_summary.png')
        plt.close(fig)
        
        
        
        
        
        permutation = [0, 1, 15, 2, 17, 3, 4, 5, 6, 7, 8, 9, 10, 16, 11, 18, 12, 19, 13, 14, 20, 21, 22, 23, 24, 25, 26]
        #bar_x_ticks = [bar_x_ticks[i] for i in permutation]
        #bar_avg = [bar_avg[i] for i in permutation]
        #bar_col = [bar_col[i] for i in permutation]
        #bar_edge_width = [bar_edge_width[i] for i in permutation]
        
        #bar_x = bar_x[permutation]
        bar_x = np.array([0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 19, 2, 13, 4, 15, 17, 19, 20, 21, 22, 23, 24, 25])
        fig = plt.figure()
        fig.set_size_inches(9, 4.5)
        
        full_shader_idx = [0, 1, 2, 3, 4, 5, 6, 7, 21, 22, 25]
        simpliifed_shader_idx = [9, 10, 11, 12, 13, 23]
        full_temporal_idx = [15, 17]
        simplified_temporal_idx = [16, 18, 19]
        
        if allow_missing_temporal:
            full_shader_idx = [0, 1, 2, 3, 4, 5, 6, 14, 15]
            simpliifed_shader_idx = [8, 9, 10, 11, 12, 16]
            full_temporal_idx = []
            simplified_temporal_idx = []

        ax = plt.subplot(111)
        #ax.set_aspect(1.0)
        plt.bar(bar_x[full_shader_idx], [bar_avg[i] for i in full_shader_idx], color=[bar_col[i] for i in full_shader_idx], edgecolor='k', linestyle='-')
        plt.bar(bar_x[simpliifed_shader_idx], [bar_avg[i] for i in simpliifed_shader_idx], color=[bar_col[i] for i in simpliifed_shader_idx], edgecolor='k', linestyle='-', hatch='\\\\')
        plt.bar(bar_x[full_temporal_idx], [bar_avg[i] for i in full_temporal_idx], color=[bar_col[i] for i in full_temporal_idx], edgecolor='k', linestyle='-', hatch='//')
        plt.bar(bar_x[simplified_temporal_idx], [bar_avg[i] for i in simplified_temporal_idx], color=[bar_col[i] for i in simplified_temporal_idx], edgecolor='k', linestyle='-', hatch='xx')
        
        #bar = ax.bar(bar_x[simplified_shader_idx], [bar_avg[i] for i in simplified_shader_idx], color=[bar_col[i] for i in simplified_shader_idx], edgecolor='k', linestyle='--', hatch='xx')
        plt.xticks(bar_x, bar_x_ticks, rotation=90, fontsize=fontsize)
        
        invisible_full = matplotlib.patches.Patch(facecolor='#DCDCDC', label='full')
        invisible_simplified = matplotlib.patches.Patch(facecolor='#DCDCDC', label='full', hatch='\\\\')
        invisible_temporal = matplotlib.patches.Patch(facecolor='#DCDCDC', label='full', hatch='//')
        
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(fontsize)
        

        plt.text(3.0, 1.07, 'denoising', fontsize=fontsize)
        plt.text(13.5, 1.07, 'simplified', fontsize=fontsize)
        plt.text(20.3, 1.07, 'post', fontsize=fontsize)
        plt.text(23.4, 1.07, 'sim', fontsize=fontsize)
        
        

        plt.axvspan(-0.5, 9.5, facecolor=[0, 0, 0], alpha=0.1)
        plt.axvspan(10.5, 18.5, facecolor=[0, 0, 0], alpha=0.1)
        plt.axvspan(19.5, 22.5, facecolor=[0, 0, 0], alpha=0.1)
        plt.axvspan(23.5, 24.5, facecolor=[0, 0, 0], alpha=0.1)
            
        plt.xlim(-0.5, 24.5)

        plt.plot(np.arange(-1, len(bar_x_ticks)+1), np.ones(len(bar_x_ticks)+2), 'k-', label='full shader')
        plt.plot(np.arange(-1, len(bar_x_ticks)+1), np.ones(len(bar_x_ticks)+2), 'k--', label='simplified shader')

        line, = plt.plot(np.arange(-1, len(bar_x_ticks)+1), np.ones(len(bar_x_ticks)+2), label='baseline')
        
        plt.text(8.7, 0.97, 'baseline', bbox=dict(facecolor='white', alpha=1, edgecolor=line.get_color(), linewidth=line.get_linewidth(), boxstyle=BoxStyle("Round", pad=0.5)), fontsize=fontsize, color=line.get_color())

        plt.legend(loc='upper right', prop={'size': fontsize}, framealpha=0.9, bbox_to_anchor=(0.8, 0.5))
        
        ax.legend([invisible_full, invisible_simplified, invisible_temporal], ['full', 'simplified', 'temporal'], fontsize=fontsize, loc='upper right', framealpha=0.9, bbox_to_anchor=(1, 0.45))

        plt.ylim(0, 1.19)
        plt.ylabel('Relative Error', fontsize=fontsize)

        plt.tight_layout()
        plt.savefig('result_figs/bar_summary_v2.png')
        plt.close(fig)
    
    
    if 'table' in mode or mode == 'all':
        # table

        str = ""

        for k in range(len(app_names)):

            app_name = app_names[k]
            
            if app_name in ['simulation']:
                continue

            avg_ratio = np.empty([len(app_shader_dir_200[app_name].keys()), 2, 3])
                
            str += """
    \\vspace{-1ex}
    \setlength{\\tabcolsep}{2.0pt}
    \\begin{table}[]
    \\begin{tabular}{c|ccccc}
    \\hline

        \multirow{2}{*}{Shader} &  & \multicolumn{2}{c}{Perceptual} & \multicolumn{2}{c}{L2 Error} \\\\ \cline{3-6} 
        & Distances: & Similar & Different & Similar & Different \\\\ \\thickhline
    """

            #for shader_name in sorted(app_shader_dir_200[app_name].keys()):
            for i in range(len(app_shader_dir_200[app_name].keys())):
                
                shader_name = sorted(app_shader_dir_200[app_name].keys())[i]
                
            
                data = app_shader_dir_200[app_name][shader_name]
                
                avg_ratio[i, 0] = data['l2'][0] / data['l2'][1]
                avg_ratio[i, 1] = data['perceptual'][0] / data['perceptual'][1]


                argmin_perceptual = np.argmin(data['perceptual'], 0)
                argmin_l2 = np.argmin(data['l2'], 0)
                if app_name == 'denoising':
                    row_data_rel = [1, 0, 2]
                else:
                    row_data_rel = [1, 0]
                count = 0

                data_strs = [None] * 4 * len(row_data_rel)

                for row in range(len(row_data_rel)):
                    for col in range(4):
                        data_row = row_data_rel[row]
                        if col < 2:
                            field = 'perceptual'
                            idx = col
                            argmin_idx = argmin_perceptual[idx]
                        else:
                            field = 'l2'
                            idx = col - 2
                            argmin_idx = argmin_l2[idx]

                        if row == 0:
                            data_strs[count] = '%.1e' % data[field][1, idx]
                        else:
                            data_strs[count] = '%02d' % (data[field][data_row, idx] / data[field][1, idx] * 100)

                        if data_row == argmin_idx:
                            data_strs[count] = '\\textbf{' + data_strs[count] + '}'

                        count += 1

                print_name = shader_name
                if print_name not in hue_shaders.keys():
                    print_name = print_name.replace('_', '\\\\', 1)
                    print_name = print_name.replace('_', '\\ ')



                if app_name == 'denoising':
                    str += """
        \multicolumn{1}{c|}{\multirow{3}{*}{\\begin{tabular}[c]{@{}c@{}}\\%s \end{tabular}}} & RGBx & %s & %s & %s & %s \\\\ \cline{2-6} 
        \multicolumn{1}{c|}{} & Ours & %s\%% & %s\%% & %s\%% & %s\%% \\\\ \cline{2-6}
        \multicolumn{1}{c|}{} & MSAA & %s\%% & %s\%% & %s\%% & %s\%% \\\\ \\thickhline""" % ((print_name, ) + tuple(data_strs))
                else:
                    str += """
    \multicolumn{1}{c|}{\multirow{2}{*}{\\begin{tabular}[c]{@{}c@{}}\\%s \end{tabular}}} & RGBx & %s & %s & %s & %s \\\\ \cline{2-6} 
        \multicolumn{1}{c|}{} & Ours & %s\%% & %s\%% & %s\%% & %s\%% \\\\ \\thickhline""" % ((print_name, ) + tuple(data_strs))

            
            print('avg ratio for', app_name)
            print(numpy.mean(avg_ratio, 0))
            numpy.save('avg_ratio_%s'% app_name, avg_ratio)
            
            str = str[:-11] + '\hline' + """
    \end{tabular}
    \caption{%s}
    \end{table}
    """ % app_name.replace('_', '\\ ')

        open('result_figs/supplemental.tex', 'w').write(str)
    
    
    if 'fig' in mode or mode == 'all':
        # result image for all apps
        crop_edge_col = [12, 144, 36]
        bbox_edge_w = 3
        
        post_processing_w = 780
        post_processing_cropped = (960 - post_processing_w) // 2

        str = ''

        no_zoom_defined = False

        for k in range(len(app_names)):
            
            app_name = app_names[k]
            
            if app_name in ['temporal', 'simulation']:
                continue
            
            fig_start = True
            fig_rows = 0


            app_data = app_shader_dir_200[app_name]
            for shader_name in sorted(app_data.keys()):

                if app_name == teaser_app and shader_name == teaser_shader:
                    # will generate a seperate teaser
                    continue

                data = app_data[shader_name]

                if shader_name == 'oceanic':
                    print('here')

                if 'img_idx' in data.keys() or 'other_view' in data.keys():

                    if 'img_idx' in data.keys():
                        if shader_name == 'mandelbrot' and use_gamma_corrected:
                            for i in range(len(data['dir'])):
                                if data['dir'][i].endswith('/'):
                                    data['dir'][i] = data['dir'][i][:-1]
                                data['dir'][i] = data['dir'][i] + '_gamma_corrected'
                            if data['gt_dir'].endswith('/'):
                                data['gt_dir'] = data['gt_dir']
                            data['gt_dir'] = data['gt_dir'] + '_gamma_corrected'

                        orig_imgs = []
                        for i in range(len(data['dir'])):
                            
                            if app_name in ['denoising', 'simplified'] and i == len(data['dir']) - 1:
                                additional_dir = 'out_inference_time'
                            else:
                                additional_dir = ''
                                
                            if app_name == 'temporal':
                                orig_img_name = os.path.join(model_parent_dir, additional_dir, data['dir'][i], '%06d27.png' % data['img_idx'])
                            else:
                                orig_img_name = os.path.join(model_parent_dir, additional_dir, data['dir'][i], '%06d.png' % data['img_idx'])
                                
                            if os.path.exists(orig_img_name):
                                orig_imgs.append(skimage.io.imread(orig_img_name))
                            else:
                                assert allow_missing_MSAA and app_name in ['denoising', 'simplified'] and i == len(data['dir']) - 1, orig_img_name
                                orig_imgs.append(np.copy(img_pl))

                        
                        if app_name == 'temporal':
                            gt_img = skimage.io.imread(os.path.join(dataset_parent_dir, data['gt_dir'], '29%05d.png' % (data['img_idx']-1)))
                        else:
                            gt_files = sorted(os.listdir(os.path.join(dataset_parent_dir, data['gt_dir'])))
                            gt_img = skimage.io.imread(os.path.join(dataset_parent_dir, data['gt_dir'], gt_files[data['img_idx']-1]))

                        orig_imgs = [gt_img] + orig_imgs
                    else:
                        orig_imgs = []
                        for i in range(len(data['other_view'])):
                            orig_imgs.append(skimage.io.imread(data['other_view'][i]))


                    for i in range(len(orig_imgs)):

                        if i == 0:
                            prefix = 'gt'
                        elif i == 1:
                            prefix = 'ours'
                        elif i == 2:
                            prefix = 'RGBx'
                        elif i == 3:
                            if app_name == 'denoising':
                                prefix = 'MSAA'
                            else:
                                prefix = 'input'
                        else:
                            raise

                        img = orig_imgs[i]

                        if app_name in ['denoising', 'post_processing']:
                            bbox = data['img_zoom_bbox']
                            crop1 = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
                            crop_w = 2

                            for c in range(len(crop_edge_col)):
                                crop1[:crop_w, :, c] = crop_edge_col[c]
                                crop1[-crop_w:, :, c] = crop_edge_col[c]
                                crop1[:, :crop_w, c] = crop_edge_col[c]
                                crop1[:, -crop_w:, c] = crop_edge_col[c]

                            skimage.io.imsave(os.path.join('result_figs', '%s_%s_%s_zoom.png' % (app_name, shader_name, prefix)), crop1)

                            for i in range(len(bbox)):
                                edge = bbox[i]
                                for current_draw in range(edge-bbox_edge_w, edge+bbox_edge_w+1):
                                    for c in range(len(crop_edge_col)):
                                        if i < 2:
                                            img[current_draw, bbox[2]-bbox_edge_w:bbox[3]+bbox_edge_w, c] = crop_edge_col[c]
                                        else:
                                            img[bbox[0]-bbox_edge_w:bbox[1]+bbox_edge_w, current_draw, c] = crop_edge_col[c]

                            
                            if 'crop_box' in data:
                                crop_box = data['crop_box']
                                img = img[crop_box[0]:crop_box[1], crop_box[2]:crop_box[3]]
                            elif app_name == 'post_processing':
                                img = img[:, post_processing_cropped:-post_processing_cropped]
                            
                            skimage.io.imsave(os.path.join('result_figs', '%s_%s_%s_box.png' % (app_name, shader_name, prefix)), img)
                        else:
                            skimage.io.imsave(os.path.join('result_figs', '%s_%s_%s.png' % (app_name, shader_name, prefix)), img)

                    if app_name == 'denoising':
                        macro_suffix = 'WithZoom'
                    elif app_name == 'post_processing':
                        macro_suffix = 'ThreeCol'
                    else:
                        macro_suffix = 'WithoutZoom'

                    #if fig_start:
                    if fig_rows == 0 or fig_rows >= max_shader_per_fig:
                        macro = '\ResultsFigWithHeader' + macro_suffix
                        fig_start = False
                        if fig_rows == 0:
                            str += """
    \\begin{figure*}
    """
                        elif fig_rows >= max_shader_per_fig:
                            fig_rows -= max_shader_per_fig
                            str += """
    \\vspace{-2ex}
    \caption{%s}
    \end{figure*}

    \\begin{figure*}
    """ % app_name
                    else: 
                        macro = '\ResultsFigNoHeader' + macro_suffix

                    fig_rows += 1

                    print_name = shader_name
                    if False:
                    #if '_simplified' in shader_name:
                        for name in hue_shaders.keys():
                            if shader_name.startswith(name):
                                print_name = name + '\\ simplified'
                                break
                    print_name = print_name.replace('_', '\\ ')

                    if app_name == 'denoising':
                        str += """
    %s{\%s}{%s%s_%s}{0\%%}{%d\%%}{100\%%}{%d\,SPP, %d\%%}
    """ % (macro, print_name, tex_prefix, app_name, shader_name, int(data['perceptual'][0, 2] / data['perceptual'][1, 2] * 100), data['msaa_sample'], int(data['perceptual'][2, 2] / data['perceptual'][1, 2] * 100))
                    elif app_name == 'post_processing':
                        str += """
    %s{\%s: 0\%%}{%s%s_%s}{100\%%}{%d\%%}
    """ % (macro, print_name, tex_prefix, app_name, shader_name, int(data['perceptual'][0, 2] / data['perceptual'][1, 2] * 100))
                    else:
                        str += """
    %s{\%s: 0\%%}{%s%s_%s}{%d\%%}{100\%%}{%d\%%}
    """ % (macro, print_name, tex_prefix, app_name, shader_name, int(data['perceptual'][0, 2] / data['perceptual'][1, 2] * 100), int(data['perceptual'][2, 2] / data['perceptual'][1, 2] * 100))

            if not fig_start:
                str += """
    \\vspace{-2ex}
    \caption{%s}
    \end{figure*}
    """ % app_name.replace('_', ' ')

        open('result_figs/result_pool.tex', 'w').write(str)
    
    if 'teaser' in mode or mode == 'all':
        # generate teaser tex
        
        crop_edge_col = [12, 144, 36]
        bbox_edge_w = 3

        app_name = teaser_app
        shader_name = teaser_shader
        data = app_shader_dir_200[app_name][shader_name]

        orig_img_w = 960
        orig_img_h = 640

        img_w = 630
        zoom_w = 320

        #orig_imgs = []
        #for i in range(len(data['other_view'])):
        #    orig_imgs.append(skimage.io.imread(data['other_view'][i]))
            
        orig_imgs = []
        for i in range(len(data['dir'])):

            if app_name in ['denoising', 'simplified'] and i == len(data['dir']) - 1:
                additional_dir = 'out_inference_time'
            else:
                additional_dir = ''

            orig_img_name = os.path.join(model_parent_dir, additional_dir, data['dir'][i], '%06d.png' % data['img_idx'])

            if os.path.exists(orig_img_name):
                orig_imgs.append(skimage.io.imread(orig_img_name))
            else:
                assert allow_missing_MSAA and app_name in ['denoising', 'simplified'] and i == len(data['dir']) - 1, orig_img_name
                orig_imgs.append(np.copy(img_pl))

        gt_files = sorted(os.listdir(os.path.join(dataset_parent_dir, data['gt_dir'])))
        gt_img = skimage.io.imread(os.path.join(dataset_parent_dir, data['gt_dir'], gt_files[data['img_idx']-1]))

        orig_imgs = [gt_img] + orig_imgs

        crop_size = (orig_img_w - img_w) // 2

        for i in range(len(orig_imgs)):
            orig_imgs[i] = orig_imgs[i][:, crop_size:-crop_size, :]
            img = orig_imgs[i]

            if i == 0:
                prefix = 'gt'
            elif i == 1:
                prefix = 'ours'
            elif i == 2:
                prefix = 'RGBx'
            elif i == 3:
                if app_name == 'denoising':
                    prefix = 'MSAA'
                else:
                    prefix = 'input'
            else:
                raise

            bbox = data['img_zoom_bbox']
            crop1 = img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            crop_w = 2

            for c in range(len(crop_edge_col)):
                crop1[:crop_w, :, c] = crop_edge_col[c]
                crop1[-crop_w:, :, c] = crop_edge_col[c]
                crop1[:, :crop_w, c] = crop_edge_col[c]
                crop1[:, -crop_w:, c] = crop_edge_col[c]

            skimage.io.imsave(os.path.join('result_figs', 'teaser_%s_%s_%s_zoom.png' % (app_name, shader_name, prefix)), crop1)

            for i in range(len(bbox)):
                edge = bbox[i]
                for current_draw in range(edge-bbox_edge_w, edge+bbox_edge_w+1):
                    for c in range(len(crop_edge_col)):
                        if i < 2:
                            img[current_draw, bbox[2]-bbox_edge_w:bbox[3]+bbox_edge_w, c] = crop_edge_col[c]
                        else:
                            img[bbox[0]-bbox_edge_w:bbox[1]+bbox_edge_w, current_draw, c] = crop_edge_col[c]

            skimage.io.imsave(os.path.join('result_figs', 'teaser_%s_%s_%s_box.png' % (app_name, shader_name, prefix)), img)

        str = """

    \\begin{teaserfigure}
    \\vspace{1ex}    
    \ResultsFigTeaser{\%s: 100\%% time, 0\%% error}{%steaser_%s_%s}{%.2f\%% time, %d\%% error}{100\%% error}{%d\%% error}
    \\vspace{-1ex}
    \caption{pl}
    \\vspace{2ex}
    \label{fig:teaser}
    \end{teaserfigure}

    """ % (shader_name, tex_prefix, app_name, shader_name, data['input_time_frag'], int(data['perceptual'][2, 2] / data['perceptual'][1, 2] * 100), int(data['perceptual'][0, 2] / data['perceptual'][1, 2] * 100))


        open('result_figs/teaser.tex', 'w').write(str)
    
    if 'html' in mode or mode == 'all':
        # generate images for html viewer
        base_dir = 'result_figs/html_viewer'
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
            
        app_name_str = ''

        for k in range(len(app_names)):
            app_name = app_names[k]

            if app_name == 'simulation':
                continue

            if app_name == 'post_processing':
                app_name_dir = 'post'
                app_name_print = 'Post Processing'
            elif app_name == 'simplified':
                app_name_dir = 'simplification'
                app_name_print = 'Simplification'
            else:
                app_name_dir = app_name
                if app_name == 'denoising':
                    app_name_print = 'Denoising'
                elif app_name == 'temporal':
                    app_name_print = 'Temporal Coherence'
                else:
                    raise
                    
            app_name_str += '%s,%s\n' % (app_name_dir, app_name_print)
            
            shader_name_str = ''

            app_name_dir = os.path.join(base_dir, app_name_dir)
            if not os.path.exists(app_name_dir):
                os.mkdir(app_name_dir)
                
            if app_name == 'denoising':
                shader_names = ['venice',
                                'oceanic',
                                'trippy',
                                'mandelbulb',
                                'mandelbrot',
                                'primitives',
                                'marble2',
                                'bricks'
                               ]
            elif app_name == 'simplified':
                shader_names = ['venice',
                                'trippy',
                                'mandelbulb',
                                'mandelbrot',
                                'bricks'
                               ]
            elif app_name == 'post_processing':
                shader_names = ['mandelbulb_blur',
                                'trippy_sharpen',
                                'trippy_simplified_sharpen'
                               ]
            elif app_name == 'temporal':
                shader_names = ['trippy_simplified',
                               'mandelbrot',
                               'mandelbulb',
                               'mandelbrot_simplified',
                               'mandelbulb_simplified'
                ]
            
            #for shader_name in app_shader_dir_200[app_name].keys():
            for shader_name in shader_names:
                shader_dir = os.path.join(app_name_dir, shader_name)
                if not os.path.exists(shader_dir):
                    os.mkdir(shader_dir)

                data = app_shader_dir_200[app_name][shader_name]

                if 'img_idx' in data.keys() or 'other_view' in data.keys():
                    
                    shader_name_str += '%s,%s\n' % (shader_name, data['print'])
                    
                    conditions_str = ''

                    if 'img_idx' in data.keys():
                        
                        if shader_name == 'mandelbrot':
                            for i in range(len(data['dir'])):
                                if data['dir'][i].endswith('/'):
                                    data['dir'][i] = data['dir'][i][:-1]
                                if 'gamma_corrected' not in data['dir'][i]:
                                    data['dir'][i] = data['dir'][i] + '_gamma_corrected'
                            if data['gt_dir'].endswith('/'):
                                data['gt_dir'] = data['gt_dir']
                            if 'gamma_corrected' not in data['gt_dir']:
                                data['gt_dir'] = data['gt_dir'] + '_gamma_corrected'


                        orig_imgs = []
                        for i in range(len(data['dir'])):
                            
                            if shader_name == 'denoising' and i == len(data['dir']) - 1:
                                additional_dir = 'out_inference_time'
                            else:
                                additional_dir = ''
                    
                            if app_name == 'temporal':
                                orig_imgs.append(os.path.join(model_parent_dir, additional_dir, data['dir'][i], '%06d27.png' % data['img_idx']))
                            else:
                                orig_imgs.append(os.path.join(model_parent_dir, additional_dir, data['dir'][i], '%06d.png' % data['img_idx']))
                        if app_name == 'temporal':
                            gt_img = os.path.join(dataset_parent_dir, data['gt_dir'], '29%05d.png' % (data['img_idx']-1))
                        else:
                            gt_files = sorted(os.listdir(os.path.join(dataset_parent_dir, data['gt_dir'])))
                            gt_img = os.path.join(dataset_parent_dir, data['gt_dir'], gt_files[data['img_idx']-1])

                        orig_imgs = [gt_img] + orig_imgs
                    else:
                        orig_imgs = []
                        for i in range(len(data['other_view'])):
                            orig_imgs.append(data['other_view'][i])

                    for i in range(len(orig_imgs)):
                        src = orig_imgs[i]
                        if i == 0:
                            dst_name = 'reference'
                            condition_print = 'Reference'
                            rel_perceptual = 0
                            rel_l2 = 0
                        elif i == 1:
                            dst_name = 'ours'
                            condition_print = 'Ours'
                            rel_perceptual = int(100 * data['perceptual'][0, 2] / data['perceptual'][1, 2])
                            rel_l2 = int(100 * data['l2'][0, 2] / data['l2'][1, 2])
                        elif i == 2:
                            dst_name = 'baseline'
                            condition_print = 'RGBx Baseline'
                            rel_perceptual = 100
                            rel_l2 = 100
                        elif i == 3:
                            if app_name == 'denoising':
                                dst_name = 'baseline2'
                                condition_print = 'SuperSampling Baseline'
                            else:
                                dst_name = 'input'
                                condition_print = 'Input'
                            try:
                                rel_perceptual = int(100 * data['perceptual'][2, 2] / data['perceptual'][1, 2])
                                rel_l2 = int(100 * data['l2'][2, 2] / data['l2'][1, 2])
                            except:
                                print(shader_name, app_name)
                                raise
                        else:
                            raise
                        dst = os.path.join(shader_dir, dst_name + '.png')
                        shutil.copyfile(src, dst)
                        
                        caption = '%s / %s / %s' % (app_name_print, data['print'], condition_print)
                        #if dst_name != 'input':
                        caption += ' / Relative Perceptual Error: %d%%; Relative L2 Error: %d%%.' % (rel_perceptual, rel_l2)
                        if app_name == 'temporal' and shader_name == 'mandelbulb_simplified':
                            caption = '"' + caption + ' \nIn this example, input is having less L2 error than RGBx baseline because the baseline is unable to synthesize to longer sequences."'
                        conditions_str += '%s,%s,%s\n' % (dst_name + '.png', condition_print, caption)
                        
                    open(os.path.join(shader_dir, 'conditions.csv'), 'w').write(conditions_str)
            
            open(os.path.join(app_name_dir, 'shaders.csv'), 'w').write(shader_name_str)
            
        open(os.path.join(base_dir, 'applications.csv'), 'w').write(app_name_str)
                
    if False:
        names_and_dirs = [('\\begin{tabular}[c]{@{}c@{}}\\oceanic \\\\ Fig. 1 \\\\ \\tracelen\ = 702\\end{tabular}',
                          '1x_1sample_oceanic_simple_tile_sigma_03_continued/test/',
                          '1x_1sample_oceanic_simple_tile_sigma_03_aux_continued/test/',
                          '1x_1sample_oceanic_simple_tile_sigma_03_continued/mean1_test'),
                         ('\\begin{tabular}[c]{@{}c@{}}\\bricks \\\\ Fig. 2 \\\\ \\tracelen\ = 56\\end{tabular}',
                          '1x_1sample_bricks_tile_sigma_03_fixed_scale_continued/test',
                          '1x_1sample_bricks_tile_sigma_03_aux_continued/test',
                          '1x_1sample_bricks_tile_sigma_03_fixed_scale_continued/mean1_test'),
                         ('\\begin{tabular}[c]{@{}c@{}}\\mandelbulb \\\\ Fig. 2 \\\\ \\tracelen\ = 232\\end{tabular}',
                          '1x_1sample_mandelbulb_tile_more_trace_sigma_03_continued/test',
                          '1x_1sample_mandelbulb_tile_sigma_03_aux_continued/test',
                          '1x_1sample_mandelbulb_tile_more_trace_sigma_03_continued/mean3_test'),
                         ('\\begin{tabular}[c]{@{}c@{}}\\marble \\\\ Fig. 2 \\\\ \\tracelen\ = 226\\end{tabular}',
                          '1x_1sample_marble_tile_sigma_03_continued/test',
                          '1x_1sample_marble_tile_sigma_03_aux_continued/test',
                          '1x_1sample_marble_tile_sigma_03_continued/mean1_test'),
                         ('\\begin{tabular}[c]{@{}c@{}}\\Texture Map \\\\ Fig. 2 \\\\ \\tracelen\ = 80\\end{tabular}',
                          '1x_1sample_texture_map_sigma_03_continued/test',
                          '1x_1sample_texture_map_sigma_03_aux_continued/test',
                          '1x_1sample_texture_map_sigma_03/mean1_test'),
                         ('\\begin{tabular}[c]{@{}c@{}}\\primitives \\\\ Fig. 5 \\\\ \\tracelen\ = 771\\end{tabular}',
                          '1x_1sample_primitives_tile_scale_800_sigma_03_continued/test',
                          '1x_1sample_primitives_tile_scale_800_sigma_03_aux_continued/test',
                          '1x_1sample_primitives_tile_scale_800_sigma_03_continued/mean1_test'),
                         ('\\begin{tabular}[c]{@{}c@{}}\\trippy \\\\ Fig. 5 \\\\ \\tracelen\ = 199\\end{tabular}',
                          '1x_1sample_trippy_heart_tile_continued/test',
                          '1x_1sample_trippy_heart_tile_aux_continued/test',
                          '1x_1sample_trippy_heart_tile_continued/mean4_test'),
                         ('\\begin{tabular}[c]{@{}c@{}}\\mandelbrot \\\\ Fig. 5 \\\\ \\tracelen\ = 131\\end{tabular}',
                          '1x_1sample_mandelbrot_tile_continued/test',
                          '1x_1sample_mandelbrot_tile_aux_continued/test',
                          '1x_1sample_mandelbrot_tile_continued/mean2_test')]

        str = """
        \multicolumn{1}{c|}{\multirow{2}{*}{Shader}} & \multicolumn{1}{c}{\multirow{2}{*}{}} & \multicolumn{2}{c}{Perceptual (\%)} & \multicolumn{2}{c}{L2 Error (\%)} \\\\ \cline{3-6}
        \multicolumn{1}{c|}{} & Distances: & Similar & Different & Similar & Different  \\\\ \\thickhline
        """

        for ind in range(len(names_and_dirs)):
            name, our_dir, aux_dir, mean_dir = names_and_dirs[ind]

            all_dirs = [our_dir, aux_dir, mean_dir]
            score = [['', '', '', ''],
                     ['', '', '', ''],
                     ['', '', '', '']]
            min_score = 10000 * numpy.ones(4)

            for i in range(len(all_dirs)):
                dir = all_dirs[i]
                if dir is not None:
                    l2_breakdown_file = os.path.join(model_parent_dir, dir, 'score_breakdown.txt')
                    perceptual_breakdown_file = os.path.join(model_parent_dir, dir, 'perceptual_tf_breakdown.txt')

                    l2_scores = open(l2_breakdown_file).read()
                    l2_scores.replace('\n', '')
                    l2_scores.replace(' ', '')
                    l2_scores = l2_scores.split(',')
                    l2_scores = [float(score) for score in l2_scores]
                    score[i][2] = '%.2f' % (100 * (l2_scores[2] ** 0.5) / 255.0)
                    score[i][3] = '%.2f' % (100 * (((l2_scores[0] + l2_scores[1]) / 2.0) ** 0.5) / 255.0)

                    perceptual_scores = open(perceptual_breakdown_file).read()
                    perceptual_scores.replace('\n', '')
                    perceptual_scores.replace(' ', '')
                    perceptual_scores = perceptual_scores.split(',')
                    perceptual_scores = [float(score) for score in perceptual_scores]
                    score[i][0] = '%.2f' % (100 * perceptual_scores[2])
                    score[i][1] = '%.2f' % (100 * (perceptual_scores[0] + perceptual_scores[1]) / 2.0)

                    for k in range(4):
                        if float(score[i][k]) < min_score[k]:
                            min_score[k] = float(score[i][k])

                else:
                    score[i][0] = 'TBD'
                    score[i][1] = 'TBD'
                    score[i][2] = 'TBD'
                    score[i][3] = 'TBD'

            for i in range(len(all_dirs)):
                for k in range(4):
                    try:
                        if float(score[i][k]) == min_score[k]:
                            score[i][k] = '\\textbf{%s}' % score[i][k]
                    except:
                        pass

            str_add = """
        \multicolumn{1}{c|}{\multirow{3}{*}{%s}} & Ours & %s & %s & %s & %s \\\\ \cline{2-6}
        \multicolumn{1}{c|}{}& RGB+Aux & %s & %s & %s & %s \\\\ \cline{2-6}
        \multicolumn{1}{c|}{}& Supersample & %s & %s & %s & %s \\\\ %s
        """ % (name, score[0][0], score[0][1], score[0][2], score[0][3],
                     score[1][0], score[1][1], score[1][2], score[1][3],
                     score[2][0], score[2][1], score[2][2], score[2][3],
               '\\hline' if (ind == len(names_and_dirs) - 1) else '\\thickhline')

            str = str + str_add

        open('error-table-data.tex', 'w').write(str)
        
if __name__ == '__main__':
    main()
