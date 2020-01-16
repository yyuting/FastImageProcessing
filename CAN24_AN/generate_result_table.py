import os
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import colorsys
import skimage
import skimage.io

hue_shaders = {
    'bricks': 0.1,
    'mandelbrot': 0.2,
    'mandelbulb': 0.3,
    'marble': 0.4,
    'oceanic': 0.5,
    'primitives': 0.6,
    'trippy': 0.7,
    'venice': 0.8,
    'boids': 0.9
}

app_shader_dir_200 = {
'denoising': {
    'bricks': {'dir': ['1x_1sample_bricks_dfs_no_rotation/test/',
                       '1x_1sample_bricks_dfs_no_rotation_aux/test/',
                       '1x_1sample_bricks_dfs_no_rotation/mean1_test/'],
               'img_idx': 6,
               'img_zoom_bbox': [240, 320, -190, -130],
               'gt_dir': '/n/fs/shaderml/datas_bricks_dsf/test_img',
               'msaa_sample': 1
              },
    'marble': {'dir': ['1x_1sample_marble_automatic_200/test/',
                       '1x_1sample_marble_automatic_200_aux/test/',
                       '1x_1sample_marble_automatic_200/mean1_test/'],
               'img_idx': 25,
               'img_zoom_bbox': [220, 220+80, 3, 3+60],
               'gt_dir': '/n/fs/shaderml/datas_marble_automatic_200/test_img',
               'msaa_sample': 1
              },
    'mandelbrot': {'dir': ['1x_1sample_mandelbrot_tile_automatic_200_repeat/test/',
                           '1x_1sample_mandelbrot_tile_automatic_200_aux_repeat/test/',
                           '1x_1sample_mandelbrot_tile_automatic_200_repeat/mean3_test/'],
                   'img_idx': 8,
                   'img_zoom_bbox': [550, 550+80, 500, 500+60],
                   'gt_dir': '/n/fs/shaderml/datas_mandelbrot_tile_automatic_200/test_img',
                   'msaa_sample': 3
                  },
    'mandelbulb': {'dir': ['1x_1sample_mandelbulb_automatic_200/test/',
                           '1x_1sample_mandelbulb_automatic_200_aux/test/',
                           '1x_1sample_mandelbulb_automatic_200/mean3_test/'],
                   'img_idx': 7,
                   'img_zoom_bbox': [370, 370+80, 370, 370+60],
                   'gt_dir': '/n/fs/shaderml/datas_mandelbulb_automatic_200/test_img',
                   'msaa_sample': 3
                  },
    'primitives': {'dir': ['1x_1sample_primitives_wheel_only_automatic_200/test/',
                           '1x_1sample_primitives_wheel_only_automatic_200_aux/test/',
                           '1x_1sample_primitives_wheel_only_automatic_200/mean1_test/'],
                   'img_idx': 1,
                   'img_zoom_bbox': [270, 270+80, 370, 370+60],
                   'gt_dir': '/n/fs/shaderml/datas_primitives_wheel_only_automatic_200/test_img',
                   'msaa_sample': 1
                  },
    'trippy': {'dir': ['1x_1sample_trippy_heart_tile_automatic_dfs_200/test/',
               '1x_1sample_trippy_heart_tile_automatic_dfs_200_aux/test/',
               '1x_1sample_trippy_heart_tile_automatic_dfs_200/mean4_test/'],
               'img_idx': 11,
               'img_zoom_bbox': [550, 550+80, 65, 65+60],
               'gt_dir': '/n/fs/shaderml/datas_trippy_heart_tile_rotation_automatic_200/test_img',
               'msaa_sample': 4
              },
    'oceanic': {'dir': ['1x_1sample_oceanic_all_raymarching_automatic_200/test/',
                '1x_1sample_oceanic_all_raymarching_automatic_200_aux/test/',
                '1x_1sample_oceanic_all_raymarching_automatic_200/mean1_test/'],
                #'img_idx': 14,
                #'img_zoom_bbox': [280, 280+80, 530, 530+60],
                'img_idx': 17,
                'img_zoom_bbox': [280, 280+80, 750, 750+60],
                'gt_dir': '/n/fs/shaderml/datas_oceanic_all_raymarching_automatic_200/test_img',
                'msaa_sample': 1
               },
    'venice': {'dir': ['1x_1sample_venice_full_automatic_200/test/',
               '1x_1sample_venice_full_proxy_automatic_400_with_initial_aux/test/',
               '1x_1sample_venice_full_automatic_200/mean1_test/'],
               'img_idx': 3,
               'img_zoom_bbox': [150, 150+80, 150, 150+60],
               'gt_dir': '/n/fs/shaderml/datas_venice_100spp_focus_far_simplified_automatic_400/test_img',
               'msaa_sample': 1
              }
    },
'simplified': {
    'bricks': {'dir': ['1x_1sample_bricks_simplified_proxy/test/',
               '1x_1sample_bricks_simplified_proxy_aux/test/',
               '1x_1sample_bricks_simplified_proxy/mean1_test'],
               'img_idx': 6,
               'gt_dir': '/n/fs/shaderml/datas_bricks_dsf/test_img'
              },
    'mandelbrot': {'dir': ['1x_1sample_mandelbrot_simplified_proxy_all_trace_patchGAN_scale_005_update_8_long/test_zero_samples/test/',
                   '1x_1sample_mandelbrot_simplified_proxy_all_trace_patchGAN_scale_005_update_8_long_aux/test/',
                   '1x_1sample_mandelbrot_simplified_proxy_all_trace_patchGAN_scale_005_update_8_long/mean1_test/'],
                   'img_idx': 8,
                   'gt_dir': '/n/fs/shaderml/datas_mandelbrot_simplified_proxy/test_img'
                  },
    'mandelbulb': {'dir': ['1x_1sample_mandelbulb_simplified_automatic_200/test/',
                   '1x_1sample_mandelbulb_simplified_automatic_200_aux/test/',
                   '1x_1sample_mandelbulb_simplified_automatic_200/mean1_test/'],
                   'other_view': ['mandelbulb_video_gt_f1.png',
                                  '1x_1sample_mandelbulb_simplified_automatic_200/render_ours/000001.png',
                                  '1x_1sample_mandelbulb_simplified_automatic_200_aux/render/000001.png',
                                  '1x_1sample_mandelbulb_simplified_automatic_200/render_mean1/000001.png']
                  },
    'trippy': {'dir': ['1x_1sample_trippy_heart_simplified_proxy_dfs_automatic_200/test/',
               '1x_1sample_trippy_heart_simplified_proxy_dfs_automatic_200_aux/test/',
               '1x_1sample_trippy_heart_simplified_proxy_dfs_automatic_200/mean1_test'],
               'img_idx': 11,
               'gt_dir': '/n/fs/shaderml/datas_trippy_heart_simplified_proxy_dfs_automatic_200/test_img'
              },
    'venice': {'dir': ['1x_1sample_venice_100spp_simplified_30_70_automatic_200/test/',
                       '1x_1sample_venice_100spp_simplified_30_70_aux/test/',
                       '1x_1sample_venice_100spp_simplified_30_70_automatic_200/mean1_test/'],
               'img_idx': 25,
               'gt_dir': '/n/fs/shaderml/datas_venice_100spp_simplified_30_70_automatic_200/test_img'
              }
    },
'temporal': {
    'mandelbrot_simplified': {'dir': ['1x_1sample_mandelbrot_simplified_proxy_temporal/test',
                              '1x_1sample_mandelbrot_simplified_proxy_temporal_aux/test'],
                              'other_view': ['mandelbrot_simplified_temporal_gt_gamma_corrected.png',
                                             '1x_1sample_mandelbrot_simplified_proxy_temporal/test_gamma_corrected/00000827.png',
                                             '1x_1sample_mandelbrot_simplified_proxy_temporal_aux/test_gamma_corrected/00000827.png',
                                             'mandelbrot_simplified_temporal_input_gamma_corrected.png']
                             },
    'mandelbrot': {'dir': ['1x_1sample_mandelbrot_full_temporal_automatic_200_correct_alpha/test',
                   '1x_1sample_mandelbrot_full_temporal_correct_alpha_aux/test'],
                   'other_view': ['mandelbrot_full_temporal_gt_gamma_corrected.png',
                                  '1x_1sample_mandelbrot_full_temporal_automatic_200_correct_alpha/test_gamma_corrected/00000727.png',
                                  '1x_1sample_mandelbrot_full_temporal_correct_alpha_aux/test_gamma_corrected/00000727.png',
                                  'mandelbrot_full_temporal_input_gamma_corrected.png']
                  },
    'mandelbulb_simplified': {'dir': ['1x_1sample_mandelbulb_simplified_temporal_corrected_alpha/test',
                              '1x_1sample_mandelbulb_simplified_temporal_corrected_alpha_aux/test'],
                              'other_view': ['mandelbulb_simplified_temporal_gt.png',
                                             '1x_1sample_mandelbulb_simplified_temporal_corrected/render/000030.png',
                                             '1x_1sample_mandelbulb_simplified_temporal_corrected_aux/render/000030.png',
                                             'mandelbulb_simplified_temporal_input.png']
                             },
    'mandelbulb': {'dir': ['1x_1sample_mandelbulb_full_temporal_larger_cap_correct_alpha/test',
                   '1x_1sample_mandelbulb_full_temporal_larger_cap_correct_alpha_aux/test'],
                   'other_view': ['/n/fs/shaderml/datas_mandelbulb_full_temporal/test_img/test_ground2900007.png',
                                  '1x_1sample_mandelbulb_full_temporal_larger_cap_correct_alpha/test/00000827.png',
                                  '1x_1sample_mandelbulb_full_temporal_larger_cap_correct_alpha_aux/test/00000827.png',
                                  'mandelbulb_full_temporal_input.png']
                  },
    'trippy_simplified': {'dir': ['1x_1sample_trippy_simplified_temporal_corrected_alpha/test',
                                  '1x_1sample_trippy_simplified_temporal_corrected_alpha_aux/test',
                                  '1x_1sample_trippy_heart_simplified_proxy_dfs_automatic_200/mean1_test'],
                          'other_view': ['/n/fs/shaderml/datas_trippy_temporal/test_img/test_ground2900010.png',
                                         '1x_1sample_trippy_simplified_temporal_corrected_alpha/test/00001127.png',
                                         '1x_1sample_trippy_simplified_temporal_corrected_alpha_aux/test/00001127.png',
                                         'trippy_simplified_temporal_input.png']
                         }
    },
'post_processing': {
    'mandelbulb_blur': {'dir': ['1x_1sample_mandelbulb_defocus_blur_automatic_200/test',
                                '1x_1sample_mandelbulb_defocus_blur_automatic_200_aux/test']},
    'trippy_sharpen': {'dir': ['1x_1sample_trippy_heart_local_laplacian_random_rotation_automatic_200/test',
                          '1x_1sample_trippy_heart_local_laplacian_random_rotation_automatic_200_aux/test']},
    'trippy_simplified_sharpen': {'dir': ['1x_1sample_trippy_simplified_local_laplacian_rotation_dataset_automatic_200_update_8/test',
                                     '1x_1sample_trippy_simplified_local_laplacian_rotation_dataset_automatic_200_update_8_aux/test']}
    },
'simulation': {
    'boids': {'dir': ['/n/fs/visualai-scr/yutingy/boids_res_20_64_validate_switch_label/test',
              '/n/fs/visualai-scr/yutingy/boids_res_20_64_validate_switch_label_aux/test']}
    }
}

app_names = ['denoising',
             'simplified',
             'temporal',
             'post_processing',
             'simulation']

max_shader_per_fig = 5

def main():
    
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
                    bar_x_ticks.append(name)
                    break

            current_col = colorsys.hsv_to_rgb(current_hue, 1, 1)
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
                neval = 2

            score = -np.ones((neval, 3))
            l2_score = -np.ones((neval, 3))

            for i in range(neval):

                dir = app_data[shader_name]['dir'][i]

                if app_name != 'temporal':
                    perceptual_breakdown_file = os.path.join(dir, 'perceptual_tf_breakdown.txt')
                else:
                    perceptual_breakdown_file = os.path.join(dir, 'vgg_breakdown.txt') 
                
                perceptual_single_file = os.path.join(dir, 'vgg.txt')
                l2_single_file = os.path.join(dir, 'all_loss.txt')

                if app_name in ['denoising', 'simplified', 'post_processing', 'temporal']:
                    perceptual_scores = open(perceptual_breakdown_file).read()
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
                    l2_breakdown_file = os.path.join(dir, 'score_breakdown.txt')
                    l2_scores = open(l2_breakdown_file).read()
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

    bar_x = np.arange(len(bar_x_ticks))
    fig = plt.figure()
    fig.set_size_inches(9, 4)

    #ax = plt.subplot(111)
    #ax.set_aspect(1.0)
    plt.bar(bar_x[full_shader_idx], [bar_avg[i] for i in full_shader_idx], color=[bar_col[i] for i in full_shader_idx], edgecolor='k', linestyle='-')
    plt.bar(bar_x[simplified_shader_idx], [bar_avg[i] for i in simplified_shader_idx], color=[bar_col[i] for i in simplified_shader_idx], edgecolor='k', linestyle='--')
    plt.xticks(bar_x, bar_x_ticks, rotation=90)

    plt.text(2.5, 1.05, 'denoising')
    plt.text(10, 1.05, 'simplified')
    plt.text(16, 1.05, 'temporal')
    plt.text(21.5, 1.05, 'post')
    plt.text(24.6, 1.05, 'sim')


    #plt.text(0, 0.9, 'baseline', withdash=True, color='b')

    for k in range(len(app_names)):
        plt.axvspan(slice_start[k], slice_end[k], facecolor=colorsys.hsv_to_rgb(0.0, k / len(app_names), k / len(app_names)), alpha=0.3)

    plt.xlim(slice_start[0], slice_end[-1])

    plt.plot(np.arange(-1, len(bar_x_ticks)+1), np.ones(len(bar_x_ticks)+2), 'k-', label='full shader')
    plt.plot(np.arange(-1, len(bar_x_ticks)+1), np.ones(len(bar_x_ticks)+2), 'k--', label='simplified shader')

    plt.plot(np.arange(-1, len(bar_x_ticks)+1), np.ones(len(bar_x_ticks)+2), label='baseline')

    plt.legend(loc=0)

    plt.ylim(0, 1.15)
    plt.ylabel('relative error')

    plt.tight_layout()
    plt.savefig('result_figs/bar_summary.png')
    plt.close(fig)
    
    
    # table
    
    str = ""
    
    for k in range(len(app_names)):
        
        app_name = app_names[k]
        
        if app_name in ['simulation']:
            continue

        str += """
\\vspace{-1ex}
\setlength{\\tabcolsep}{2.0pt}
\\begin{table}[]
\\begin{tabular}{c|ccccc}
\\hline

    \multirow{2}{*}{Shader} &  & \multicolumn{2}{c}{Perceptual} & \multicolumn{2}{c}{L2 Error} \\\\ \cline{3-6} 
    & Distances: & Similar & Different & Similar & Different \\\\ \\thickhline
"""

        for shader_name in sorted(app_shader_dir_200[app_name].keys()):
            data = app_shader_dir_200[app_name][shader_name]

            
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
    \multicolumn{1}{c|}{\multirow{3}{*}{\\begin{tabular}[c]{@{}c@{}}\\%s\\\\ Fig1 \end{tabular}}} & RGBx & %s & %s & %s & %s \\\\ \cline{2-6} 
    \multicolumn{1}{c|}{} & Ours & %s\%% & %s\%% & %s\%% & %s\%% \\\\ \cline{2-6}
    \multicolumn{1}{c|}{} & MSAA & %s\%% & %s\%% & %s\%% & %s\%% \\\\ \\thickhline""" % ((print_name, ) + tuple(data_strs))
            else:
                str += """
\multicolumn{1}{c|}{\multirow{2}{*}{\\begin{tabular}[c]{@{}c@{}}\\%s \end{tabular}}} & RGBx & %s & %s & %s & %s \\\\ \cline{2-6} 
    \multicolumn{1}{c|}{} & Ours & %s\%% & %s\%% & %s\%% & %s\%% \\\\ \\thickhline""" % ((print_name, ) + tuple(data_strs))

        str = str[:-11] + '\hline' + """
\end{tabular}
\caption{%s}
\end{table}
""" % app_name.replace('_', '\\ ')
        
    open('result_figs/supplemental.tex', 'w').write(str)
    
    
    # result image for all apps
    crop_edge_col = [12, 144, 36]
    bbox_edge_w = 3
    
    str = ''
    
    no_zoom_defined = False
    
    for k in range(len(app_names)):
        
        app_name = app_names[k]
        fig_start = True
        fig_rows = 0
        
        if app_name == 'denoising':
            
            str += """
\\newcommand{\ResultsFigStartWithZoom}{
\setlength{\\tabcolsep}{1pt}
\setlength{\h}{1.15in}
\\begin{tabular}{cccccc}
}

\\newcommand{\ResultsFigEndWithZoom}[6]{
\includegraphics[height=\h]{result_figs/#2_gt_box} & \includegraphics[height=\h]{result_figs/#2_ours_box} & \includegraphics[height=\h]{result_figs/#2_gt_zoom} & \includegraphics[height=\h]{result_figs/#2_ours_zoom} & \includegraphics[height=\h]{result_figs/#2_RGBx_zoom} & \includegraphics[height=\h]{result_figs/#2_MSAA_zoom} \\tablegap
{\small {#1}} & & 
{\small {#3}} & 
{\small {#4}} & 
{\small {#5}} & 
{\small {#6}} 
\end{tabular}
}

\\newcommand{\ResultsFigWithHeaderWithZoom}[6]{
\ResultsFigStartWithZoom
(a) Reference & (b) Our result & (c) Reference & (d) Ours & (e) RGBx & (f) Supersample\\\\
\\addlinespace[4pt]
\ResultsFigEndWithZoom{#1}{#2}{#3}{#4}{#5}{#6}
}

\\newcommand{\ResultsFigNoHeaderWithZoom}[6]{
\ResultsFigStartWithZoom
\\addlinespace[4pt]
\ResultsFigEndWithZoom{#1}{#2}{#3}{#4}{#5}{#6}
}

"""
            
        elif not no_zoom_defined:
            no_zoom_defined = True
            str += """
\\newcommand{\ResultsFigStartWithoutZoom}{
\setlength{\\tabcolsep}{1pt}
\setlength{\h}{1.15in}
\\begin{tabular}{cccc}
}

\\newcommand{\ResultsFigEndWithoutZoom}[4]{
\includegraphics[height=\h]{result_figs/#2_gt} & \includegraphics[height=\h]{result_figs/#2_ours} & \includegraphics[height=\h]{result_figs/#2_RGBx} & \includegraphics[height=\h]{result_figs/#2_input} \\tablegap
{\small {#1}} &
{\small {#3}} & 
{\small {#4}} & 
\end{tabular}
}

\\newcommand{\ResultsFigWithHeaderWithoutZoom}[4]{
\ResultsFigStartWithoutZoom
(a) Reference & (b) Our result & (c) RGBx & (d) Input to Network \\\\
\\addlinespace[4pt]
\ResultsFigEndWithoutZoom{#1}{#2}{#3}{#4}
}

\\newcommand{\ResultsFigNoHeaderWithoutZoom}[4]{
\ResultsFigStartWithoutZoom
\\addlinespace[4pt]
\ResultsFigEndWithoutZoom{#1}{#2}{#3}{#4}
}

"""
        
        app_data = app_shader_dir_200[app_name]
        for shader_name in sorted(app_data.keys()):
            data = app_data[shader_name]

            
            
            if 'img_idx' in data.keys() or 'other_view' in data.keys():
                
                if 'img_idx' in data.keys():
                    if shader_name == 'mandelbrot':
                        for i in range(len(data['dir'])):
                            if data['dir'][i].endswith('/'):
                                data['dir'][i] = data['dir'][i][:-1]
                            data['dir'][i] = data['dir'][i] + '_gamma_corrected'
                        if data['gt_dir'].endswith('/'):
                            data['gt_dir'] = data['gt_dir']
                        data['gt_dir'] = data['gt_dir'] + '_gamma_corrected'

                    orig_imgs = []
                    for i in range(len(data['dir'])):
                        if app_name == 'temporal':
                            orig_imgs.append(skimage.io.imread(os.path.join(data['dir'][i], '%06d27.png' % data['img_idx'])))
                        else:
                            orig_imgs.append(skimage.io.imread(os.path.join(data['dir'][i], '%06d.png' % data['img_idx'])))
                    if app_name == 'temporal':
                        gt_img = skimage.io.imread(os.path.join(data['gt_dir'], '29%05d.png' % (data['img_idx']-1)))
                    else:
                        gt_files = sorted(os.listdir(data['gt_dir']))
                        gt_img = skimage.io.imread(os.path.join(data['gt_dir'], gt_files[data['img_idx']-1]))

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

                        skimage.io.imsave(os.path.join('result_figs', '%s_%s_%s_box.png' % (app_name, shader_name, prefix)), img)
                    else:
                        skimage.io.imsave(os.path.join('result_figs', '%s_%s_%s.png' % (app_name, shader_name, prefix)), img)
                    
                if app_name == 'denoising':
                    macro_suffix = 'WithZoom'
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
                if 'simplified' in shader_name:
                    for name in hue_shaders.keys():
                        if shader_name.startswith(name):
                            print_name = name + '\\ simplified'
                            break
                
                if app_name == 'denoising':
                    str += """
%s{\%s}{%s_%s}{}{%.1e}{%.1e}{%d\,SPP, %.1e}
""" % (macro, print_name, app_name, shader_name, data['perceptual'][0, 2], data['perceptual'][1, 2], data['msaa_sample'], data['perceptual'][2, 2])
                else:
                    str += """
%s{\%s}{%s_%s}{%.1e}{%.1e}
""" % (macro, print_name, app_name, shader_name, data['perceptual'][0, 2], data['perceptual'][1, 2])
        
        if not fig_start:
            str += """
\\vspace{-2ex}
\caption{%s}
\end{figure*}
""" % app_name
        
    open('result_figs/result_pool.tex', 'w').write(str)

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
                    l2_breakdown_file = os.path.join(dir, 'score_breakdown.txt')
                    perceptual_breakdown_file = os.path.join(dir, 'perceptual_tf_breakdown.txt')

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
