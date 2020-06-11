import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import numpy as np
import colorsys

matplotlib.rcParams['xtick.minor.size'] = 0
matplotlib.rcParams['xtick.minor.width'] = 0
matplotlib.rcParams['ytick.minor.size'] = 0
matplotlib.rcParams['ytick.minor.width'] = 0

plot_rel_trace = True

markeredgewidth = 6

shader_names = ['mandelbulb', 'mandelbrot', 'primitives', 'trippy']

slicing_datas = {
    'mandelbrot': {
        'baseline': [
            '1x_1sample_mandelbrot_with_bg_aux_largest_capacity',
            '1x_1sample_mandelbrot_with_bg_all'
        ],
        'good_oracle': [
            '1x_1sample_mandelbrot_with_bg_highest_score_05',
            '1x_1sample_mandelbrot_with_bg_highest_score_025',
            '1x_1sample_mandelbrot_with_bg_highest_score_0125',
            '1x_1sample_mandelbrot_with_bg_highest_score_00625',
            '1x_1sample_mandelbrot_with_bg_highest_score_003125'
        ],
        'bad_oracle': [
            '1x_1sample_mandelbrot_with_bg_lowest_score_05',
            '1x_1sample_mandelbrot_with_bg_lowest_score_025',
            '1x_1sample_mandelbrot_with_bg_lowest_score_0125',
            '1x_1sample_mandelbrot_with_bg_lowest_score_00625',
            '1x_1sample_mandelbrot_with_bg_lowest_score_003125'
        ],
        'subsample': [
            '1x_1sample_mandelbrot_with_bg_stratified_subsample_2',
            '1x_1sample_mandelbrot_with_bg_stratified_subsample_4',
            '1x_1sample_mandelbrot_with_bg_stratified_subsample_8',
            '1x_1sample_mandelbrot_with_bg_stratified_subsample_16',
            '1x_1sample_mandelbrot_with_bg_stratified_subsample_32'
        ]
    },
    'trippy': {
        'baseline': [
            '1x_1sample_trippy_aux_largest_capacity',
            '1x_1sample_trippy_subsample_2'
        ],
        'good_oracle': [
            '1x_1sample_trippy_subsample_2_highest_score_05_new',
            '1x_1sample_trippy_subsample_2_highest_score_025_new',
            '1x_1sample_trippy_subsample_2_highest_score_0125_new',
            '1x_1sample_trippy_subsample_2_highest_score_00625_new',
            '1x_1sample_trippy_subsample_2_highest_score_003125_new',
            '1x_1sample_trippy_subsample_2_highest_score_0015625_new'
        ],
        'bad_oracle': [
            '1x_1sample_trippy_subsample_2_lowest_score_05_new',
            '1x_1sample_trippy_subsample_2_lowest_score_025_new',
            '1x_1sample_trippy_subsample_2_lowest_score_0125_new',
            '1x_1sample_trippy_subsample_2_lowest_score_00625_new',
            '1x_1sample_trippy_subsample_2_lowest_score_003125_new',
            '1x_1sample_trippy_subsample_2_lowest_score_0015625_new'
        ],
        'subsample': [
            '1x_1sample_trippy_stratified_subsample_4',
            '1x_1sample_trippy_stratified_subsample_8',
            '1x_1sample_trippy_stratified_subsample_16',
            '1x_1sample_trippy_stratified_subsample_32',
            '1x_1sample_trippy_stratified_subsample_64',
            '1x_1sample_trippy_stratified_subsample_128'
        ]
    },
    'mandelbulb': {
        'baseline': [
            '1x_1sample_mandelbulb_with_bg_aux_largest_capacity',
            '1x_1sample_mandelbulb_with_bg_all'
        ],
        'good_oracle': [
            '1x_1sample_mandelbulb_with_bg_highest_score_05_correct_trace_len',
            '1x_1sample_mandelbulb_with_bg_highest_score_025',
            '1x_1sample_mandelbulb_with_bg_highest_score_0125',
            '1x_1sample_mandelbulb_with_bg_highest_score_00625'
        ],
        'bad_oracle': [
            '1x_1sample_mandelbulb_with_bg_lowest_score_05_correct_trace_len',
            '1x_1sample_mandelbulb_with_bg_lowest_score_025',
            '1x_1sample_mandelbulb_with_bg_lowest_score_0125',
            '1x_1sample_mandelbulb_with_bg_lowest_score_00625'
        ],
        'subsample': [
            '1x_1sample_mandelbulb_with_bg_stratified_subsample_2',
            '1x_1sample_mandelbulb_with_bg_stratified_subsample_4',
            '1x_1sample_mandelbulb_with_bg_stratified_subsample_8',
            '1x_1sample_mandelbulb_with_bg_stratified_subsample_16'
        ]
    },
    'primitives': {
        'baseline': [
            '1x_1sample_primitives_aux_largest_capacity',
            '1x_1sample_primitives_all'
        ],
        'good_oracle': [
            '1x_1sample_primitives_highest_score_05_correct_trace_len',
            '1x_1sample_primitives_highest_score_025_new',
            '1x_1sample_primitives_highest_score_0125_new',
            '1x_1sample_primitives_highest_score_00625_new',
        ],
        'bad_oracle': [
            '1x_1sample_primitives_lowest_score_05_correct_trace_len',
            '1x_1sample_primitives_lowest_score_025_new',
            '1x_1sample_primitives_lowest_score_0125_new',
            '1x_1sample_primitives_lowest_score_00625_new',
        ],
        'subsample': [
            '1x_1sample_primitives_subsample_2',
            '1x_1sample_primitives_stratified_subsample_4',
            '1x_1sample_primitives_stratified_subsample_8',
            '1x_1sample_primitives_stratified_subsample_16',
        ]
    }
}

def get_info(dirname, is_baseline=False, base_dir=None, get_runtime=False):
    
    full_dirname = os.path.join('/mnt/shadermlnfs1/shadermlvm/playground/models', dirname)
    inference_time_dirname = os.path.join('/mnt/shadermlnfs1/shadermlvm/playground/models/out_inference_time', dirname)
    
    
    if is_baseline:
        orig_base_dir = base_dir
        assert base_dir is not None
        base_dir = os.path.join('/mnt/shadermlnfs1/shadermlvm/playground/models', base_dir)
        col_aux_inds = np.load(os.path.join(base_dir, 'col_aux_inds.npy'))
        trace_len = col_aux_inds.shape[0]
    else:
        specified_ind = np.load(os.path.join(full_dirname, 'specified_ind.npy'))
        trace_len = specified_ind.shape[0]
        
    perceptual_file = os.path.join(full_dirname, 'test/perceptual_tf.txt')
    perceptual_score = float(open(perceptual_file).read())
    
    l2_file = os.path.join(full_dirname, 'test/score.txt')
    l2_score = float(open(l2_file).read())

    ans = {
        'trace': trace_len, 
        'perceptual': perceptual_score,
        'l2': l2_score,
    }
    
    if get_runtime:
        time_stat_file = os.path.join(inference_time_dirname, 'test/time_stats.txt')
        time_vals = open(time_stat_file).read().split(',')
        median_runtime = float(time_vals[0])
        ans['runtime'] = median_runtime
    
    
    if is_baseline:
        
        time_stat_file = os.path.join('/mnt/shadermlnfs1/shadermlvm/playground/models/out_inference_time', orig_base_dir, 'test/time_stats.txt')
        time_vals = open(time_stat_file).read().split(',')
        median_runtime = float(time_vals[0])
        
        trace_contrib = np.load(os.path.join(base_dir, 'train/taylor_exp_vals.npy'))
        all_trace_len = trace_contrib.shape[0]
        ans['all_trace'] = all_trace_len
        
        perceptual_file = os.path.join(base_dir, 'test/perceptual_tf.txt')
        perceptual_score = float(open(perceptual_file).read())

        l2_file = os.path.join(base_dir, 'test/score.txt')
        l2_score = float(open(l2_file).read())
        
        ans['all_perceptual'] = perceptual_score
        ans['all_l2'] = l2_score
        ans['all_runtime'] = median_runtime
    
    return ans

def main():
    
    fontsize = 14
    
    # 0: good, subsample, bad
    # 1: good, bad, subsample
    # 2: subsample, good, bad
    # 3: subsample, bad, good
    # 4: bad, good, subsample
    # 5: bad, subsample, good
    relationships = []
    for i in range(6):
        if i == 0:
            order = ['good', 'subsample', 'bad']
        elif i == 1:
            order = ['good', 'bad', 'subsample']
        elif i == 2:
            order = ['subsample', 'good', 'bad']
        elif i == 3:
            order = ['subsample', 'bad', 'good']
        elif i == 4:
            order = ['bad', 'good', 'subsample']
        elif i == 5:
            order = ['bad', 'subsample', 'good']
            
        relationships.append({'order': order, 'l2': {}, 'perceptual': {}})
        
    total_count = {'l2': 0, 'perceptual': 0}
    
    colors = {'subsample': colorsys.hsv_to_rgb(0.83, 1, 0.7),
              'good_oracle': colorsys.hsv_to_rgb(0.67, 1, 0.7),
              'bad_oracle': colorsys.hsv_to_rgb(0.99, 1, 0.7),
              'RGBx': colorsys.hsv_to_rgb(0.4, 1, 0.7), 
              'runtime': colorsys.hsv_to_rgb(0.1, 1, 0),
              'all': colorsys.hsv_to_rgb(0.1, 1, 0.7)}
    
    
    assert len(slicing_datas.keys()) == 4
    shader_count = 1
    
    fig_total = plt.figure()
    fig_total.set_size_inches(7, 7)
        
    for shader_name in shader_names:
        shader_datas = slicing_datas[shader_name]
        
        plt.figure(fig_total.number)
        ax1 = plt.subplot(2, 2, shader_count)
        shader_count += 1
        
        all_plot_datas = {}
        
        baseline_info = None
        
        for category in sorted(shader_datas.keys()):
            
            if category == 'baseline':
                assert len(shader_datas[category]) == 2
                baseline_info = get_info(shader_datas[category][0], True, shader_datas[category][1], get_runtime=True)
                continue
            
            datapoints = shader_datas[category]

            perceptual = np.empty(len(datapoints))
            xvals = np.empty(len(datapoints))
            l2 = np.empty(len(datapoints))
            
            if category == 'subsample':
                runtime = np.empty(len(datapoints))
            
            for i in range(len(datapoints)):
                dirname = datapoints[i]
                data_info = get_info(dirname, get_runtime='subsample' in category)
                perceptual[i] = data_info['perceptual']
                xvals[i] = data_info['trace']
                l2[i] = data_info['l2']
                if category == 'subsample':
                    runtime[i] = data_info['runtime']
                                
            all_plot_datas[category] = {'perceptual': perceptual, 'trace': xvals, 'l2': l2}
            
            if category == 'subsample':
                all_plot_datas[category]['runtime'] = runtime
            
        assert baseline_info is not None
        
        for metric in ['perceptual', 'l2']:
        
            fig = plt.figure()
            fig.set_size_inches(7, 3.5)
            
            comparisons = {}

            for category in all_plot_datas.keys():

                current_perceptual = all_plot_datas[category][metric]
                
                current_trace = all_plot_datas[category]['trace']
                
                if plot_rel_trace:
                    current_trace = 2. ** (-np.arange(current_trace.shape[0]) - 1)

                plt.plot(current_trace, current_perceptual, 'o', linewidth=10, markeredgewidth=markeredgewidth, label=category, color=colors[category])
                
                if metric == 'perceptual':
                    plt.figure(fig_total.number)
                    plt.plot(current_trace, current_perceptual / baseline_info[metric], 'o', linewidth=10, markeredgewidth=markeredgewidth, label=category, color=colors[category])
                
                    plt.figure(fig.number)

                current_perceptual = np.concatenate((current_perceptual, np.array([baseline_info[metric]])))
                if plot_rel_trace:
                    current_trace = np.concatenate((current_trace, [2. ** (-current_trace.shape[0] - 1)]))
                else:
                    current_trace = np.concatenate((current_trace, np.array([baseline_info['trace']])))

                plt.plot(current_trace, current_perceptual, linewidth=2.25, color=colors[category])
                
                if metric == 'perceptual':
                    
                    current_perceptual = np.concatenate((np.array([baseline_info['all_perceptual']]), current_perceptual))
                    if plot_rel_trace:
                        current_trace = np.concatenate(([1], current_trace))
                    else:
                        current_trace = np.concatenate((np.array([baseline_info['all_trace']]), current_trace))
                    
                    plt.figure(fig_total.number)
                    plt.plot(current_trace, current_perceptual / baseline_info[metric], linewidth=2.25, color=colors[category])
                    plt.figure(fig.number)
                
                
                if 'good' in category:
                    comparisons['good'] = all_plot_datas[category][metric]
                elif 'bad' in category:
                    comparisons['bad'] = all_plot_datas[category][metric]
                elif 'subsample' in category:
                    comparisons['subsample'] = all_plot_datas[category][metric]

            plt.plot([baseline_info['trace']], [baseline_info[metric]], 's', linewidth=10, markeredgewidth=markeredgewidth, label='RGBx', color=colors['RGBx'])
            
            if metric == 'perceptual':
                plt.figure(fig_total.number)
                plt.plot([current_trace[-1]], [1.0], 's', linewidth=10, markeredgewidth=markeredgewidth, label='RGBx', color=colors['RGBx'])
                plt.title(shader_name)
                
                all_runtime = np.concatenate(([baseline_info['all_runtime']], all_plot_datas['subsample']['runtime'], [baseline_info['runtime']]))
                
                all_runtime /= all_runtime[0]
                
                plt.plot(current_trace, all_runtime, '^', markeredgewidth=markeredgewidth, color=colors['runtime'], zorder=1, label='runtime')
                plt.plot(current_trace, all_runtime, linewidth=2.5, color=colors['runtime'], zorder=1)
                
                #ax2 = ax1.twinx()
                #ax2.plot(current_trace, 10 + np.arange(current_trace.shape[0])[::-1], '^', markeredgewidth=markeredgewidth, color=colors['runtime'], zorder=1, label='runtime')
                #ax2.plot(current_trace, 10 + np.arange(current_trace.shape[0])[::-1], linewidth=2.5, color=colors['runtime'], zorder=1)
                #ax2.set_ylim(0, 20)
                
                ax1.plot([current_trace[0]], [baseline_info['all_perceptual'] / baseline_info['perceptual']], 'p', linewidth=10, markeredgewidth=markeredgewidth, label='all trace', color=colors['all'])
                
                #ax1.set_zorder(ax2.get_zorder()+1)
                #ax1.patch.set_visible(False)
                
                ax1.set_ylim(0.1, 1.1)
                
                if shader_count % 2 == 1:
                    plt.yticks([], [])
                    
                plt.xscale('log')
                
                if plot_rel_trace:
                    x_tick_labels = ['1/%d' % (d + 1) for d in range(current_trace.shape[0])]
                    x_tick_labels[0] = '1'
                    x_tick_labels[-1] = '0'
                    plt.xticks(current_trace, x_tick_labels)
                
                plt.figure(fig.number)

            plt.xlabel('Trace Length used in Model', fontsize=fontsize)
            plt.ylabel('%s error' % metric, fontsize=fontsize)
            plt.legend()

            plt.tight_layout()

            plt.savefig(os.path.join('result_figs', '%s_trace_contribution_%s.png' % (shader_name, metric)))
            plt.close(fig)
            
            assert comparisons['good'].shape == comparisons['bad'].shape and comparisons['good'].shape == comparisons['subsample'].shape
            
            for i in range(len(relationships)):
                order = relationships[i]['order']
                
                if 'accum' not in relationships[i][metric].keys():
                    relationships[i][metric]['accum'] = 0
                    
                count = np.sum((comparisons[order[0]] <= comparisons[order[1]]) * (comparisons[order[1]] <= comparisons[order[2]]))
                
                relationships[i][metric]['accum'] = relationships[i][metric]['accum'] + count
                relationships[i][metric][shader_name] = count
                
            total_count[metric] = total_count[metric] + comparisons['good'].shape[0]
                        
        print(shader_name)
        
    plt.figure(fig_total.number)
    plt.tight_layout()

    plt.savefig(os.path.join('result_figs', 'test.png'))
    plt.close(fig_total)
       
    assert total_count['l2'] == total_count['perceptual']
    
    summary_str = """
Total data points: %d
""" % (total_count['l2'])
    
    for metric in ['perceptual', 'l2']:
        summary_str += """
Statistic in %s error:
""" % metric
        for i in range(len(relationships)):
            summary_str += """
Datapoints with relationship %s < %s < %s: %d (%d%%)
""" % (relationships[i]['order'][0], relationships[i]['order'][1], relationships[i]['order'][2], relationships[i][metric]['accum'], int(relationships[i][metric]['accum'] / total_count[metric] * 100))
            
            for shader_name in sorted(relationships[i][metric].keys()):
                summary_str += '    '
                if shader_name != 'accum':
                    summary_str += '%s: %d' % (shader_name, relationships[i][metric][shader_name])
            summary_str += '\n'
    
    open('result_figs/summary.txt', 'w').write(summary_str)
    
        
if __name__ == '__main__':
    main()