import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import numpy as np
import colorsys

slicing_datas = {
    'mandelbrot': {
        'baseline': [
            '1x_1sample_mandelbrot_aux_largest_capacity',
            '1x_1sample_mandelbrot_all'
        ],
        'good_oracle': [
            '1x_1sample_mandelbrot_highest_score_05_new',
            '1x_1sample_mandelbrot_highest_score_025_new',
            '1x_1sample_mandelbrot_highest_score_0125_new',
            '1x_1sample_mandelbrot_highest_score_00625_new',
            '1x_1sample_mandelbrot_highest_score_003125_new'
        ],
        'bad_oracle': [
            '1x_1sample_mandelbrot_lowest_score_05_new',
            '1x_1sample_mandelbrot_lowest_score_025_new',
            '1x_1sample_mandelbrot_lowest_score_0125_new',
            '1x_1sample_mandelbrot_lowest_score_00625_new',
            '1x_1sample_mandelbrot_lowest_score_003125_new'
        ],
        'subsample': [
            '1x_1sample_mandelbrot_stratified_subsample_2',
            '1x_1sample_mandelbrot_stratified_subsample_4',
            '1x_1sample_mandelbrot_stratified_subsample_8',
            '1x_1sample_mandelbrot_stratified_subsample_16',
            '1x_1sample_mandelbrot_stratified_subsample_32'
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
            '1x_1sample_mandelbulb_aux_largest_capacity',
            '1x_1sample_mandelbulb_all'
        ],
        'good_oracle': [
            '1x_1sample_mandelbulb_highest_score_05_new',
            '1x_1sample_mandelbulb_highest_score_025_new',
            '1x_1sample_mandelbulb_highest_score_0125_new',
            '1x_1sample_mandelbulb_highest_score_00625_new'
        ],
        'bad_oracle': [
            '1x_1sample_mandelbulb_lowest_score_05_new',
            '1x_1sample_mandelbulb_lowest_score_025_new',
            '1x_1sample_mandelbulb_lowest_score_0125_new',
            '1x_1sample_mandelbulb_lowest_score_00625_new'
        ],
        'subsample': [
            '1x_1sample_mandelbulb_stratified_subsample_2',
            '1x_1sample_mandelbulb_stratified_subsample_4',
            '1x_1sample_mandelbulb_stratified_subsample_8',
            '1x_1sample_mandelbulb_stratified_subsample_16'
        ]
    },
    'primitives': {
        'baseline': [
            '1x_1sample_primitives_aux_largest_capacity',
            '1x_1sample_primitives_all'
        ],
        'good_oracle': [
            '1x_1sample_primitives_highest_score_05_new',
            '1x_1sample_primitives_highest_score_025_new',
            '1x_1sample_primitives_highest_score_0125_new',
            '1x_1sample_primitives_highest_score_00625_new',
            '1x_1sample_primitives_highest_score_003125_new'
        ],
        'bad_oracle': [
            '1x_1sample_primitives_lowest_score_05_new',
            '1x_1sample_primitives_lowest_score_025_new',
            '1x_1sample_primitives_lowest_score_0125_new',
            '1x_1sample_primitives_lowest_score_00625_new',
            '1x_1sample_primitives_lowest_score_003125_new'
        ],
        'subsample': [
            '1x_1sample_primitives_subsample_2',
            '1x_1sample_primitives_stratified_subsample_4',
            '1x_1sample_primitives_stratified_subsample_8',
            '1x_1sample_primitives_stratified_subsample_16',
            '1x_1sample_primitives_stratified_subsample_32'
        ]
    }
}

def get_info(dirname, is_rgbx=False, base_dir=None):
    full_dirname = os.path.join('/mnt/shadermlnfs1/shadermlvm/playground/models', dirname)
    
    if is_rgbx:
        assert base_dir is not None
        col_aux_inds = np.load(os.path.join('/mnt/shadermlnfs1/shadermlvm/playground/models', base_dir, 'col_aux_inds.npy'))
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
        'l2': l2_score
    }
    
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
    
    colors = {'subsample': colorsys.hsv_to_rgb(0, 1, 0.7),
              'good_oracle': colorsys.hsv_to_rgb(0.25, 1, 0.7),
              'bad_oracle': colorsys.hsv_to_rgb(0.5, 1, 0.7),
              'RGBx': colorsys.hsv_to_rgb(0.75, 1, 0.7)}
        
    for shader_name in slicing_datas.keys():
        shader_datas = slicing_datas[shader_name]
        
        all_plot_datas = {}
        
        rgbx_info = None
        
        if shader_name == 'mandelbulb':
            shader_name = 'mandelbulb'
        
        for category in shader_datas.keys():
            
            if category == 'baseline':
                assert len(shader_datas[category]) == 2
                rgbx_info = get_info(shader_datas[category][0], True, shader_datas[category][1])
                continue
            
            datapoints = shader_datas[category]

            perceptual = np.empty(len(datapoints))
            xvals = np.empty(len(datapoints))
            l2 = np.empty(len(datapoints))
            
            for i in range(len(datapoints)):
                dirname = datapoints[i]
                data_info = get_info(dirname)
                perceptual[i] = data_info['perceptual']
                xvals[i] = data_info['trace']
                l2[i] = data_info['l2']
                
            all_plot_datas[category] = {'perceptual': perceptual, 'trace': xvals, 'l2': l2}
            
        assert rgbx_info is not None
        
        for metric in ['perceptual', 'l2']:
        
            fig = plt.figure()
            fig.set_size_inches(7, 3.5)
            
            comparisons = {}

            for category in all_plot_datas.keys():

                current_perceptual = all_plot_datas[category][metric]
                current_trace = all_plot_datas[category]['trace']

                plt.plot(current_trace, current_perceptual, 'o', linewidth=10, markeredgewidth=10, label=category, color=colors[category])

                current_perceptual = np.concatenate((current_perceptual, np.array([rgbx_info[metric]])))
                current_trace = np.concatenate((current_trace, np.array([rgbx_info['trace']])))

                plt.plot(current_trace, current_perceptual, linewidth=2.25, color=colors[category])
                
                if 'good' in category:
                    comparisons['good'] = all_plot_datas[category][metric]
                elif 'bad' in category:
                    comparisons['bad'] = all_plot_datas[category][metric]
                elif 'subsample' in category:
                    comparisons['subsample'] = all_plot_datas[category][metric]

            plt.plot([rgbx_info['trace']], [rgbx_info[metric]], 'o', linewidth=10, markeredgewidth=10, label='RGBx', color=colors['RGBx'])

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