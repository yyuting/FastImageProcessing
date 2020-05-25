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
        'good oracle': [
            '1x_1sample_mandelbrot_highest_score_05',
            '1x_1sample_mandelbrot_highest_score_025',
            '1x_1sample_mandelbrot_highest_score_0125',
            '1x_1sample_mandelbrot_highest_score_00625',
            '1x_1sample_mandelbrot_highest_score_003125'
        ],
        'bad_oracle': [
            '1x_1sample_mandelbrot_lowest_score_05',
            '1x_1sample_mandelbrot_lowest_score_025',
            '1x_1sample_mandelbrot_lowest_score_0125',
            '1x_1sample_mandelbrot_lowest_score_00625',
            '1x_1sample_mandelbrot_lowest_score_003125'
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
            '1x_1sample_trippy_subsample_2_highest_score_05',
            '1x_1sample_trippy_subsample_2_highest_score_025',
            '1x_1sample_trippy_subsample_2_highest_score_0125',
            '1x_1sample_trippy_subsample_2_highest_score_00625',
            '1x_1sample_trippy_subsample_2_highest_score_003125',
            '1x_1sample_trippy_highest_score_0015625'
        ],
        'bad_oracle': [
            '1x_1sample_trippy_subsample_2_lowest_score_05',
            '1x_1sample_trippy_subsample_2_lowest_score_025',
            '1x_1sample_trippy_subsample_2_lowest_score_0125',
            '1x_1sample_trippy_subsample_2_lowest_score_00625',
            '1x_1sample_trippy_subsample_2_lowest_score_003125',
            '1x_1sample_trippy_lowest_score_0015625'
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
    'primitives': {
        'baseline': [
            '1x_1sample_primitives_aux_largest_capacity',
            '1x_1sample_primitives_all'
        ],
        'good_oracle': [
            '1x_1sample_primitives_highest_score_05',
            '1x_1sample_primitives_highest_score_025',
            '1x_1sample_primitives_highest_score_0125'
        ],
        'bad_oracle': [
            '1x_1sample_primitives_lowest_score_05',
            '1x_1sample_primitives_lowest_score_025',
            '1x_1sample_primitives_lowest_score_0125'
        ],
        'subsample': [
            '1x_1sample_primitives_subsample_2',
            '1x_1sample_primitives_stratified_subsample_4',
            '1x_1sample_primitives_stratified_subsample_8',
            '1x_1sample_primitives_stratified_subsample_16'
        ]
    },
    'mandelbulb': {
        'baseline': [
            '1x_1sample_mandelbulb_aux_largest_capacity',
            '1x_1sample_mandelbulb_all'
        ],
        'good_oracle': [
            '1x_1sample_mandelbulb_highest_score_05',
            '1x_1sample_mandelbulb_highest_score_025',
            '1x_1sample_mandelbulb_highest_score_0125',
            '1x_1sample_mandelbulb_highest_score_00625'
        ],
        'bad_oracle': [
            '1x_1sample_mandelbulb_lowest_score_05',
            '1x_1sample_mandelbulb_lowest_score_025',
            '1x_1sample_mandelbulb_lowest_score_0125',
            '1x_1sample_mandelbulb_lowest_score_00625'
        ],
        'subsample': [
            '1x_1sample_mandelbulb_stratified_subsample_2',
            '1x_1sample_mandelbulb_stratified_subsample_4',
            '1x_1sample_mandelbulb_stratified_subsample_8',
            '1x_1sample_mandelbulb_stratified_subsample_16'
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
    
    for shader_name in slicing_datas.keys():
        shader_datas = slicing_datas[shader_name]
        
        all_plot_datas = {}
        
        rgbx_info = None
        
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

            for category in all_plot_datas.keys():

                current_perceptual = all_plot_datas[category][metric]
                current_trace = all_plot_datas[category]['trace']

                plt.plot(current_trace, current_perceptual, 'o', linewidth=10, markeredgewidth=10, label=category)

                current_perceptual = np.concatenate((current_perceptual, np.array([rgbx_info[metric]])))
                current_trace = np.concatenate((current_trace, np.array([rgbx_info['trace']])))

                plt.plot(current_trace, current_perceptual, linewidth=2.25)

            plt.plot([rgbx_info['trace']], [rgbx_info[metric]], 'o', linewidth=10, markeredgewidth=10, label='RGBx')

            plt.xlabel('Trace Length used in Model', fontsize=fontsize)
            plt.ylabel('%s error' % metric, fontsize=fontsize)
            plt.legend()

            plt.tight_layout()

            plt.savefig(os.path.join('result_figs', '%s_trace_contribution_%s.png' % (shader_name, metric)))
            plt.close(fig)
        
        print(shader_name)
    
        
if __name__ == '__main__':
    main()