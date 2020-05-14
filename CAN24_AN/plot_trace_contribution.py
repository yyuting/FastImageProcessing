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
            '1x_1sample_mandelbrot_highest_score_025'
        ],
        'bad_oracle': [
            '1x_1sample_mandelbrot_lowest_score_05'
        ],
        'subsample': [
            '1x_1sample_mandelbrot_stratified_subsample_2',
            '1x_1sample_mandelbrot_stratified_subsample_4',
            '1x_1sample_mandelbrot_stratified_subsample_8'
        ]
    },
    'trippy': {
        'baseline': [
            '1x_1sample_trippy_aux_largest_capacity',
            '1x_1sample_trippy_subsample_2'
        ],
        'good_oracle': [
            '1x_1sample_trippy_subsample_2_highest_score_05',
            '1x_1sample_trippy_subsample_2_highest_score_025'
        ],
        'bad_oracle': [
            '1x_1sample_trippy_subsample_2_lowest_score_05',
            '1x_1sample_trippy_subsample_2_lowest_score_025'
        ],
        'subsample': [
            '1x_1sample_trippy_stratified_subsample_4'
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
    
    ans = {
        'trace': trace_len, 
        'perceptual': perceptual_score
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
            
            for i in range(len(datapoints)):
                dirname = datapoints[i]
                data_info = get_info(dirname)
                perceptual[i] = data_info['perceptual']
                xvals[i] = data_info['trace']
                
            all_plot_datas[category] = {'perceptual': perceptual, 'trace': xvals}
            
        assert rgbx_info is not None
        
        fig = plt.figure()
        fig.set_size_inches(7, 3.5)
        
        for category in all_plot_datas.keys():
        
            current_perceptual = all_plot_datas[category]['perceptual']
            current_trace = all_plot_datas[category]['trace']
            
            plt.plot(current_trace, current_perceptual, 'o', linewidth=10, markeredgewidth=10, label=category)
            
            current_perceptual = np.concatenate((current_perceptual, np.array([rgbx_info['perceptual']])))
            current_trace = np.concatenate((current_trace, np.array([rgbx_info['trace']])))
            
            plt.plot(current_trace, current_perceptual, linewidth=2.25)
            
        plt.plot([rgbx_info['trace']], [rgbx_info['perceptual']], 'o', linewidth=10, markeredgewidth=10, label='RGBx')
                    
        plt.xlabel('Trace Length used in Model', fontsize=fontsize)
        plt.ylabel('Perceptual Error', fontsize=fontsize)
        plt.legend()
        
        plt.tight_layout()

        plt.savefig(os.path.join('result_figs', '%s_trace_contribution.png' % shader_name))
        plt.close(fig)
        
        print(shader_name)
    
        
if __name__ == '__main__':
    main()