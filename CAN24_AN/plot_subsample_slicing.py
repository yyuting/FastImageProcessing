import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import numpy as np
import colorsys

base_dir = 'mandelbrot_slice'

def get_val(mode, metric):
    def func(datapoint):
        if mode in ['test', 'train']:
            if metric == 'l2':
                file = 'score.txt'
            elif metric == 'perceptual':
                file = 'perceptual_tf.txt'
            else:
                raise
            filename = os.path.join(datapoint['dir'], mode, file)
            yval = float(open(filename).read())
            return yval
        else:
            if metric == 'l2':
                file = 'score_breakdown.txt'
            elif metric == 'perceptual':
                file = 'perceptual_tf_breakdown.txt'
            else:
                raise
            filename = os.path.join(datapoint['dir'], 'test', file)
            line = open(filename).read()
            vals = line.split(',')
            if mode == 'test_close':
                yval = float(vals[0])
            elif mode == 'test_far':
                yval = float(vals[1])
            elif mode == 'test_middle':
                yval = float(vals[2])
            else:
                raise
            return yval
    return func

def get_time(datapoint):
    filename = os.path.join(datapoint['dir'], 'test', 'time_stats.txt')
    vals = open(filename).read().split(',')
    return float(vals[0]), float(vals[1]), float(vals[2])

# thre -1 and subsample -1 represents RGBA
slicing_datas = [
    {'thre': -1, 'subsample': -1, 'trace': 4, 'dir': '1x_1sample_mandelbrot_tile_automatic_200_aux_repeat'},
    {'thre': 12, 'subsample': 71, 'trace': 12, 'dir': '1x_1sample_mandelbrot_tile_automatic_12'},
    {'thre': 25, 'subsample': 25, 'trace': 25, 'dir': '1x_1sample_mandelbrot_tile_automatic_25'},
    {'thre': 50, 'subsample': 11, 'trace': 46, 'dir': '1x_1sample_mandelbrot_tile_automatic_50_repeat'},
    {'thre': 100, 'subsample': 5, 'trace': 103, 'dir': '1x_1sample_mandelbrot_tile_automatic_100_repeat'},
    {'thre': 200, 'subsample': 3, 'trace': 167, 'dir': '1x_1sample_mandelbrot_tile_automatic_200_repeat'},
    {'thre': 400, 'subsample': 1, 'trace': 381, 'dir': '1x_1sample_mandelbrot_tile_automatic_400'}
]

plots = [
    {'name': 'perceptual_train', 'get_func': get_val('train', 'perceptual'), 'ylabel': 'perceptual_train', 'title': 'perceptual train'},
    {'name': 'perceptual_test', 'get_func': get_val('test', 'perceptual'), 'ylabel': 'perceptual_test', 'title': 'perceptual test'},
    {'name': 'perceptual_test_close', 'get_func': get_val('test_close', 'perceptual'), 'ylabel': 'perceptual_test_close', 'title': 'perceptual test close'},
    {'name': 'perceptual_test_far', 'get_func': get_val('test_far', 'perceptual'), 'ylabel': 'perceptual_test_far', 'title': 'perceptual test far'},
    {'name': 'perceptual_test_middle', 'get_func': get_val('test_middle', 'perceptual'), 'ylabel': 'perceptual_test_middle', 'title': 'perceptual test middle'},
    {'name': 'l2_train', 'get_func': get_val('train', 'l2'), 'ylabel': 'l2_train', 'title': 'l2 train'},
    {'name': 'l2_test', 'get_func': get_val('test', 'l2'), 'ylabel': 'l2_test', 'title': 'l2 test'},
    {'name': 'l2_test_close', 'get_func': get_val('test_close', 'l2'), 'ylabel': 'l2_test_close', 'title': 'l2 test close'},
    {'name': 'l2_test_far', 'get_func': get_val('test_far', 'l2'), 'ylabel': 'l2_test_far', 'title': 'l2 test far'},
    {'name': 'l2_test_middle', 'get_func': get_val('test_middle', 'l2'), 'ylabel': 'l2_test_middle', 'title': 'l2 test middle'},
    {'name': 'inference_time', 'get_func': get_time, 'ylabel': 'inference_time', 'title': 'inference time', 'errorbar': True}
]

def main():
    
    fontsize = 14

    perceptual = np.empty(len(slicing_datas))
    runtime = np.empty(len(slicing_datas))
    xvals = np.empty(len(slicing_datas))
    
    for i in range(len(slicing_datas)):
        data = slicing_datas[i]
        xvals[i] = data['trace']
        
        perceptual[i] = get_val('test', 'perceptual')(data)
        
        if i > 0:
            perceptual[i] /= perceptual[0]
        
        runtime[i] = get_time(data)[0]
    
    perceptual[0] = 1
    runtime *= 1000
        
    #fig = plt.figure()
    #plt.plot(xvals, perceptual_dif, label='different')
    #plt.plot(xvals, perceptual_sim, label='similar')
    #plt.plot(xvals, runtime, label='inference time')
    
    #plt.plot(xvals, perceptual_dif, '.', markersize=12)
    #plt.plot(xvals, perceptual_sim, '.', markersize=12)
    #plt.plot(xvals, runtime, '.', markersize=12)
    
    
    fig, ax1 = plt.subplots()
    fig.set_size_inches(7, 3.5)

    ax1.set_xlabel('Trace Length after Subsampling', fontsize=fontsize)
    
    marker, = ax1.plot(xvals[0], perceptual[0], 'o', color=[0.1, 0.7, 0.1], linewidth=10, markeredgewidth=10)
    
    line0, = ax1.plot(xvals, perceptual, color=colorsys.hsv_to_rgb(0.7, 1, 0.5), label='perceptual', linewidth=2.25)
    ax1.set_ylabel('Relative Error', fontsize=fontsize, color=line0.get_color())
    ax1.plot(xvals, perceptual, '.', markersize=12, color=line0.get_color())
    ax1.tick_params(axis='y', labelcolor=line0.get_color())
    ax1.set_ylim(0, 1.1)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.plot(xvals[0], runtime[0], 'o', color=marker.get_color(), linewidth=10, markeredgewidth=10)
    line2, = ax2.plot(xvals, runtime, color=colorsys.hsv_to_rgb(1, 1, 0.5), label='inference time', linewidth=2.25)
    ax2.plot(xvals, runtime, '.', markersize=12, color=line2.get_color())
    ax2.set_ylabel('Inference Time (ms)', fontsize=fontsize, color=line2.get_color())  # we already handled the x-label with ax1
    ax2.set_ylim(0, 260)
    ax2.tick_params(axis='y', labelcolor=line2.get_color())
    
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_yticklabels()):
            label.set_fontsize(fontsize)
    
    #plt.legend([line0, line2], ['Relative Error', 'Inference Time'], loc=3)
    
    plt.text(250, 60, 'Relative Error', fontsize=fontsize, color=line0.get_color())
    plt.text(250, 195, 'Inference Time', fontsize=fontsize, color=line2.get_color())
    
    plt.text(20, 230, 'RGBx Baseline', fontsize=fontsize, color=marker.get_color())
    
    plt.tight_layout()

    plt.savefig(os.path.join('result_figs', 'mandelbrot_subsample.png'))
    plt.close(fig)
    
    if False:
        for plot in plots:
            xvals = np.empty(len(slicing_datas))
            yvals = np.empty(len(slicing_datas))
            if plot.get('errorbar', False):
                errs = np.empty((2, len(slicing_datas)))
            for i in range(len(slicing_datas)):
                data = slicing_datas[i]
                xvals[i] = data['trace']
                yval = plot['get_func'](data)
                if plot.get('errorbar', False):
                    yvals[i] = yval[0]
                    errs[0, i] = -(yval[1] - yval[0])
                    errs[1, i] = yval[2] - yval[0]
                else:
                    yvals[i] = yval
            fig = plt.figure()
            if plot.get('errorbar', False):
                plt.errorbar(xvals, yvals, yerr=errs)
            else:
                plt.plot(xvals, yvals)

            plt.plot(xvals, yvals, '.', markersize=12)

            plt.xlabel('trace #')
            plt.ylabel(plot['ylabel'])
            plt.title(plot['title'])
            plt.tight_layout()

            plt.savefig(os.path.join(base_dir, plot['name'] + '.png'))
            plt.close(fig)
        
if __name__ == '__main__':
    main()