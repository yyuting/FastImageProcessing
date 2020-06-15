import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import numpy as np
import colorsys
import matplotlib.patheffects as path_effects
from matplotlib.patches import BoxStyle


dirs = ['scratch/boids_res_20_64_validate_switch_label/test',
        'scratch/boids_res_20_64_validate_switch_label_aux/test',
        'scratch/boids_res_20_64_validate_switch_label/mean1_test']

def main():
    
    errs = []
    fontsize = 12
    linewidth = 2
    offset = 2.5
    
    for i in range(3):
        dir = dirs[i]
        err_file = os.path.join(dir, 'all_l2.npy')
        errs.append(np.mean(np.load(err_file), 1))
        
        
        
    xvals = np.arange(1, errs[0].shape[0] + 1)
    
    idx = np.arange(xvals.shape[0])[15:]
    
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 4)
    
    
    line0, = plt.plot(xvals[idx], errs[0][idx], label='Our Method', linewidth=linewidth, color=[0, 0, 1])
    line1, = plt.plot(xvals[idx], errs[1][idx], label='I/O Baseline', linewidth=linewidth, color=[1, 0, 0])
    line2, = plt.plot(xvals[idx], errs[2][idx], label='Naive Baseline', linewidth=linewidth, color=[0, 0, 0])
    
    #plt.plot(xvals[19:64], errs[0][19:64], color=line0.get_color(), path_effects=[path_effects.SimpleLineShadow(offset=(0, -offset))], linewidth=linewidth)
    #plt.plot(xvals[19:64], errs[1][19:64], color=line1.get_color(), path_effects=[path_effects.SimpleLineShadow(offset=(0, -offset))], linewidth=linewidth)
    #plt.plot(xvals[19:64], errs[2][19:64], color=line2.get_color(), path_effects=[path_effects.SimpleLineShadow(offset=(0, -offset))], linewidth=linewidth)
    
    plt.axvspan(20, 64, facecolor=[0, 0, 0], alpha=0.15)
    plt.text(31.5, 0.012, 'Training', fontsize=fontsize, bbox=dict(facecolor='white', alpha=1, edgecolor='k', linewidth=linewidth, boxstyle=BoxStyle("Round", pad=1)))
    
    
    #plt.ylim(0, 0.0139)
    
    plt.xlabel('Inference Step Size', fontsize=fontsize)
    plt.ylabel('L2 error', fontsize=fontsize)
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(fontsize)
    
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('result_figs/boids_metric.png')
    
if __name__ == '__main__':
    main()