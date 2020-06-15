import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy
import numpy as np
import os

dirs = ['/n/fs/visualai-scr/yutingy/boids_res_20_64_validate_switch_label/test',
        '/n/fs/visualai-scr/yutingy/boids_res_20_64_validate_switch_label_aux/test'
]

errs = {}

fig = plt.figure()
for dir in dirs:
    err = np.load(os.path.join(dir, 'all_seq_err.npy'))
    err = np.mean(err, 0)
    
    if 'aux' in dir:
        label = 'baseline'
    else:
        label = 'ours'
        
    errs[label] = err
    
    plt.plot(np.arange(err.shape[0]), err, label=label)
    
plt.legend()
plt.xlabel('# of simulation steps')
plt.ylabel('L2 error')

plt.savefig('result_figs/boids_seq_err_len_150.png')
plt.close(fig)


fig = plt.figure()
plt.plot(np.arange(errs['ours'].shape[0]), errs['ours'] / errs['baseline'])
plt.xlabel('# of simulation steps')
plt.ylabel('ours err percentage relative to baseline')
plt.title('ratio at length 100 is %f' % (errs['ours'][-1] / errs['baseline'][-1]))
plt.savefig('result_figs/boids_seq_err_ratio_len_150.png')
plt.close(fig)