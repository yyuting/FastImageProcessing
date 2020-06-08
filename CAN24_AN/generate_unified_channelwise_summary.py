import pickle
import sys
import numpy
import numpy as np
import os

thre = 0.01
   
dict = {
    0: 'R color',
    1: 'G color',
    2: 'B color'
}
    

pickle.dump( dict, open( "dict.pickle", "wb" ) )

test = pickle.load( open( "dict.pickle", "rb" ) )

print(test)

dir = sys.argv[1]
shadername = sys.argv[2]

save_trace_dict_file = os.path.join(dir, 'trace_dict_%s.pickle' % shadername)

if len(sys.argv) > 3:
    dict_file = sys.argv[3]
    trace_dict = pickle.load( open( dict_file, "rb" ) )
else:
    if os.path.exists(save_trace_dict_file):
        trace_dict = pickle.load( open( save_trace_dict_file, "rb" ) )
    else:
        trace_dict = {}

channelwise_file = os.path.join(dir, 'encoder_channelwise_taylor_vals_%s.npy' % shadername)
channelwise_score = np.load(channelwise_file)

assert channelwise_score.shape[0] == 48

ind_txt_file = os.path.join(dir, 'encoder_max_channelwise_ind_%s.txt' % shadername)
ind_strs = open(ind_txt_file).read().split('\n')



count = 0
processed = True

summary_str = ''

for line in ind_strs:
    if line.startswith('encoder channel '):
        current_ch = int(line[16:-1])
        processed = False
        assert current_ch == count
        count += 1
    else:
        if len(line):
            assert not processed
            
            processed = True
            
            summary_str += '\nencoder channel %d:\n' % current_ch
            
            inds = [int(val) for val in line.split(',')]
            
            current_score = np.sort(channelwise_score[current_ch])[::-1]
            
            for i in range(len(inds)):
                if current_score[i] < thre:
                    break
                    
                if not inds[i] in trace_dict.keys():
                    print('We are scanning channel %d now!' % current_ch)
                    print('need manual input for trace %d in shader %s' % (inds[i], shadername))
                    ans = input()
                    trace_dict[inds[i]] = ans
                    pickle.dump( trace_dict, open( save_trace_dict_file, "wb" ) )
                    
                summary_str += '%d (%s)\n' % (inds[i], trace_dict[inds[i]])

summary_str = 'Looking for at most 20 feature with contribution score > %f\n\n' % thre + summary_str
open(os.path.join(dir, 'auto_generated_summary_%s.txt' % shadername), 'w').write(summary_str)
                          
                    
        