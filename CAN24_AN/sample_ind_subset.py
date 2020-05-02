import numpy
import os
import sys
import numpy as np

def main():
    score_file = sys.argv[1]
    col_aux_inds_file = sys.argv[2]
    dst = sys.argv[3]
    mode = sys.argv[4]
    sample_portion = float(sys.argv[5])
    
    score = np.load(score_file)
    col_aux_inds = np.load(col_aux_inds_file)
    
    sample_budget = int((score.shape[0] * sample_portion))
    
    if mode in ['highest_score', 'lowest_score']:
        
        sampled_ind = col_aux_inds.tolist()
        
        sorted_ind = np.argsort(score)
        if mode == 'highest_score':
            sorted_ind = sorted_ind[::-1]
            
        for ind in sorted_ind:
            
            if ind not in col_aux_inds:
                sampled_ind.append(ind)
            
            if len(sampled_ind) == sample_budget:
                break
                
    _, score_shortname = os.path.split(score_file)
                
    dst_filename = os.path.join(dst, mode + '_' + score_shortname + str(sample_portion) + '.npy')
    np.save(dst_filename, sampled_ind)
    
if __name__ == '__main__':
    main()