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
        
    if mode in ['highest_score', 'lowest_score']:
        
        sample_budget = int((score.shape[0] * sample_portion))
        
        sampled_ind = col_aux_inds.tolist()
        
        if len(sampled_ind) >= sample_budget:
            raise "RGBx length already larger than sample budget"
        
        sorted_ind = np.argsort(score)
        if mode == 'highest_score':
            sorted_ind = sorted_ind[::-1]
            
        for ind in sorted_ind:
            
            if ind not in col_aux_inds:
                sampled_ind.append(ind)
            
            if len(sampled_ind) == sample_budget:
                break
                
    elif mode in ['highest_subsample', 'lowest_subsample', 'subsample']:
        
        
        
        sampled_ind = col_aux_inds.tolist()
        
        sample_portion = int(sample_portion)
        
        ind = 0
        
        remaining_inds = np.arange(score.shape[0])
        col_aux_inds = np.sort(col_aux_inds)[::-1]
        
        for ind in col_aux_inds:
            remaining_inds = np.concatenate((remaining_inds[:ind], remaining_inds[ind+1:]))
            
        for i in range(0, remaining_inds.shape[0], sample_portion):
            start_i = i
            end_i = min(i + sample_portion, remaining_inds.shape[0])
            
            all_inds = remaining_inds[start_i:end_i]
            
            if mode == 'highest_subsample':
                chosen_i = i + np.argmax(score[all_inds])
            elif mode == 'lowest_subsample':
                chosen_i = i + np.argmin(score[all_inds])
            else:
                # using stratified subsample
                chosen_i = np.random.randint(start_i, high=end_i)
                
            chosen_ind = remaining_inds[chosen_i]
            sampled_ind.append(chosen_ind)
            
    else:
        raise
                
    _, score_shortname = os.path.split(score_file)
                
    dst_filename = os.path.join(dst, mode + '_' + score_shortname + str(sample_portion) + '.npy')
    np.save(dst_filename, sampled_ind)
    
if __name__ == '__main__':
    main()