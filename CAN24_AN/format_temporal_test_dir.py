import shutil
import os
import sys
import numpy as np
import numpy

def main():
    base_dir = sys.argv[1]
    dir = os.path.join(base_dir, 'test_img')
    
    count = 0
    
    camera_pos = []
    
    for mode in ['test_close', 'test_far', 'test_middle']:
        if mode == 'test_middle':
            nframes = 20
        else:
            nframes = 5
            
        camera_pos.append(np.load(os.path.join(base_dir, '%s.npy' % mode)))
            
        for i in range(nframes):
            for t in [0, 1, 29]:
                src = os.path.join(dir, '%s_ground%d%05d.png' % (mode, t, i))
                dst = os.path.join(dir, 'test_ground%d%05d.png' % (t, count))
                os.rename(src, dst)
                
            count += 1
                
    camera_pos = np.concatenate(camera_pos, 0)
    np.save(os.path.join(base_dir, 'test.npy'), camera_pos)
    
if __name__ == '__main__':
    main()