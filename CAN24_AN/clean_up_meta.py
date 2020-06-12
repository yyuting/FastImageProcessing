import os
import sys
import shutil

def has_meta(dirname):
    current_files = os.listdir(dirname)
    has_meta = False
    
    if 'checkpoint' in current_files:
        return 'model.ckpt.meta' in current_files
    else:
        for subdir in current_files:
            if os.path.isdir(os.path.join(dirname, subdir)):
                current_files = os.listdir(os.path.join(dirname, subdir))
                if 'model.ckpt.meta' not in current_files:
                    return False
                else:
                    has_meta = True
    return has_meta

def clean_up_meta(dirname):
    current_files = os.listdir(dirname)
    has_meta = False
    
    if 'checkpoint' in current_files:
        if 'model.ckpt.meta' in current_files:
            os.remove(os.path.join(dirname, 'model.ckpt.meta'))
    else:
        for subdir in current_files:
            if os.path.isdir(os.path.join(dirname, subdir)):
                current_files = os.listdir(os.path.join(dirname, subdir))
                if 'model.ckpt.meta' in current_files:
                    os.remove(os.path.join(dirname, subdir, 'model.ckpt.meta'))
    

dir = sys.argv[1]

subdirs = os.listdir(dir)

keep_dir = None

if 'best_val' in subdirs and has_meta(os.path.join(dir, 'best_val')):
    keep_dir = 'best_val'
    
for subdir in sorted(subdirs):
    
    if not os.path.isdir(os.path.join(dir, subdir)):
        continue
    
    if has_meta(os.path.join(dir, subdir)):
        if keep_dir is None:
            keep_dir = subdir
        elif keep_dir == subdir:
            continue
        else:
            clean_up_meta(os.path.join(dir, subdir))