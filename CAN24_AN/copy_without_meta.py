import os
import sys
import shutil

src_dir = sys.argv[1]
dst_dir = sys.argv[2]

for root, dirs, files in os.walk(src_dir):
    for file in files:
        if not file.endswith('.meta'):
            src_file = os.path.join(root, file)
            
            current_src_dir = src_file.replace(src_dir, '')
            if current_src_dir.startswith('/'):
                current_src_dir = current_src_dir[1:]
            
            dst_file = os.path.join(dst_dir, current_src_dir)
            
            current_dst_dir, _ = os.path.split(dst_file)
            
            if not os.path.isdir(current_dst_dir):
                os.makedirs(current_dst_dir)
            
            shutil.copyfile(src_file, dst_file)
            
    print('finished root:')
    print(root)