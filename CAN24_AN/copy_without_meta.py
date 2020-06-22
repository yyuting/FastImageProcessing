import os
import sys
import shutil

src_dir = sys.argv[1]
dst_dir = sys.argv[2]

for root, dirs, files in os.walk(src_dir):
    for file in files:
        if not file.endswith('.meta'):
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_dir, src_file.replace(src_dir, ''))
            
            dst_dir, _ = os.path.split(dst_file)
            
            if not os.path.isdir(dst_dir):
                os.makedirs(dst_dir)
            
            shutil.copyfile(src_file, dst_file)
            
    print('finished root:')
    print(root)