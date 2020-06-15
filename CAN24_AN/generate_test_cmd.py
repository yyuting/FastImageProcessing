import os
import sys

model_dirs_file = sys.argv[1]
example_cmd_file = sys.argv[2]
example_model = sys.argv[3]
out_file = sys.argv[4]

dirs = open(model_dirs_file).read().split('\n')

cmd = open(example_cmd_file).read()

str = ''

for dir in dirs:
    new_cmd = cmd.replace(example_model, dir)
    str += new_cmd + '\n'
    
open(out_file, 'w').write(str)