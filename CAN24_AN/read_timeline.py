import os
import json
import numpy
import sys
import glob

def read_time_dur(data_list):
    t_min = numpy.inf
    t_max = -numpy.inf

    for item in data_list:
        if 'ts' in item.keys():
            if item['ts'] < t_min:
                t_min = item['ts']
            if 'dur' in item.keys():
                if item['dur'] + item['ts'] > t_max:
                    t_max = item['dur'] + item['ts']

    return t_max - t_min

def main():
    dataroot = sys.argv[1]
    subfolder = sys.argv[2]

    cwd = os.getcwd()
    os.chdir(os.path.join(dataroot, subfolder))
    files = glob.glob('nn_*.json')
    files.sort(key=os.path.getmtime)
    files = [file for file in files if file.startswith('nn_') and file.endswith('.json')]
    os.chdir(cwd)

    print(files)

    nburns = 10
    file_no = len(files) - nburns
    times = numpy.zeros(file_no)

    for n in range(len(files)):

        if n < nburns:
            continue

        filename = files[n]
        with open(os.path.join(dataroot, subfolder, filename)) as file:
            data = json.load(file)
        data_list = data['traceEvents']
        
        times[n-nburns] = read_time_dur(data_list)
    numpy.save(os.path.join(dataroot, subfolder, 'timeline_value.npy'), times)

    print('min time', numpy.min(times))
    print('mean time', numpy.mean(times))
    print('median_time', numpy.median(times))

if __name__ == '__main__':
    main()
