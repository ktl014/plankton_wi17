import sys
import os
from datetime import datetime

if __name__ == '__main__':
    cwd = os.getcwd()
    subsets = os.listdir(cwd)
    if len(subsets) != 10:
        print(subsets)
        raise ValueError('Need 10 subsets')
    
    time = []
    for data in subsets:
        time.append(datetime.strptime(data, '%Y-%m-%d_%H:%M:%S'))
    
    time.sort()
    for i,subset in enumerate(time):
        print "Rename " + subset.strftime('%Y-%m-%d_%H:%M:%S') + ' to ' + str(i)
        os.rename(subset.strftime('%Y-%m-%d_%H:%M:%S'),str(i))
    

    
    