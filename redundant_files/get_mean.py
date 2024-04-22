import numpy as np
import pdb
fp = open('../foreground_ratios.txt','r')
lines = fp.readlines()
fp.close()

ratios = []

for idx in range(len(lines)):

    ratio = float(lines[idx].strip())
    ratios.append(ratio)

ratios = np.array(ratios)
pdb.set_trace()