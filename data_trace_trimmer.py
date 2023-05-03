from matplotlib import pyplot as plt
import numpy as np
from numpy import save
import os
import operator
from scipy.signal import find_peaks

######################################################################################################################
"""
This takes in 1 channels worth of data with size (event_num, num_of_samps), trims it to a specified window around the max value, and then saves it as a numpy array
If for example the window size is 100 and the maximum occurs beflore 50 or after 206, the first 100 sampoles or last 100 samples is chosen instead.
"""
######################################################################################################################
PathToARIANNA = os.environ['ARIANNA_Experiment']
input_File = np.load(PathToARIANNA + '/data/arianna_data.py')

window_size = 100 # final trace length
t_len = 256  # input trace length

times = np.arange(t_len)
trimmed_array = np.empty((len(input_file), window_size))
for i, event in enumerate(input_file):
    
    index1, mid_pt1 = max(enumerate(event), key=operator.itemgetter(1))
    index2, mid_pt2 = min(enumerate(event), key=operator.itemgetter(1))
    if abs(mid_pt1) > abs(mid_pt2):
        mid_pt = index1
    else:
        mid_pt = index2

    if mid_pt > window_size / 2 - 1 and mid_pt < (t_len - 1) - window_size / 2:
        trimmed_array[i] = event[int(mid_pt - window_size / 2):int(mid_pt + window_size / 2)]
    elif mid_pt > (t_len - 1) - window_size / 2:
        trimmed_array[i] = event[(t_len - 1) - window_size:(t_len - 1)]
    elif mid_pt < window_size / 2:
        trimmed_array[i] = event[0:int(window_size)]
        mid_pt = int(window_size / 2)
     
save(os.path.join(PathToARIANNA, '/data/trimmed100_arianna_data.py')), trimmed_array)
