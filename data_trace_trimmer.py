from matplotlib import pyplot as plt
import numpy as np
from numpy import save
import os
import operator
from scipy.signal import find_peaks


data = np.load('/Users/astrid/Desktop/ML_paper/cross_correlation_study/data_signal_LPDA_ch0_0.036sa_10.0vrms_no_noise.npy')

input_file = data
window_size = 100
t_len = 256  # trace length

times = np.arange(t_len)
trimmed_array = np.empty((len(input_file), window_size))
for i, event in enumerate(input_file):
    # event = event[0]
    # diffs1 = np.zeros(len(event) - 2)
    # diffs2 = np.zeros(len(event) - 2)
    # for j in range(len(event) - 2):
    #     diffs1[j] = abs(event[j + 1] - event[j])
    #     diffs2[j] = abs(event[j + 2] - event[j])

    # index1, mid_pt1 = max(enumerate(diffs1), key=operator.itemgetter(1))
    # index2, mid_pt2 = max(enumerate(diffs2), key=operator.itemgetter(1))
    # if mid_pt1 > mid_pt2:
    #     mid_pt = index1
    index1, mid_pt1 = max(enumerate(event), key=operator.itemgetter(1))
    index2, mid_pt2 = min(enumerate(event), key=operator.itemgetter(1))
    if abs(mid_pt1) > abs(mid_pt2):
        mid_pt = index1
    else:
        mid_pt = index2

    if mid_pt > window_size / 2 - 1 and mid_pt < (t_len - 1) - window_size / 2:
        trimmed_array[i] = event[int(mid_pt - window_size / 2):int(mid_pt + window_size / 2)]
        # plt.plot(times[int(mid_pt - window_size / 2):int(mid_pt + window_size / 2)], trimmed_array[i])
    elif mid_pt > (t_len - 1) - window_size / 2:
        trimmed_array[i] = event[(t_len - 1) - window_size:(t_len - 1)]
        # plt.plot(times[255 - window_size:255], trimmed_array[i])
    elif mid_pt < window_size / 2:
        trimmed_array[i] = event[0:int(window_size)]
        mid_pt = int(window_size / 2)
        # plt.plot(times[int(mid_pt - window_size / 2):int(mid_pt + window_size / 2)], trimmed_array[i])

        # plt.plot(times[int(mid_pt - window_size / 2):int(mid_pt + window_size / 2)], trimmed_array[i])
        # plt.plot(np.linspace(1, 240, 240), event)
        # plt.show()

save(os.path.join("/Users/astrid/Desktop/ML_paper/cross_correlation_study/trimmed100_data_signal_LPDA_ch0_0.036sa_10.0vrms_no_noise.npy"), trimmed_array)  # change
