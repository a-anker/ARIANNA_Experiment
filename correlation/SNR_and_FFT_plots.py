import os
from matplotlib import pyplot as plt
import numpy as np
from numpy import save, load
import keras
import time
import glob
import json
from matplotlib import pyplot as plt
from numpy import save, load
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
from matplotlib.lines import Line2D
from matplotlib import rc

######################################################################################################################
"""
This script plots the probability network output values vs snr with dl_snr_hist(),
the ffts of a particular input data set the user specifies with fft(), and 
generally plots the snr values for all four measured and simulated signal and noise data sets in plot_all_snrs()
"""
######################################################################################################################
PathToARIANNA = os.environ['ARIANNA_Experiment']
mn = np.load(PathToARIANNAData + '/data/measured_noise.npy') 
ms = np.load(PathToARIANNAData + '/data/measured_signal.npy') 
sn = np.load(PathToARIANNAData + '/data/noise.npy') 
ss = np.load(PathToARIANNAData + '/data/signal.npy') 
model_path = PathToARIANNAData + '/models_h5_files/'
model = keras.models.load_model(model_path + 'trained_CNN_1l-10-4-10-do0.5_fltn_sigm_valloss_p4_noise0-18k_amp_below_70mV_CRsignal0-6k_repeat_sig_shuff_monitortraining_0.h5')
chans = 4

if ss.ndim==2: # for data of shape (event_num, samples), this reformats it into 3 dimensions
    ss = np.reshape(ss, (ss.shape[0], 1, ss.shape[1]))
    sn = np.reshape(sn, (sn.shape[0], 1, sn.shape[1]))
    ms = np.reshape(ms, (ms.shape[0], 1, ms.shape[1]))
    mn = np.reshape(mn, (mn.shape[0], 1, mn.shape[1]))
        
signal = np.reshape(ss, (ss.shape[0], ss.shape[1], ss.shape[2],1))
msignal = np.reshape(ms, (ms.shape[0], ms.shape[1], ms.shape[2],1))
noise = np.reshape(mn, (mn.shape[0], mn.shape[1], mn.shape[2],1))
sim_noise = np.reshape(sn, (sn.shape[0], sn.shape[1], sn.shape[2],1))

fig = plt.figure()
ax = fig.add_subplot(111)

def get_snrs(data):
    ary = np.zeros((data.shape[0]))
    for evt in range(data.shape[0]):
        max_ch_val=-1
        for ch in range(chans):
            trace = data[evt,ch]
            mx_intermed = np.amax(abs(trace))   # units in mV
            if mx_intermed>max_ch_val:
                max_ch_val=mx_intermed
       
        ary[evt] = max_ch_val*1000 #just max val and not snr
    return ary

def dl_snr_hist(data):
        probsm = model.predict(data)
        msnrs = get_snrs(data) 
        plt.scatter(probsm,msnrs)
        plt.ylabel('max(|amplitude|) [mV]',fontsize=17)
        plt.xlabel('network output',fontsize=17)
        plt.title('noise')
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.show()


def plot_all_snrs():
        msnrs = get_snrs(msignal) 
        ssnrs = get_snrs(signal) 
        nsnrs = get_snrs(noise) 
        snsnrs = get_snrs(sim_noise) 

        d=False
        ax.hist(msnrs, bins=10, histtype='step', color='black', linestyle='solid',
                label='tagged CRs', linewidth=1.5, density=d)
        ax.hist(ssnrs, bins=700, histtype='step', color='blue', linestyle='solid',
                label='CR signal', linewidth=1.5, density=d)
        ax.hist(sn_snrs, bins=5, histtype='step', color='blue', linestyle='solid',
                label='simulated thermal noise', linewidth=1.5, density=d,weights=weights)
        ax.hist(nsnrs, bins=600, histtype='step', color='black', linestyle='dashdot',
                label='st52: experimental data', linewidth=1.5, density=d)
        ax.axvline(x=60,color = 'red',linestyle='dashed',label='cut line')

        handles, labels = ax.get_legend_handles_labels()
        new_handles = [Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle()) for h in handles[1:]]
        new_handles = [handles[0],new_handles[0],new_handles[1]]
        plt.legend(loc='upper right', handles=new_handles, labels=labels, fontsize=18)

        plt.xlabel('max(|amplitude|) [mV]', fontsize=18) #'max(|amplitude|) / ' + r"$V_{RMS}^{noise}$"
        plt.ylabel('events', fontsize=18)
        # plt.title('stn 52')
        plt.xticks(size=18)
        plt.yticks(size=18)
        plt.xlim((20, 250))
        plt.ylim((10**(-1), 10**(5)))
        plt.yscale('log')
        plt.show()


def fft(data):
        step = 1 * 10**(-9)
        sample = 256
        N = 511 #number of sample points
        T = 1.0 * 10 ** (-9) #sample spacing
        ff = np.fft.rfftfreq(N, T) / 10**6

        for i in data:
                fig, axs = plt.subplots(4, 2)
                for x,ch in enumerate(axs):
                        for c in ch:
                                valnoise = np.fft.rfft(i[x]) / step * 2 ** 0.5
                                print(valnoise.shape,ff.shape)
                                c.plot(ff, np.abs(valnoise) / 256, label=f'tagged CRs')
                plt.xlabel('MHz')
                plt.show()


def main():
    plot_all_snrs()
#     dl_snr_hist(signal)
#     fft(signal)


if __name__== "__main__":
    main()
