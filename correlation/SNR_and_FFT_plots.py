import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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


fig = plt.figure()
ax = fig.add_subplot(111)

noise = np.load(f'/Users/astrid/Desktop/st52_deeplearning/data/stn52_2019_cr_0123568_all_shuffledBB.npy') #[42000:]
# n_mask_blw60 = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stn52_2019_cr_0123568_all_shuffledBB_all_amp_below_70mV_True.npy')
# mask = n_mask_blw60>0
# noise = noise[mask]
# noise = noise[18000:]
sim_noise =  np.load('/Users/astrid/Desktop/st61_deeplearning/data/simulated_noise/data_noise_4.3SNR_all.npy')

signal = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stn52_sim_cr.npy')

msignal = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stn52_2019_cr_candidates.npy')
print(msignal.shape)
noise = np.expand_dims(noise, axis=-1)
# noises = np.expand_dims(noises, axis=-1)
fig = plt.figure()
ax = fig.add_subplot(111)

signal = np.reshape(signal, (signal.shape[0], signal.shape[1],signal.shape[2],1))
msignal = np.reshape(msignal, (msignal.shape[0], msignal.shape[1],msignal.shape[2],1))
noise = np.reshape(noise, (noise.shape[0], noise.shape[1],noise.shape[2],1))
sim_noise = np.reshape(sim_noise, (sim_noise.shape[0], sim_noise.shape[1],sim_noise.shape[2],1))



def get_snrs(data):
    ary = np.zeros((data.shape[0]))
    for evt in range(data.shape[0]):
        max_ch_val=-1
        for ch in range(4):
            trace = data[evt,ch]
            mx_intermed = np.amax(abs(trace))   # units in mV
            if mx_intermed>max_ch_val:
                max_ch_val=mx_intermed
       
        ary[evt] = max_ch_val*1000 #just max val and not snr
    return ary

def dl_snr_hist():
        model = keras.models.load_model(f'/Users/astrid/Desktop/st52_deeplearning/h5s/trained_CNN_1l-10-4-10-do0.5_fltn_sigm_valloss_p4_noise0-18k_amp_below_70mV_CRsignal0-6k_repeat_sig_shuff_monitortraining_0.h5')
        probsm = model.predict(noise)
        # count=0
        # for i in probsm:
        #         if i>0.5:
        #                 count+=1
        # print(count)
        msnrs = get_snrs(noise[:,4:]) 
        plt.scatter(probsm,msnrs)
        plt.ylabel('max(|amplitude|) [mV]',fontsize=17)
        plt.xlabel('network output',fontsize=17)
        plt.title('noise')
        plt.xticks(size=15)
        plt.yticks(size=15)
        plt.show()
# dl_snr_hist()


def plot_all_snrs():

        msnrs = get_snrs(msignal[:,4:]) 
        ssnrs = get_snrs(signal[:,4:]) 
        nsnrs = get_snrs(noise[:,4:]) 
        sn_snrs = get_snrs(sim_noise[:,:4]) 

        d=False
        # ax.hist(msnrs, bins=10, histtype='step', color='black', linestyle='solid',
        #         label='tagged CRs', linewidth=1.5, density=d)
        # ax.hist(ssnrs, bins=700, histtype='step', color='blue', linestyle='solid',
        #         label='CR signal', linewidth=1.5, density=d)
        weights=0.3/(np.ones(len(sn_snrs)))
 
        ax.hist(sn_snrs, bins=5, histtype='step', color='blue', linestyle='solid',
                label='simulated thermal noise', linewidth=1.5, density=d,weights=weights)
        ax.hist(nsnrs, bins=600, histtype='step', color='black', linestyle='dashdot',
                label='st52: experimental data', linewidth=1.5, density=d)
        ax.axvline(x=60,color = 'red',linestyle='dashed',label='60 mV cut line')



        handles, labels = ax.get_legend_handles_labels()
        print(handles)
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
plot_all_snrs()


def fft(data):
        step = 1 * 10**(-9)
        sample = 256
        
        # Number of sample points
        N = 511
        # sample spacing
        T = 1.0 * 10 ** (-9)
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



# fft(msignal)
