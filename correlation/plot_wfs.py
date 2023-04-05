import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from matplotlib import pyplot as plt
import numpy as np
from numpy import save, load
from matplotlib.lines import Line2D
from datetime import datetime
import keras
import time
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils

path = "/Users/astrid/Desktop/st61_deeplearning/data/2ndpass/"
ch = 8

signal = np.load(os.path.join(path, "stn61_simulation_noise_trace.npy"))[1500:] #10606 unique signal events. 10606 * 7 = 74242
noise = np.load(os.path.join(path, "stn61_2of4trigger_noise_all_shuffledAA.npy")) #[10000:]
data = noise
signal = np.reshape(signal, (signal.shape[0], signal.shape[1],signal.shape[2],1))
noise = np.reshape(noise, (noise.shape[0], noise.shape[1],noise.shape[2],1))

model = keras.models.load_model(f'/Users/astrid/Desktop/st61_deeplearning/cross_val_study/h5s/trained_CNN_1l-10-8-10h5_fltn_sigm_valloss_p4_n0-10k_s0-1500_shuff_monitortraining_0.h5')
probs = model.predict(signal)
probn = model.predict(noise)

x = np.linspace(1, 256, 256)

for i,prob in enumerate(probn):
    print(i)
    if prob>0.5:
        print(i,prob)
        fig, axs = plt.subplots(8, sharex=True)
    
        axs[0].plot(x, 1000*data[i, 0])
        axs[1].plot(x, 1000*data[i, 1])
        axs[2].plot(x, 1000*data[i, 2])
        axs[3].plot(x, 1000*data[i, 3])
        axs[4].plot(x, 1000*data[i, 4])
        axs[5].plot(x, 1000*data[i, 5])
        axs[6].plot(x, 1000*data[i, 6])
        axs[7].plot(x, 1000*data[i, 7])

        axs[7].set_xlabel('time [ns]',fontsize=18)
        
        for i in [1,2,3,4,5,6,7]:
            axs[i].set_ylabel(f'ch{i}',labelpad=10,rotation=0,fontsize=13)
            axs[i].set_ylim(-80,80)
            axs[i].set_xlim(-3,260)
            axs[i].tick_params(labelsize=13)
        axs[0].tick_params(labelsize=13)
        axs[0].set_ylabel(f'ch{0}',labelpad=3,rotation=0,fontsize=13)
        axs[i].set_xlim(-3,260)
        # axs[3].set_ylabel('mV',labelpad=0)
        fig.text(0.03, 0.5, 'voltage [mV]', ha='center', va='center', rotation='vertical',fontsize=18)
        plt.xticks(size=13)
        # plt.yticks(size=15)
        plt.show()


#with index starting at 10k, event 35449 (from a zero index system) gives the event prob of 1 and event 47537 gives a prob of 0.9882375
