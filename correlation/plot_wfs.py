import os
from matplotlib import pyplot as plt
import numpy as np
from numpy import save, load
from matplotlib.lines import Line2D
import keras

######################################################################################################################
"""
This script takes 8 channel data and plots individual events that are above the netout_cut probability. A model needs to be specified to run this plotting script.
To plot signal, the data variable needs to be set to s and to plot noise, it needs to be set to n.
"""
######################################################################################################################
PathToARIANNA = os.environ['ARIANNA_Experiment']
n = np.load(PathToARIANNAData + '/data/noise.npy') #input a subset of the data here so that you can validate on the other set
s = np.load(PathToARIANNAData + '/data/signal.npy') #make sure the signal and noise subset of data are the same size
model_path = PathToARIANNAData + '/models_h5_files/'
model = keras.models.load_model(model_path + f'trained_CNN_1l-10-8-10h5_fltn_sigm_valloss_p4_n0-10k_s0-1500_shuff_monitortraining_0.h5')
netout_cut = 0.5 #change this number to plot waveforms above this network output cut value
if s.ndim==2: # for data of shape (event_num, samples), this reformats it into 3 dimensions
    s = np.reshape(s, (s.shape[0], 1, s.shape[1]))
    n = np.reshape(n, (n.shape[0], 1, n.shape[1]))
signal = np.reshape(s, (s.shape[0], s.shape[1], s.shape[2], 1))
noise = np.reshape(n, (n.shape[0], n.shape[1], n.shape[2], 1))
ch = signal.shape[1]

data = s #change this value to n to plot the noise data file events


def plot():
    probs = model.predict(data)
    x = np.linspace(1, 256, 256)
    for i,prob in enumerate(probs):
        if prob>netout_cut:
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
            fig.text(0.03, 0.5, 'voltage [mV]', ha='center', va='center', rotation='vertical',fontsize=18)
            plt.xticks(size=13)
            # plt.yticks(size=15)
            plt.show()
            
def main()
    plot()
            
if __name__== "__main__":
    main()
