import os
from matplotlib import pyplot as plt
import numpy as np
from numpy import save, load
import keras
import time
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
from matplotlib.lines import Line2D
from scipy.stats import wasserstein_distance
from scipy.stats import kstest
import seaborn as sns
from scipy.stats import ks_2samp
from numpy import random
import time
    
######################################################################################################################
"""
This script loads the pre-trained model and evaluates it on some test data to obtain network output curves for noise, signal, and measured signal.
The measured data is then varied in each bin by the set amount of iteration. Bin values above 4 events vary by a Gaussian. Values below vary by a Poisson.
The Wasserstein distance between this measured signal and signal plus the the distance between the measured signal and noise distributions are computed and plotted.
If the distributions of signal are similar, this plot is expected to give a low x value for their distribution, and their distribution shouldn't overlap with the noise distribution.
"""
###################################################################################################################### 
PathToARIANNA = os.environ['ARIANNA_Experiment']
n = np.load(PathToARIANNAData + '/data/noise.npy') #input a subset of the data here so that you can validate on the other set
s = np.load(PathToARIANNAData + '/data/signal.npy') #make sure the signal and noise subset of data are the same size
ms = np.load(PathToARIANNAData + '/data/measured_signal.npy')
model_path = PathToARIANNAData + '/models_h5_files/'
model = keras.models.load_model(model_path + 'trained_CNN_1l-10-4-10_fltn_sigm_valloss_p4_noise_lowampl_lt60mV_0-20k_CRsignal_no-weight_0-5k_repeat_sig4x_shuff_monitortraining_4upLPDAs_0.h5')
num_bins=100
num_iterations = 1000

if s.ndim==2: # for data of shape (event_num, samples), this reformats it into 3 dimensions
    s = np.reshape(s, (s.shape[0], 1, s.shape[1]))
    n = np.reshape(n, (n.shape[0], 1, n.shape[1]))
    ms = np.reshape(ms, (msl.shape[0], 1, ms.shape[1]))

signal = np.reshape(s, (s.shape[0], s.shape[1], s.shape[2],1))
msignal = np.reshape(ms, (ms.shape[0], ms.shape[1], ms.shape[2],1))
noise = np.reshape(n, (n.shape[0], n.shape[1], n.shape[2],1))


def plot_each_hist(data1,var_data):
        plt.hist(data1,bins=100,histtype='step',range=(0,1),label='experimental CR, varied')
        plt.hist(var_data,bins=100,histtype='step',range=(0,1),label='experimental CR, actual')
        # plt.hist(probs2,bins=100,histtype='step',range=(0,1),label='signal')
        plt.yscale('log')
        plt.legend(loc='upper left')
        plt.xlabel('network output')
        plt.ylabel('events')
        plt.show()

def remake_hist(x,y):
        new_ind = 0
        ary = np.zeros((100000))
        for i in range(num_bins): #this value needs to make the total number of bins
                if y[i] == 0:
                        continue
                # print(new_ind,int(y[i]),x[i])
                ary[new_ind:new_ind+int(y[i])] = (x[i]+x[i+1])/2*np.ones((int(y[i])))
                # print((x[i]))
                new_ind = int(y[i])+new_ind
        ary = ary[0:new_ind]
        return ary

def wass_distances(d):
        new_ary = np.zeros(d.shape)
        for bn,y in enumerate(d):
                if y>4:
                        new_y = np.random.normal(y,scale=np.sqrt(y))
                        if new_y<0:
                                new_ary[bn] = 0
                        else:
                                new_ary[bn] = new_y 
                else:
                        if y==0:
                                y=1
                        new_y = np.random.poisson(y)
                        if new_y<0:
                                new_ary[bn] = 0
                        else:  
                                new_ary[bn] = new_y

        fin_ary = remake_hist(binss, new_ary)
        plot_each_hist(fin_ary, probsm)
        wass_s = wasserstein_distance(fin_ary,probs2)
        wass_n = wasserstein_distance(fin_ary,probn2)
        return  wass_s,wass_n #wass dist from input to meas exp CR

def main():
     
     fig = plt.figure()
     ax = fig.add_subplot(111)
     probs2 = model.predict(signal)[:,0]
     probsm = model.predict(msignal)[:,0]
     probn2 = model.predict(noise)[:,0]
     y_mCR,binss = np.histogram(probsm,bins=num_bins,range=(0,1))
     y_S,binss = np.histogram(probs2,bins=num_bins,range=(0,1))
     y_N,binss = np.histogram(probn2,bins=num_bins,range=(0,1))

     WD = np.zeros((num_iterations,3))
     for i in range(num_iterations):
             WD[i,0], WD[i,1] =  wass_distances(d=y_mCR)
             WD[i,2] = (WD[i,1]-WD[i,0])

     #print(max(WD[:,0]),min(WD[:,0]))
     plt.hist(WD[:,0],label='WD between simCR and measCR',histtype='step',lw=2)
     plt.hist(WD[:,1],label='WD between measN and measCR',histtype='step',lw=2)

     plt.legend(fontsize=18,loc='upper center')
     plt.xlabel('Wasserstein Distance (WD)', fontsize=20)
     plt.ylabel('Iterations', fontsize=20)
     plt.xticks(size=18)
     plt.yticks(size=18)
     plt.ylim(1,10**3)
     plt.yscale('log')
     plt.show()


if __name__== "__main__":
    main()
