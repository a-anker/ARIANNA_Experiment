import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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
     
num_bins=100

noise = np.load(f'/Users/astrid/Desktop/st52_deeplearning/data/stn52_2019_cr_0123568_all_shuffledBB.npy') #[42000:,4:]
n_mask_blw60 = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stn52_2019_cr_0123568_all_shuffledBB_all_amp_below_60mV_True.npy')
# n_mask_blw60 = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stn52_2019_cr_0123568_all_shuffledBB_all_amp_below_50mV_True.npy')

mask = n_mask_blw60>0
noise = noise[mask]
noise = noise[20000:,4:]

signal = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stn52_sim_cr.npy')
# lst = np.load('/Users/astrid/Desktop/st52_deeplearning/data/mask_evt_list_orderingCC_signal_stn52_sim_cr.npy')
# s = signal[lst]
signal = signal[5000:,4:]

msignal = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stn52_2019_cr_candidates.npy')[:,4:]


noise = np.expand_dims(noise, axis=-1)
# noises = np.expand_dims(noises, axis=-1)
fig = plt.figure()
ax = fig.add_subplot(111)

signal = np.reshape(signal, (signal.shape[0], signal.shape[1],signal.shape[2],1))
msignal = np.reshape(msignal, (msignal.shape[0], msignal.shape[1],msignal.shape[2],1))
noise = np.reshape(noise, (noise.shape[0], noise.shape[1],noise.shape[2],1))


model = keras.models.load_model(f'/Users/astrid/Desktop/st52_deeplearning/h5s/trained_CNN_1l-10-4-10_fltn_sigm_valloss_p4_noise_lowampl_lt60mV_0-20k_CRsignal_no-weight_0-5k_repeat_sig4x_shuff_monitortraining_4upLPDAs_0.h5')

probs2 = model.predict(signal)[:,0]
probsm = model.predict(msignal)[:,0]
probn2 = model.predict(noise)[:,0]


y_mCR,binss = np.histogram(probsm,bins=num_bins,range=(0,1))
y_S,binss = np.histogram(probs2,bins=num_bins,range=(0,1))
y_N,binss = np.histogram(probn2,bins=num_bins,range=(0,1))


# print(wasserstein_distance(probs2,probsm)) # 0.1875437593462966
# print(wasserstein_distance(probn2,probsm)) # 0.7954946465578849
# print(wasserstein_distance(probn2,probs2)) # 0.9830350246816826

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


def main(d):
        new_ary = np.zeros(d.shape)
        for bn,y in enumerate(d):
                if y>4:
                        new_y = np.random.normal(y,scale=np.sqrt(y))
                        if new_y<0:
                                new_ary[bn] = 0
                        else:
                                new_ary[bn] = new_y 
                        # print(y,new_ary[bn])
                
                else:
                        if y==0:
                                y=1
                        new_y = np.random.poisson(y)
                        if new_y<0:
                                new_ary[bn] = 0
                        else:  
                                new_ary[bn] = new_y
                        # print(y,new_ary[bn])

        fin_ary = remake_hist(binss,new_ary)
        # for i in range(1000):
        #         print(fin_ary[i],np.sort(probs2)[i])
        # print(fin_ary)
        plot_each_hist(fin_ary,probsm)
        wass_s = wasserstein_distance(fin_ary,probs2)
        wass_n = wasserstein_distance(fin_ary,probn2)
        return  wass_s,wass_n #wass dist from input to meas exp CR

WD = np.zeros((1000,3))
for i in range(1000):
        WD[i,0],WD[i,1] =  main(d=y_mCR)
        WD[i,2] = (WD[i,1]-WD[i,0])

# ary = np.zeros((10000))
# for x in [1,2,3,4]:
#         for i in range(10000):
#                 ary[i] = np.random.poisson(x)
#         # print(np.std(ary))
#         plt.hist(ary,bins=12,histtype='step',linewidth=1.5,label=f'lambda = {x}')

# plt.xlabel('mean')
# plt.ylabel('counts')
# plt.legend()
# plt.show()


print(max(WD[:,0]),min(WD[:,0]))
plt.hist(WD[:,0],label='WD between simCR and measCR',histtype='step',lw=2)
plt.hist(WD[:,1],label='WD between measN and measCR',histtype='step',lw=2)


    
# plt.hist(WD[:,2],label='difference in distances')
plt.legend(fontsize=18,loc='upper center')
plt.xlabel('Wasserstein Distance (WD)', fontsize=20)
plt.ylabel('Iterations', fontsize=20)
plt.xticks(size=18)
plt.yticks(size=18)
plt.ylim(1,10**3)
plt.yscale('log')
plt.show()







