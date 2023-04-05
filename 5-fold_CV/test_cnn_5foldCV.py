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
from sklearn.model_selection import KFold, train_test_split


def data(k,d_size):
        path = "/Volumes/External/arianna_data/"
        s1 = np.load(os.path.join(path, "trimmed100_data_signal_3.6SNR_1ch_0000.npy"))
        s2 = np.load(os.path.join(path, "trimmed100_data_signal_3.6SNR_1ch_0001.npy"))
        signal = np.vstack((s1,s2))
        sshuf = np.load('/Volumes/External/arianna_data/random_shuff_index_for_trimmed100_signal_0and1_ABC.npy')
        signal = signal[sshuf]

        noise = np.zeros((200000, 100), dtype=np.float32)
        for i in range(2):
          noise[(i) * 100000:(i + 1) * 100000] = np.load(os.path.join(path, f"trimmed100_data_noise_3.6SNR_1ch_{i:04d}.npy")).astype(np.float32)
        nshuf = np.load('/Volumes/External/arianna_data/random_shuff_index_for_trimmed100_noise_0and1_ABC.npy')
        noise = noise[nshuf]

        s_tot = np.zeros((k,d_size,100))
        n_tot = np.zeros((k,d_size,100))
        for i in range(k):
                s_tot[i] = signal[(i)*d_size : (i + 1)*d_size]
        for i in range(k):
                n_tot[i] = noise[(i)*d_size : (i + 1)*d_size]
        folds = list(KFold(n_splits=5).split(s_tot,n_tot))
        return folds, s_tot, n_tot

Folds, signal, noise = data(5,24000)

signal = np.reshape(signal, (signal.shape[0], signal.shape[1],signal.shape[2],1,1))
noise = np.reshape(noise, (noise.shape[0], noise.shape[1],noise.shape[2],1,1))

print(signal.shape,noise.shape)
fig = plt.figure()
ax = fig.add_subplot(111)
count=0
c = ['blue','red','black','green','grey']

fig = plt.figure()
ax = fig.add_subplot(111)

models = []
for i in range(5):
        model = keras.models.load_model(f'/Volumes/External/ML_paper/5fold_cross_val/trainings_pat8/trained_CNN_100samp_1L5-10_pat8_lr0.001_do0.5_5fold_24k_each_omitted_3rdtrial_datashuffdABC_[{i}].h5')
        n_pred = model.predict(noise[i])  
        s_pred = model.predict(signal[i])  
        ax.hist(n_pred, bins=20, range=(0, 1), histtype='step',color=c[i], linestyle='dashed', linewidth=1.5,label=f'n{i}') 
        ax.hist(s_pred, bins=20, range=(0, 1), histtype='step',color=c[i], linestyle='solid',linewidth=1.5,label=f's{i}')


plt.xlabel('network output', fontsize=17)
plt.ylabel('normalized events', fontsize=17)
plt.xticks(size=15)
plt.yticks(size=15)
# plt.yscale('log')
# plt.title('')
handles, labels = ax.get_legend_handles_labels()
new_handles = [Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle()) for h in handles]
plt.legend(loc='upper center', handles=new_handles, labels=labels, fontsize=14)
plt.show()
