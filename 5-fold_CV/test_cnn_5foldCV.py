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
from sklearn.model_selection import KFold, train_test_split

######################################################################################################################
"""
This script takes in previously trained h5 files using 5-fold CV to plot the network output. 
This script should be used with train_cnn_5foldCV.py to get the trained models with the correct training/testing data sets.

data() creates the data set folds and the main() function takes the CV folds, reshapes the data, 
and plots the network output of each fold for signal and noise data toegther
"""
######################################################################################################################
PathToARIANNA = os.environ['ARIANNA_Experiment']
n = np.load(PathToARIANNAData + '/data/noise.npy') #input a subset of the data here so that you can validate on the other set
s = np.load(PathToARIANNAData + '/data/signal.npy') #make sure the signal and noise subset of data are the same size
#actual model name is specified below when loading the model
model_path = PathToARIANNAData + '/models_h5_files/'
CV_num = 5 #this value can be changed to have the data split into more or less cross validiation groups
CV_size = 24000 #this gives the amount of data in each fold of the CV
 
if s.ndim==2: # for data of shape (event_num, samples), this reformats it into 3 dimensions
    s = np.reshape(s, (s.shape[0], 1, s.shape[1]))
    n = np.reshape(n, (n.shape[0], 1, n.shape[1]))
        
        
def data(s, n, k, d_size):
        s_tot = np.zeros((k,d_size,100))
        n_tot = np.zeros((k,d_size,100))
        for i in range(k):
                s_tot[i] = s[(i)*d_size : (i + 1)*d_size]
        for i in range(k):
                n_tot[i] = n[(i)*d_size : (i + 1)*d_size]
        folds = list(KFold(n_splits=5).split(s_tot,n_tot))
        return folds, s_tot, n_tot


def main():    
        Folds, signal, noise = data(s, n, CV_num, CV_size)
        signal = np.reshape(signal, (signal.shape[0], signal.shape[1],signal.shape[2],1,1))
        noise = np.reshape(noise, (noise.shape[0], noise.shape[1],noise.shape[2],1,1))
        print(signal.shape,noise.shape)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        count=0
        c = ['blue','red','black','green','grey', 'yellow','pink','orange','purple','cyan']

        for i in range(CV_num):
                model = keras.models.load_model(model_path + f'trained_CNN_100samp_1L5-10_pat8_lr0.001_do0.5_5fold_24k_each_omitted_3rdtrial_datashuffdABC_[{i}].h5')
                n_pred = model.predict(noise[i])  
                s_pred = model.predict(signal[i])  
                ax.hist(n_pred, bins=20, range=(0, 1), histtype='step',color=c[i], linestyle='dashed', linewidth=1.5,label=f'n{i}') 
                ax.hist(s_pred, bins=20, range=(0, 1), histtype='step',color=c[i], linestyle='solid', linewidth=1.5,label=f's{i}')

        plt.xlabel('network output', fontsize=17)
        plt.ylabel('normalized events', fontsize=17)
        plt.xticks(size=15)
        plt.yticks(size=15)
        # plt.yscale('log')
        handles, labels = ax.get_legend_handles_labels()
        new_handles = [Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle()) for h in handles]
        plt.legend(loc='upper center', handles=new_handles, labels=labels, fontsize=14)
        plt.show()


if __name__== "__main__":
    main()
