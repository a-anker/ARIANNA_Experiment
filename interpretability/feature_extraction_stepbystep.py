import os
import glob
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
import time

PathToARIANNA = os.environ['ARIANNA_analysis']
######################################################################################################################
"""
This script plots the intermediate stages of a 1 layer CNN with x amount of kernels with y size. The comments on shape are for an eight channel 1D convolution with 10 10x1 kernels.
"""
######################################################################################################################
PathToARIANNA = os.environ['ARIANNA_Experiment']
n = np.load(PathToARIANNAData + '/data/noise.npy') #input a subset of the data here so that you can validate on the other set
s = np.load(PathToARIANNAData + '/data/signal.npy') #make sure the signal and noise subset of data are the same size
model_path = PathToARIANNAData + '/models_h5_files/'
model = keras.models.load_model(model_path + 'trained_model.h5')
iterations = 100 #how many events are looped over and plotted
chs = 1 #number of channels in the data

if s.ndim==2: # for data of shape (event_num, samples), this reformats it into 3 dimensions
    s = np.reshape(s, (s.shape[0], 1, s.shape[1]))
    n = np.reshape(n, (n.shape[0], 1, n.shape[1]))
d=s       
signal = np.reshape(s, (s.shape[0], s.shape[1], s.shape[2],1))
noise = np.reshape(n, (n.shape[0], n.shape[1], n.shape[2],1))


def plot_input_wf():
	x = np.linspace(1, 256, 256)
	fig, axs = plt.subplots(chs, sharex=True)
	plt.xlim(1,256)
	for i in range(chs):
		axs[i].plot(x, data[evt_num][i,:,0])
	axs[chs-1].set(xlabel='sample')
	axs[0].set(ylabel='V')
	axs[0].set(title=f'{label} {evt_num}, classif={model.predict(pred_data[:,0:chs])[evt_num]}')
	plt.show()

def dl_stepbystep(data, evt_num, label):
	manual_data = data[evt_num][0:chs,:,0] # (8,256,1)
	pred_data = data
	weights = model.get_weights()[0]  # (8, 10, 1, 10)
	weights = weights[:,:,0,:]  # shape is (fil size1,fil size2, n_filters)
	filter_size1,filter_size2, n_filters = weights.shape
	w_bias = model.get_weights()[1]  # (10,)
	fc = model.get_weights()[2]  # (2470x1)
	fc_bias = model.get_weights()[3]  # (1,)
	shifts = 256 - filter_size2 + 1
	second_dim = int(manual_data.shape[1] / filter_size2) - 1

	plot_input_wf()
	
	# _____________________________________________________________________________________
	# convolution__________________________________________________________________________
	s1 = np.zeros((shifts, n_filters))
	for i in range(n_filters):
	    weight = weights[:, :,i]
	    for j in range(shifts):
	        s1[j][i] = np.sum(manual_data[:,j:j + filter_size2] * weight) + w_bias[i]

	# plot convolutions
	x = np.linspace(5, 15, 10)
	fig, axs = plt.subplots(chs, sharex=True)
	plt.xlim(1,256)
	for i in range(chs):
		axs[i].plot(x, weights[i,:,0])
		axs[i].plot(x+25, weights[i,:,1])
		axs[i].plot(x+50, weights[i,:,2])
		axs[i].plot(x+75, weights[i,:,3])
		axs[i].plot(x+100, weights[i,:,4])
		axs[i].plot(x+125, weights[i,:,5])
		axs[i].plot(x+150, weights[i,:,6])
		axs[i].plot(x+175, weights[i,:,7])
		axs[i].plot(x+200, weights[i,:,8])
		axs[i].plot(x+225, weights[i,:,9])
		
	axs[chs-1].set(xlabel='sample')
	axs[0].set(ylabel='V')
	axs[0].set(title=f'{label} {evt_num} convolution weights 10 {chs}x10 filter')
	plt.show()
	# _____________________________________________________________________________________
	
	
	# _____________________________________________________________________________________
	# plot the 10 CNN filters after conv the 10 8x10 filters and adding bias_______________
	#s1[s1 <= 0] = 0 #uncomment to plot the same thing below with relu activation applied
	x3 = np.linspace(1, shifts, shifts)
	fig3, axs3 = plt.subplots(10, sharex=True)
	for i in range(10):
		axs3[i].plot(x3, s1[:,i])
	axs3[0].set(title='noise: result of convolution weights, 10 filters ')
	plt.show()
	# _____________________________________________________________________________________

	
	# _____________________________________________________________________________________
	# relu activation
	s1[s1 <= 0] = 0

	# # max pooling (if used)
	# s2 = np.zeros((second_dim, n_filters))
	# for i in range(n_filters):
	#     for k in range(second_dim):
	#         j = 10 * k
	#         s2[k][i] = max(s1[j:j + 10, i])  # s2 is 23x10

	# flatten and fc layer
	s3 = s1.flatten()

	# bias added
	s4 = np.matmul(fc[:, 0], s3) + fc_bias[0]
	
	# sigmoid activation
	smoid = 1 / (1 + np.exp(-s4))
	print('network output: ',smoid)
	# _____________________________________________________________________________________
	
	
	# _____________________________________________________________________________________
	# plots the step with flattening applied and also the fc weights plot together_________
	plt.plot(np.linspace(1,s3.shape[0],s3.shape[0]),fc[:, 0],color='red',label='fc weights')
	plt.plot(np.linspace(1,s3.shape[0],s3.shape[0]),s3,color='blue',label='flattening applied')
	plt.title(f'{label} {evt_num}: these multiplied together, ~{round(fc_bias[0])} bias added result: {round(s4,3)}')
	plt.show()
	# _____________________________________________________________________________________
	

for i in range(iterations):
	dl_stepbystep(data=signal,i,label='neutrino signal') #change signal value to noise to plot noise data
	
if __name__== "__main__":
    main()
