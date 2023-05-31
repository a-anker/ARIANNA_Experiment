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
model = keras.models.load_model(model_path + 'trained_for_extract_weights_100samp_cnn_1layer5-10_mp10_s1_1output.h5')
iterations = 100 #how many events are looped over and plotted
chs = 1 #number of channels in the data

if s.ndim==2: # for data of shape (event_num, samples), this reformats it into 3 dimensions
    s = np.reshape(s, (s.shape[0], 1, s.shape[1]))
    n = np.reshape(n, (n.shape[0], 1, n.shape[1]))
d=s       
signal = np.reshape(s, (s.shape[0], s.shape[1], s.shape[2],1))
noise = np.reshape(n, (n.shape[0], n.shape[1], n.shape[2],1))


def plot_input_wf():
	#plot input waveform
	x = np.linspace(1, d.shape[2], d.shape[2])
	fig, axs = plt.subplots(2, sharex=True)
	
	axs[0].plot(x, s[evt_num], color='black') #######
	axs[0].set_ylabel('Voltage [V]',fontsize=19)
	axs[0].tick_params(axis='both', which='major', labelsize=16)
	axs[1].set_xlabel('sample',fontsize=19)
	axs[0].set_title(f'{label}: classif={float(model.predict(np.expand_dims(manual_data,0))[0]):.2f}',fontsize=15) #e for scifi notation, f for float

	# plt.show()

	
def main(evt_num,chs,label):

	data = noise

	manual_data = data[evt_num][0:chs,:,0] # (8,256,1)
	
	pred_data = data
	model = keras.models.load_model(f'/Users/astrid/Desktop/st61_deeplearning/h5_models/trained_CNN_1l-10-8-10-do0.5_fltn_sigm_valloss_p4_allampl37-74k_5-10ksig_shuff_monitortraining_1.h5')
	# model = keras.models.load_model(f'/Users/astrid/Desktop/st61_deeplearning/h5_models/trained_CNN_1l-10-2-10-do0.5_fltn_sigm_valloss_p4_allampl37-74k_5-10ksig_shuff_monitortraining_2,3dlpda_4.h5')
	# model = keras.models.load_model(f'/Users/astrid/Desktop/st61_deeplearning/h5_models/trained_CNN_1l-10-8-20_do0.5_fltn_sigm_50k_simnoise_0-5k_simsig_0.h5')

	# model.summary()
	
	print(model.predict(pred_data[:,0:chs])[evt_num])
	

	#______________________________________
	#plot input waveform
	x = np.linspace(1, 256, 256)
	fig, axs = plt.subplots(chs, sharex=True)
	plt.xlim(1,256)
	for i in range(chs):
		axs[i].plot(x, data[evt_num][i,:,0])
		# axs[i].set(ylim=(-0.1,0.1))
	axs[chs-1].set(xlabel='sample')
	axs[0].set(ylabel='V')
	axs[0].set(title=f'{label} {evt_num}, classif={model.predict(pred_data[:,0:chs])[evt_num]}')
	plt.show()
	#______________________________________

	weights = model.get_weights()[0]  # (8, 10, 1, 10)
	weights = weights[:,:,0,:]  # shape is (fil size1,fil size2, n_filters)
	filter_size1,filter_size2, n_filters = weights.shape
	w_bias = model.get_weights()[1]  # (10,)
	fc = model.get_weights()[2]  # (2470x1)
	fc_bias = model.get_weights()[3]  # (1,)

	

	shifts = 256 - filter_size2 + 1
	second_dim = int(manual_data.shape[1] / filter_size2) - 1

	# convolution
	s1 = np.zeros((shifts, n_filters))
	for i in range(n_filters):
	    weight = weights[:, :,i]

	    for j in range(shifts):
	        
	        s1[j][i] = np.sum(manual_data[:,j:j + filter_size2] * weight) + w_bias[i]
	#______________________________________
	#plot convolutions
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
		# axs[i].set(ylim=(-2.75,2.75))

	axs[chs-1].set(xlabel='sample')
	axs[0].set(ylabel='V')
	axs[0].set(title=f'{label} {evt_num} convolution weights 10 {chs}x10 filter')
	plt.show()
	#______________________________________


	#______________________________________
	# plot the 10 convolution filter waveforms after convoluting the 10 8x10 filters and adding bias
	s1[s1 <= 0] = 0 #uncomment to plot the same thing below with relu activation applied
	x3 = np.linspace(1, shifts, shifts)
	fig3, axs3 = plt.subplots(10, sharex=True)
	for i in range(10):
		axs3[i].plot(x3, s1[:,i])
		# axs3[i].set(ylim=(-1,25))

	axs3[0].set(title=f'{label} {evt_num}: relu activation applied ')
	# axs3[0].set(title='noise: result of convolution weights, 10 filters ')
	plt.show()
	#______________________________________

	# relu activation
	s1[s1 <= 0] = 0


	# # max pooling
	# s2 = np.zeros((second_dim, n_filters))
	# for i in range(n_filters):
	#     for k in range(second_dim):
	#         j = 10 * k
	#         s2[k][i] = max(s1[j:j + 10, i])  # s2 is 23x10

	# flatten and fc layer
	# s2 = s2.T

	s3 = s1.flatten()

	s4 = np.matmul(fc[:, 0], s3) + fc_bias[0]
	print(s4)

	# #______________________________________
	#plots the step with flattening applied and also the fc weights plot together
	plt.plot(np.linspace(1,s3.shape[0],s3.shape[0]),fc[:, 0],color='red',label='fc weights')
	plt.plot(np.linspace(1,s3.shape[0],s3.shape[0]),s3,color='blue',label='flattening applied')
	plt.title(f'{label} {evt_num}: these multiplied together, ~{round(fc_bias[0])} bias added result: {round(s4,3)}')
	# plt.title(f'noise: flattening applied')
	# plt.ylim(-.5,5.5)
	# plt.legend()
	plt.show()
	# #______________________________________	
	
	
	
	

def dl_stepbystep(data, evt_num, label):
	manual_data = data[evt_num] # (2048,1,1)
	
	
	
	weights = model.get_weights()[0]  # (10, 1, 10)
	weights = weights[:,:,0,:]  # shape is (fil size1,fil size2, n_filters)
	filter_size1,filter_size2, n_filters = weights.shape
	w_bias = model.get_weights()[1]  # (10,)
	fc = model.get_weights()[2]  # (2039x1)
	fc_bias = model.get_weights()[3]  # (1,)

	# fig, axs = plt.subplots(5, sharex=True)
	# for i in range(n_filters):
	# 	axs[i].plot(np.linspace(1,256,256), fc[i*256:i*256+256, 0],color='red')
	# plt.show()
	
	
	shifts = n.shape[1]*1 - filter_size1 + 1
	second_dim = int(manual_data.shape[0] / filter_size1) - 1

	# convolution
	s1 = np.zeros((shifts, n_filters))
	for i in range(n_filters):
	    weight = weights[:, 0,i]
	    for j in range(shifts):
	       
	        s1[j][i] = np.sum(manual_data[j:j + filter_size1,0,0] * weight) + w_bias[i]
	
	# #______________________________________
	#plot convolutions
	# x = np.linspace(5, 15, 10)
	# fig, axs = plt.subplots(1, sharex=True)
	# plt.xlim(1,256)
	
	# for i in range(10):
	# 	axs.plot(x+(i*100),weights[:,0,i])

	c = ['blue','green','grey','purple', 'orange']
	x = np.linspace(5, 15, 10)
	# fig, axs = plt.subplots(10, sharex=True)
	for i in range(n_filters):
		axs[1].plot(x+20*i, weights[:,0,i],color=c[i])
		axs[1].set_ylabel('5 filters',fontsize=19)
	plt.tick_params(axis='both', which='major', labelsize=16)
	# plt.show()

	# axs[chs-1].set(xlabel='sample')
	# axs[0].set(ylabel='V')
	# axs.set(title=f'convolution weights 10 10x1 filters')

	# plt.show()
	#______________________________________


	# #______________________________________
	# # plot the 10 convolution filter waveforms after convoluting the 10 8x10 filters and adding bias
	s1[s1 <= 0] = 0 #uncomment to plot the same thing below with relu activation applied
	x3 = np.linspace(1, shifts, shifts)
	fig3, axs3 = plt.subplots(n_filters, sharex=True)
	for i in range(n_filters):
		axs3[i].plot(x3, s1[:,i],label='after convolution') # color=c[i] use when plotting initial but not for final comparison
		axs3[i].set_ylabel(f'filter {i}',fontsize=19)
		axs3[i].tick_params(axis='both', which='major', labelsize=16)
		# axs3[i].plot(x3, s1[:,i]*fc[i::5, 0],color='red')
		
	axs3[n_filters-1].set_xlabel('sample',fontsize=19)
	axs3[0].set_title(f'{label}: after convolution + relu activation',fontsize=15)
	# axs3[0].set(title=f'{label} {evt_num}: relu activation applied ')
	# # axs3[0].set(title='noise: result of convolution weights, 10 filters ')
	# plt.show()
	# #______________________________________


	# max pooling__
	s2 = np.zeros((second_dim, n_filters))
	for i in range(n_filters):
	    for k in range(second_dim):
	        j = 10 * k
	        s2[k][i] = max(s1[j:j + 10, i])  # s2 is 23x10

	fig4, axs4 = plt.subplots(n_filters, sharex=True)
	for i in range(n_filters):
		axs4[i].plot(np.linspace(1,9,9), s2[:,i]*10,label='after max pool (x10)') # color=c[i] use when plotting initial but not for final comparison
		axs4[i].set_ylabel(f'filter {i}',fontsize=19)
		axs4[i].tick_params(axis='both', which='major', labelsize=16)
	
	axs4[n_filters-1].set_xlabel('sample',fontsize=19)
	# plt.show()
	#____________________________________________
	# flatten and fc layer
	# s2 = s2.T
	s3 = s2.flatten()
	print(s3,s2)

	#__
	# s1 = np.swapaxes(s1,0,1)
	# print(s1.shape)
	# s3 = s1.flatten()
	## print(fc[:, 0].shape, s3.shape)
	## print(fc_bias[0])

	s4 = np.matmul(fc[:, 0], s3) + fc_bias[0]
	print(fc_bias[0])

	# _________________
	
	# _________________
	# plt.show()


	# _________________
	#plots the step with flattening applied and also the fc weights plot together
	print(fc[i::5, 0].shape[0])
	for i in range(5):
		axs4[i].plot(np.linspace(1,fc[i::5, 0].shape[0],fc[i::5, 0].shape[0]),fc[i::5, 0],color='red',linestyle='dashed',label='fc weights')
	# plt.plot(np.linspace(1,s3.shape[0],s3.shape[0]),fc[:, 0],color='red',label='fc weights')
	# plt.plot(np.linspace(1,s3.shape[0],s3.shape[0]),s3,color='blue',label='flattening applied')
	axs4[0].set_title(f'{label}: plots multiplied + {round(fc_bias[0])} = {round(s4,3)} before sigmoid activation',fontsize=15)
	
	# plt.title(f'noise: flattening applied')
	# plt.ylim(-.5,5.5)
	# plt.yscale('log')
	plt.legend(loc='lower right',fontsize=15)
	plt.show()
	# #______________________________________

	print(s4)
	# sigmoid activation
	smoid = 1 / (1 + np.exp(-s4))
	print('sigmoid',smoid)


for i in range(iterations):
	dl_stepbystep(data=signal,i,label='neutrino signal') #change signal value to noise to plot noise data
	
	
if __name__== "__main__":
    main()
