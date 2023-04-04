import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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



path = "/Volumes/External/arianna_data/"
ch = 1


s = np.load(os.path.join(path, "trimmed100_data_signal_3.6SNR_1ch_0000.npy"))

n = np.zeros((600000, 100), dtype=np.float32)
for i in range(6):
  n[(i) * 100000:(i + 1) * 100000] = np.load(os.path.join(path, f"trimmed100_data_noise_3.6SNR_1ch_{i:04d}.npy")).astype(np.float32)


# #______________
# # artifact data
# model = keras.models.load_model(os.path.join("/Volumes/External/ML_paper/a_efficiency/trained_CNN_100samp_1L5-10_0-n600k_s100k_addartifact-noise-50mV-to-1st-10samps.h5"))
# n[:,0:10] = 0.05*np.ones(10) ###############################################################add a pulse artifact for 1st 10 samples at 50mV
# #_______________

ch=1
noise = np.reshape(n, (n.shape[0], n.shape[1] * ch,1,1))
signal = np.reshape(s, (s.shape[0], s.shape[1] * ch,1,1))
# print(signal.shape, noise.shape)



def main(evt_num,chs,label):

	data = signal ######change this one and the one on line 55
	manual_data = data[evt_num] # (2048,1,1)
	
	pred_data = data
	model = keras.models.load_model(os.path.join("/Volumes/External/ML_paper/a_efficiency",'trained_for_extract_weights_100samp_cnn_1layer5-10_mp10_s1_1output.h5'))
	model.summary()
	# print(model.predict(signal))

	#______________________________________
	#plot input waveform
	x = np.linspace(1, 100, 100)
	fig, axs = plt.subplots(2, sharex=True)
	
	axs[0].plot(x, s[evt_num], color='black') #######
	# plt.xticks(size=16)
	# plt.yticks(size=16)
	axs[0].set_ylabel('Voltage [V]',fontsize=19)
	axs[0].tick_params(axis='both', which='major', labelsize=16)
	axs[1].set_xlabel('sample',fontsize=19)
	# axs[1].yticks(size=16)
	axs[0].set_title(f'{label}: classif={float(model.predict(np.expand_dims(manual_data,0))[0]):.2f}',fontsize=15) #e for scifi notation, f for float

	# plt.show()
	#______________________________________

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
# for i in range(37265):
# 	main(i)

# main(11)
for i in range(50):
	main(i,chs=1,label='neutrino signal')
