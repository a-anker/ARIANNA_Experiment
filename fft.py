from NuRadioReco.utilities import units,  fft
import NuRadioReco.detector.detector
import NuRadioReco.modules.io.NuRadioRecoio as NuRadioRecoio
import numpy as np
import matplotlib.pyplot as plt
import time
import NuRadioReco.modules.channelResampler
import NuRadioReco.modules.channelTemplateCorrelation
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.io.eventWriter
import datetime
import NuRadioReco.detector
import NuRadioReco.utilities
import NuRadioReco.framework.parameters
import NuRadioReco.modules.io.NuRadioRecoio
import NuRadioReco.modules.channelBandPassFilter
from NuRadioReco.utilities import units
import NuRadioReco.modules.channelStopFilter
import keras


######################################################################################################################
"""
This script takes in a .nur file, extracts the fft values, and saves them to a numpy array along with the frequency values for all waveforms. 
Then the script plots the average FFT value for each channel and compares simulated to measure FFT.

All variables listed at the beginning that start with the word 'input' are input data files that are needed to specify from the /data folder in this github.
The other variables are names for files that will be saved in this script and input data specific parameters that the user will specify

"""
######################################################################################################################
PathToARIANNA = os.environ['ARIANNA_Experiment'] 
PathToARIANNAData = PathToARIANNA + '/data'
fft_file = 'stn52_fft_all.npy' #this will be the saved array of all fft values for all channels
input_nur_file = 'stn52_sim_cr.nur' #this is specified by the user and needs to be the .nur file that the ffts will be extracted from
input_sim_avg_fft = 'stop_bandpass_incl_stn52_sim_cr_all9629_avg_fft.npy' #this is specified by the user and contains the average fft per channel for simulated data. Can be created from the first part of this script and save tot_sum variable instead
num_chan = 4 #number of channels in the .nur data
num_samp = 257 
station_num = 52 #specify the station id number here

def get_npy_and_plot(files,num_chan):
	
	#get numpy array of average fft and saves it
	data = NuRadioRecoio.NuRadioRecoio(files)
	ary = np.zeros((100000,num_chan,num_samp)) #creates a larger array than needed (100k events) and then it will be cut after all data is run through
	freqs = np.zeros((num_samp))
	count = 0 #keeps track of how many events are in this file
	for i, evt in enumerate(data.get_events()):
	    stn = evt.get_station(station_num)
	    count=i
	    for j, ch in enumerate(stn.iter_channels()):
	        freqs = ch.get_frequencies()/units.MHz
	        ary[i,j] = abs(ch.get_frequency_spectrum())
		
	ary = ary[0:count] #gets rid of excess zeros in array
	print(ary.shape) 
	np.save(os.path.join(PathToARIANNAData, fft_file), ary) #saves all ffts for all channels
	np.save(os.path.join(PathToARIANNAData, 'frequencies.npy'), freqs) #saves frequency values that remain constant for all waveforms (the x values)
	
	
	tot_sum = ary.mean(axis=0) #takes the mean values of each FFT bin to give one single average FFT waveform per channel
	
	#plot fft
	plot_dims = np.array([(1,1),(2,1),(3,1),(4,1),(4,2),(4,2),(4,2),(4,2)]
	plot_dims_all = np.array([(1,0),(2,0),(3,0),(4,0),(1,1),(2,1),(3,1),(4,1)]
	fig = plt.figure(figsize=(7,5))
	axes = fig.subplots(nrows=plot_dims[num_chan,0], ncols=plot_dims[num_chan,1], sharex=True, sharey=True)
	sim_avg_fft = np.load(os.path.join(PathToARIANNAData, input_sim_avg_fft)) #this is the averaged simulated fft per channel

	for k in range(num_chan):
		ax = axes[plot_dims_all[k,0], plot_dims_all[k,1]]
		ax.set_title(f'ch{k}')
		ax.plot(freqs, tot_sum[k], label='avg experimental')
		ax.set_ylabel('normalized P')
		ax.set_xlim([0,512])
		ax.plot(freqs, sim_avg_fft[k]/(max(sim_avg_fft[k])-min(sim_avg_fft[k])), label='avg simulated') #comment this out to only plot the singular fft
				 
			 
	axes[4,0].set_xlabel('Frequency [MHz]')
	plt.legend()
	plt.show()
	plt.close(fig)
				 
get_npy_and_plot(files=os.path.join(PathToARIANNAData, input_nur_file), num_chan)








