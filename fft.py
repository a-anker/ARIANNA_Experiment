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


# # # Fast-Fourier-transform (fft): give the frequency spectrum for a given time-domain wave. 

# #_________________this script eliminates the stop and adds bandpass filter, then saves the files as the output_name variable
# # files = ['/Users/astrid/Desktop/st52_deeplearning/data/stn52_2019_cr_candidates.nur']
# files = ['/Users/astrid/Desktop/st52_deeplearning/data/stn52_sim_cr_part02.nur']
# data = NuRadioRecoio.NuRadioRecoio(files)

# output_name='stop_bandpass_incl_stn52_sim_cr_part02.nur'
# # #load all modules
# channelResampler=NuRadioReco.modules.channelResampler.channelResampler()
# channelResampler.begin(debug=False)
# # channelTemplateCorrelation=NuRadioReco.modules.channelTemplateCorrelation.channelTemplateCorrelation(template_directory=template_directory)
# channelSignalReconstructor=NuRadioReco.modules.channelSignalReconstructor.channelSignalReconstructor()
# det=NuRadioReco.detector.detector.Detector()
# channelBandPassFilter = NuRadioReco.modules.channelBandPassFilter.channelBandPassFilter()
# channelBandPassFilter.begin()
# channelStopFilter = NuRadioReco.modules.channelStopFilter.channelStopFilter()
# eventWriter=NuRadioReco.modules.io.eventWriter.eventWriter()
# eventWriter.begin(output_name,max_file_size=102400)  

# for i,evt in enumerate(data.get_events()):
#     stn = evt.get_station(52)
#     station_time=stn.get_station_time()
#     det.update(station_time)
#     channelStopFilter.run(evt,stn,det)
#     channelBandPassFilter.run(evt, stn, det, passband=[80 * units.MHz, 500 * units.MHz], filter_type='butter', order = 10)
#     eventWriter.run(evt)
#     # for j,ch in enumerate(stn.iter_channels()):
# eventWriter.end()	
# # _________________


#_________________ plot average fft of tagged CR data

def fft_plot(files):
	
	data = NuRadioRecoio.NuRadioRecoio(files)
	print(data)

	ary = np.zeros((100000,8,257))
	freqs = np.zeros((257))
	count = 0
	for i,evt in enumerate(data.get_events()):
	    stn = evt.get_station(52)
	    count=i
	    for j,ch in enumerate(stn.iter_channels()):

	        frequencies = ch.get_frequencies()/units.MHz
	        freqs = frequencies

	        trace_fft = abs(ch.get_frequency_spectrum())
	        # print(len(trace_fft))
	        ary[i,j] = trace_fft

	ary = ary[0:count]
	print(ary.shape)
	tot_sum = ary.mean(axis=0)

	fig = plt.figure(figsize=(7, 5))
	axes = fig.subplots(nrows=4, ncols=2, sharex=True, sharey=True)


	np.save('/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_stn52_sim_cr_all9629_avg_fft.npy',tot_sum)
	for i in range(8):
	    if i == 0:
	        ax = axes[2, 0]
	        ax.set_title('ch0')
	        # ax.set_ylabel('Amplitude (mV)')
	    elif i == 1:
	        ax = axes[2, 1]
	        ax.set_title('ch1')
	    elif i == 2:
	        ax = axes[3, 0]
	        ax.set_title('ch2')
	        # ax.set_ylabel('Amplitude (mV)')
	    elif i == 3:
	        ax = axes[3, 1]
	        ax.set_title('ch3')
	    elif i == 4:
	        ax = axes[0, 0]
	        ax.set_title('ch4')
	        # ax.set_ylabel('Amplitude (mV)')
	    elif i == 5:
	        ax = axes[0, 1]
	        ax.set_title('ch5')
	    elif i == 6:
	        ax = axes[1, 0]
	        ax.set_title('ch6')
	        # ax.set_ylabel('Amplitude (mV)')
	        # ax.set_xlabel('Time (ns)')
	    elif i == 7:
	        ax = axes[1, 1]
	        ax.set_title('ch7')
	    #     ax.set_xlabel('Time (ns)')
	    # ax.set_title(f'channel{i}')
	  
	    ax.plot(freqs, tot_sum[i])
	    # ax.set_title(
	    #     f'stn {station_id} run {run_num} evt {evt_id} time:{stn_time}')

	plt.show()
	plt.close(fig)

# fft_plot('stop_bandpass_incl_station_52.nur')
# fft_plot(['/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_stn52_sim_cr.nur','/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_stn52_sim_cr_part02.nur'])

# '/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_stn52_sim_cr_part02.nur'
#_________________


#_________________create npy arrays from tagged cr data for ffts and votage traces

def save_fft(files):
	
	data = NuRadioRecoio.NuRadioRecoio(files)
	ary = np.zeros((85,8,257))
	f = np.zeros((257))
	freqs = np.zeros((257))
	for i,evt in enumerate(data.get_events()):
	    stn = evt.get_station(52)
	    for j,ch in enumerate(stn.iter_channels()):

	        frequencies = ch.get_frequencies()/units.MHz
	        freqs = frequencies
	        f=freqs
	        trace_fft = abs(ch.get_frequency_spectrum())
	        ary[i,j] = trace_fft

	        
	# np.save('/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_station_52_fft.npy',ary)
	np.save('/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_station_52_freqs.npy',f)

# save_fft('/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_station_52.nur')


def save_wfs(files):
	wfs = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stn52_2019_cr_candidates.npy')

	data = NuRadioRecoio.NuRadioRecoio(files)
	ary_wfs = np.zeros((85,8,512))
	for i,evt in enumerate(data.get_events()):
	    stn = evt.get_station(52)
	    for j,ch in enumerate(stn.iter_channels()):
	        wf = ch.get_trace()
	        plt.plot(np.linspace(1,512,512),wf)
	        plt.plot(np.linspace(1,256,256),wfs[i,j])
	        plt.show()

	        ary_wfs[i,j] = wf     

	# np.save('/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_station_52_voltage_trace.npy',ary_wfs)

# save_wfs('/Users/astrid/Desktop/st52_deeplearning/data/stn52_2019_cr_candidates.nur')

# save_wfs('/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_noResamp_station_52.nur')

#_________________



#_________________

def fft_dl():
	
	# wfs = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_station_52_voltage_trace.npy')
	ffts = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_station_52_fft.npy')
	print(ffts.shape)
	f = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_station_52_freqs.npy')
	wfs = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stn52_2019_cr_candidates.npy')

	sim_avg_fft = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_stn52_sim_cr_all9629_avg_fft.npy')
	# print(sim_avg_fft.shape)

	wfs = np.reshape(wfs, (wfs.shape[0], wfs.shape[1],wfs.shape[2],1))
	model = keras.models.load_model(f'/Users/astrid/Desktop/st52_deeplearning/h5s/trained_CNN_1l-10-4-10-do0.5_fltn_sigm_valloss_p4_noise0-18k_amp_below_70mV_CRsignal0-6k_repeat_sig_shuff_monitortraining_0.h5')
	# model = keras.models.load_model(f'/Users/astrid/Desktop/st52_deeplearning/h5s/trained_CNN_1l-10-4-10_fltn_sigm_valloss_p4_noise_lowampl_lt70mV_0-20k_CRsignal_no-weight_0-5k_repeat_sig4x_shuff_monitortraining_4upLPDAs_0.h5')

	# probs = model.predict(wfs)

	# s = np.zeros((100,8,257))
	# n = np.zeros((100,8,257))
	# count_s=0
	# count_n=0
	# for i,p in enumerate(probs):
	# 	if p>=0.5:
	# 		s[count_s] = ffts[i]
	# 		count_s+=1
	# 	else:			
	# 		n[count_n] = ffts[i]
	# 		count_n+=1

	# s = s[0:count_s]
	# n = n[0:count_n]

	# print(s.shape,n.shape)

	# both = np.vstack((s,n))
	# both = both.mean(axis=0)
	# tot_sum_low = n.mean(axis=0)
	# tot_sum_high = s.mean(axis=0)

	tot_sum_low = ffts.mean(axis=0)

	fig = plt.figure(figsize=(7, 5))
	axes = fig.subplots(nrows=4, ncols=2, sharex=True, sharey=True)
	c = ['blue','red']
	# for k,tot in enumerate([tot_sum_low,tot_sum_high]):
	for k,tot in enumerate([tot_sum_low]):
		# fig = plt.figure(figsize=(7, 5))
		# axes = fig.subplots(nrows=4, ncols=2, sharex=True) #, sharey=True)

		for i in range(8):
		    if i == 0:
		        ax = axes[2, 0]
		        ax.set_title('ch0: down LPDA')
		        ax.tick_params(bottom=False)
		        # ax.set_ylabel('Amplitude (mV)')
		    elif i == 1:
		        ax = axes[2, 1]
		        ax.set_title('ch1: down LPDA')
		        ax.tick_params(bottom=False)
		    elif i == 2:
		        ax = axes[3, 0]
		        ax.set_title('ch2: dipole')
		        # ax.set_ylabel('Amplitude (mV)')
		    elif i == 3:
		        ax = axes[3, 1]
		        ax.set_title('ch3: dipole')
		    elif i == 4:
		        ax = axes[0, 0]
		        ax.set_title('ch4: up LPDA')
		        ax.tick_params(bottom=False)
		        # ax.set_ylabel('Amplitude (mV)')
		    elif i == 5:
		        ax = axes[0, 1]
		        ax.set_title('ch5:up LPDA')
		        ax.tick_params(bottom=False)
		    elif i == 6:
		        ax = axes[1, 0]
		        ax.set_title('ch6: up LPDA')
		        ax.tick_params(bottom=False)
		        # ax.set_ylabel('Amplitude (mV)')
		        # ax.set_xlabel('Time (ns)')
		    elif i == 7:
		        ax = axes[1, 1]
		        ax.set_title('ch7: up LPDA')
		        ax.tick_params(bottom=False)
		    #     ax.set_xlabel('Time (ns)')
		    # ax.set_title(f'channel{i}')
		  	
		  	
		    ax.plot(f, sim_avg_fft[i]/(max(sim_avg_fft[i])-min(sim_avg_fft[i])), lw=1,label='avg simulated CR')
		    # ax.plot(f, sim_avg_fft[i], lw=1,label='avg sim CR')
		    axes[3, 0].set_xlabel('Frequency [MHz]')
		    axes[3, 1].set_xlabel('Frequency [MHz]')
		    # ax.set_ylabel('Power[V * s^2]')
		    ax.set_ylabel('normalized P')
		    ax.plot(f, tot[i]/(max(tot[i])-min(tot[i])), lw=1,label='avg experimental CR')
		    # ax.plot(f, tot[i], lw=1,label='avg tagged CR ')
		    ax.set_xlim([0,512])
		    # ax.set_ylim([0,0.702])
		    # ax.set_ylim([0,0.702])
		    
		    # ax.set_title(
		    #     f'stn {station_id} run {run_num} evt {evt_id} time:{stn_time}')
	#indent 1 to plot both low and high, indent 2 to plot individually
	plt.legend()
	# plt.xlabel('Frequency [MHz]')
	plt.show()
	plt.close(fig)

# fft_dl()




#_________________


#_________________

def fft_dl_4chan():
	
	# wfs = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_station_52_voltage_trace.npy')
	ffts = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_station_52_fft.npy')[:,4:]
	f = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_station_52_freqs.npy')
	wfs = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stn52_2019_cr_candidates.npy')[:,4:]

	sim_avg_fft = np.load('/Users/astrid/Desktop/st52_deeplearning/data/stop_bandpass_incl_stn52_sim_cr_all9629_avg_fft.npy')[4:]
	# print(sim_avg_fft.shape)

	wfs = np.reshape(wfs, (wfs.shape[0], wfs.shape[1],wfs.shape[2],1))
	# model = keras.models.load_model(f'/Users/astrid/Desktop/st52_deeplearning/h5s/trained_CNN_1l-10-4-10_fltn_sigm_valloss_p4_noise0-42k_all_CRsignal_no-weight_0-7k_repeat_sig6x_shuff_monitortraining_4-7upLPDA0.h5')
	
	model = keras.models.load_model(f'/Users/astrid/Desktop/st52_deeplearning/h5s/trained_CNN_1l-10-4-10_fltn_sigm_valloss_p4_noise_lowampl_lt60mV_0-20k_CRsignal_no-weight_0-5k_repeat_sig4x_shuff_monitortraining_4upLPDAs_0.h5')

	probs = model.predict(wfs)

	s = np.zeros((100,4,257))
	n = np.zeros((100,4,257))
	count_s=0
	count_n=0
	for i,p in enumerate(probs):
		if p>=0.5:
			s[count_s] = ffts[i]
			count_s+=1
		else:			
			n[count_n] = ffts[i]
			count_n+=1

	s = s[0:count_s]
	n = n[0:count_n]

	print(s.shape,n.shape)

	both = np.vstack((s,n))
	both = both.mean(axis=0)
	tot_sum_low = n.mean(axis=0)
	tot_sum_high = s.mean(axis=0)

	fig = plt.figure(figsize=(7, 5))
	axes = fig.subplots(nrows=4, ncols=1, sharex=True, sharey=True)
	c = ['blue','red']
	for k,tot in enumerate([tot_sum_low,tot_sum_high]):
		# fig = plt.figure(figsize=(7, 5))
		# axes = fig.subplots(nrows=4, ncols=1, sharex=True) #, sharey=True)

		for i in range(4):
		    if i == 0:
		        ax = axes[0]
		        ax.set_title('ch4: up LPDA')
		        ax.tick_params(bottom=False)
		        # ax.set_ylabel('Amplitude (mV)')
		    elif i == 1:
		        ax = axes[1]
		        ax.set_title('ch5: up LPDA')
		        ax.tick_params(bottom=False)
		    elif i == 2:
		        ax = axes[2]
		        ax.set_title('ch6: up LPDA')
		        ax.tick_params(bottom=False)
		        # ax.set_ylabel('Amplitude (mV)')
		        # ax.set_xlabel('Time (ns)')
		    elif i == 3:
		        ax = axes[3]
		        ax.set_title('ch7, up LPDA')
		        ax.tick_params(bottom=False)
		    #     ax.set_xlabel('Time (ns)')
		    # ax.set_title(f'channel{i}')
		  	
		    if k==0:
		    	ax.plot(f, sim_avg_fft[i]/(max(sim_avg_fft[i])-min(sim_avg_fft[i])), lw=1,label='avg simulated CR')
		    # ax.plot(f, sim_avg_fft[i], lw=1,label='avg sim CR')
		    titles = 'experimental CR: classif. <0.5'
		    if k>0:
		    	titles = 'experimental CR: classif. >0.5'
		    ax.set_ylabel('normalized P')
		  
		    ax.plot(f, tot[i]/(max(tot[i])-min(tot[i])), lw=1,label=f'{titles}')
		    # ax.plot(f, tot[i], lw=1,label=k,color=c[k])
		    ax.set_xlim([0,512])
		    # ax.set_ylim([0,0.702])
		    # ax.set_ylim([0,0.702])
		    
		    # ax.set_title(
		    #     f'stn {station_id} run {run_num} evt {evt_id} time:{stn_time}')
	#indent 1 to plot both low and high, indent 2 to plot individually
	plt.legend(loc='upper right')
	plt.xlabel('Frequency [MHz]')
	plt.xlim(50,550)
	plt.ylim(0,1.2)
	plt.show()
	plt.close(fig)

fft_dl_4chan()









# # second way: manual fft (works for any wave)
# evt = data.get_event([236, 2])
# stn = evt.get_station(51)
# chn = stn.get_channel(0)
# trace = chn.get_trace()
# frequencies = chn.get_frequencies()/units.MHz
# sampling_rate = chn.get_sampling_rate()
# trace_fft = abs(fft.time2freq(trace, sampling_rate))

# fig, ax = plt.subplots()
# ax.set_title('Manual fft')
# ax.plot(frequencies, trace_fft)
# ax.set_xlabel('frequencies [MHz]')
# ax.set_ylabel('power')


# # using fft to analyze the frequency of a sin wave

# x = np.linspace(0, 100, 1000)
# y = np.sin(x)
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 5))
# ax1.set_title('time domain')
# ax1.plot(x, y)
# ax1.set_xlabel('time')
# ax1.set_ylabel('amplitude')

# fft_trace = abs(fft.time2freq(y, 1))
# ax2.set_title('frequency domain')
# ax2.plot(range(len(fft_trace)), fft_trace)
# ax2.set_xlabel('frequency')
# ax2.set_ylabel('amplitude')


# # inverse-fft : get time-domain wave from a frequency spectrum

# y = fft.freq2time(fft_trace, 1)
# ax3.set_title('inverse fft of frequency spectrum')
# ax3.plot(range(len(y)), y)
# ax3.set_xlabel('time')
# ax3.set_ylabel('amplitude')


# plt.show()
