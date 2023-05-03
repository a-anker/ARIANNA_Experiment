from matplotlib import pyplot as plt
from scipy.signal import correlate
from numpy import save, load
import numpy as np
import os
import time

######################################################################################################################
"""
This script calculates the correlation of input signal and noise files with an input template to get prediction values between 0 and 1. 
Then the same data is run through a previously trained model to obtain the network output values between 0 and 1. 
Each set of signal and noise pairs are then scanned through to calculate the signal efficiency vs noise rejection and make the plot.
"""
######################################################################################################################
PathToARIANNA = os.environ['ARIANNA_Experiment']
noise = np.load(PathToARIANNAData + '/data/measured_noise.npy') 
signal = np.load(PathToARIANNAData + '/data/measured_signal.npy') 
templ = np.load(PathToARIANNAData + '/template_study/trimmed100_data_signal_LPDA_ch0_0.036sa_10.0vrms_no_noise.npy')[0, :]
h5_model = "/trained_for_extract_weights_100samp_cnn_1layer5-10_mp10_s1_1output.h5"

def correl_calc(data):
    xcorr_array = np.zeros((data.shape[0]))
    for i in range(len(data)):
      
        # first value is stationary and then second input is convolved over the first. First value is first val in template and last value in 2nd input
        xcorr = correlate(templ, data[i], mode='full', method='auto')  # / (np.sum(templ ** 2) * np.sum(data[i] ** 2)) ** 0.5
        print((xcorr[1], templ, data[i]))
        xcorrpos = np.argmax(np.abs(xcorr))
        print(xcorrpos)
        xcorr = xcorr[xcorrpos]
        xcorr_array[i] = xcorr
    return xcorr_array
 

def cnn_efficiency():
    n_signal = signal.shape[0]
    n_noise = noise.shape[0]
    s = np.expand_dims(signal, axis=-1)
    n = np.expand_dims(noise, axis=-1)

    model = keras.models.load_model(PathToARIANNA + h5_model)
    sig_corr = model.predict(s)
    noise_corr = model.predict(n)
    
    n_dpt = 800
    ary = np.zeros((2, n_dpt))
    vals = np.zeros((n_dpt))  # array of threshold cuts
    vals[:n_dpt] = np.linspace(0, 1, n_dpt)
    for i, threshold in enumerate(vals):
        eff_signal = np.sum((np.abs(sig_corr) > threshold) == True) / len(sig_corr)
        eff_noise = np.sum((np.abs(noise_corr) > threshold) == False) / len(noise_corr)

        if(eff_noise < 1):
            reduction_factor = (1 / (1 - eff_noise))
            ary[0][i] = reduction_factor
        else:
            reduction_factor = (len(noise_corr))
            ary[0][i] = reduction_factor
        ary[1][i] = eff_signal
    return ary[1][1:], ary[0][1:]


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    templ_sig_corr = correl_calc(signal)
    templ_noise_corr = correl_calc(noise)
    x, y = cnn_efficiency()

    plt.plot(x[0::10], y[0::10], label='CNN: 100 samples', linewidth=3) #syntax [0::10] plots every 10 events to give a smoother curve
    plt.plot(templ_signal_corr[0::10], templ_noise_corr[0::10], label='template: 100 samples', linewidth=3) #syntax [0::10] plots every 10 events to give a smoother curve
    
    plt.legend(loc='lower left')
    plt.yscale('log')
    plt.grid(True, 'major', 'both', linewidth=0.5)
    plt.xlabel('signal efficiency', fontsize=15)
    plt.ylabel('noise reduction factor', fontsize=15)
    plt.show()


if __name__== "__main__":
    main()
