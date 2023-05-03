import os
import glob
import json
from matplotlib import pyplot as plt
import numpy as np
from numpy import save, load
from matplotlib.lines import Line2D
from matplotlib import rc
import pickle5 as pickle
from scipy.signal import correlate
from matplotlib.lines import Line2D
import datetime
import NuRadioReco.utilities.units as units
import NuRadioReco.detector.detector as detector
import NuRadioReco.modules.io.NuRadioRecoio
from NuRadioReco.framework.parameters import stationParameters as stnp
from NuRadioReco.framework.parameters import channelParameters as chp
import NuRadioReco.modules.channelSignalReconstructor
import NuRadioReco.modules.channelAddCableDelay
import NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator
import matplotlib.colors as colors
import time
import scipy

######################################################################################################################
"""
This script plots the signal-to-noise radio (SNR) versus the correlation of the LPDAs for lines, scatters, and colorbar plots.
The inputs are signal and noise files, a neutrino template, and a few other relevant distributions.
"""
######################################################################################################################
PathToARIANNA = os.environ['ARIANNA_Experiment']
path = '/Users/astrid/Desktop/st61_deeplearning/data/2ndpass'
noise = np.load(PathToARIANNAData + '/data/measured_noise.npy')[:,0:4]
signal = np.load(PathToARIANNAData + '/data/measured_signal.npy')[:,0:4]
template = np.load(PathToARIANNAData + '/correlation/templates_nu_station_61.pickle')
rms_val = 0.0105 #give in Volts

#both of these data sets were extracted from another plot with this useful program: https://apps.automeris.io/wpd/
effcut_97 = np.load(PathToARIANNAData + '/correlation/97p_confidence_curve_neutrino_efficiency_2.npy')
red_dots = np.load(PathToARIANNAData +  '/correlation/red_points_dipole_cut_53_SNR_chi.npy')
indx1 = 45449 #these are the index values of the noise data points misclassified by the DL network
indx2 = 57537

def prep_templ():
    objects = []
    with (open(template, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    templ = objects[0][np.deg2rad(120)][np.deg2rad(27.5)][np.deg2rad(1.0)] 
    templ_array = np.zeros((,512))
    for i in range(8):
        templ_array[i] = templ[i]
    templ = scipy.signal.resample(templ_array,5120,axis=1)
    return templ


def correl(data):
    templ = prep_templ()
    ary_vals = np.zeros((len(data),2))
    for i in range(len(data)):
    
        ary = scipy.signal.resample(data[i],2560,axis=1)

        xcorr0 = correlate(templ[0],ary[0],mode='full')
        xcorr0 = xcorr0 / ((np.sum(templ[0] ** 2) * np.sum(ary[0] ** 2)) ** 0.5)

        xcorr1 = correlate(templ[1],ary[1],mode='full')
        xcorr1 = xcorr1 / ((np.sum(templ[1] ** 2) * np.sum(ary[1] ** 2)) ** 0.5)

        xcorr2 = correlate(templ[2],ary[2],mode='full')
        xcorr2 = xcorr2 / ((np.sum(templ[2] ** 2) * np.sum(ary[2] ** 2)) ** 0.5)

        xcorr3 = correlate(templ[3],ary[3],mode='full')
        xcorr3 = xcorr3 / ((np.sum(templ[3] ** 2) * np.sum(ary[3] ** 2)) ** 0.5)

        mean02 = sum([np.max(abs(xcorr0)),np.max(abs(xcorr2))])/2
        mean13 = sum([np.max(abs(xcorr1)),np.max(abs(xcorr3))])/2
        mx_corr = np.max([mean02,mean13])

        max_ch_val = 0
        for ch in range(data.shape[1]):
            trace = ary[ch]
            mx_intermed = np.max(abs(trace))
            if mx_intermed>max_ch_val:
                max_ch_val=mx_intermed
        max_snr = max_ch_val/rms_val
        ary_vals[i] = np.array((max_snr,mx_corr))
    return ary_vals

def main():
    nary = correl(noise)
    sary = correl(signal)

    fig, ax = plt.subplots()
    yspace = np.linspace(0, 1, 100)
    xspace = np.logspace(np.log10(3.0), np.log10(200), 100)

    s = ax.hist2d(sary[:,0],sary[:,1],label=f'signal: {signal.shape[0]}',bins=(xspace,yspace),cmap=plt.cm.Blues,norm=colors.LogNorm())
    n = ax.hist2d(nary[:,0],nary[:,1],label=f'all events: {noise.shape[0]}',cmin = 1,bins=(xspace,yspace),cmap=plt.cm.viridis,norm=colors.LogNorm())

    plt.scatter(red_dots[:,0],red_dots[:,1],color='red',s=30,label='94.8% dipole cut: 53 events')
    ax.scatter(nary[indx1,0],nary[indx1,1], s=80,marker='^',color='white')
    ax.scatter(nary[indx2,0],nary[indx2,1], s=80,marker='^',color='white')
    plt.plot(effcut_97[:,0],effcut_97[:,1],linestyle='dashed',color='black',linewidth=4,label='97.1% neutrino efficiency')

    plt.ylabel(r'$\chi_{LPDA}$',fontsize=18)
    plt.xlabel(r'SNR ( max(|amplitude|) / $V^{noise}_{RMS}$ )',fontsize=18)
    plt.xscale('log')
    plt.ylim(0,1)
    plt.xticks(size=18)
    plt.yticks(size=18)

    cb = fig.colorbar(n[3],ax=ax)
    cb.set_label('noise event density',fontsize=18)

    cb2 = fig.colorbar(s[3],ax=ax)
    cb2.set_label('signal event density',fontsize=18)

    plt.legend(fontsize=18)
    plt.show()

    
if __name__== "__main__":
    main()
