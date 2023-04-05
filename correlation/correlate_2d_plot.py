import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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
#####note, run without conda environment for the pick 5 command
# x = np.load('/Users/astrid/Desktop/stn61_data_snr_chi_all_evts_10GHz.npy',allow_pickle=True)



# [np.deg2rad(120)][np.deg2rag(27.5)][np.deg2rad(1.0)][channel_id]
objects = []
with (open("templates_nu_station_61.pickle", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
# print(len(objects))
# print(objects[0][np.deg2rad(120)][np.deg2rad(27.5)][np.deg2rad(1.0)])
# print(objects[0].keys())
# templ = objects[0][2.6179938779914944][0.4799655442984406][0.03490658503988659] #2nd template values _2
templ = objects[0][np.deg2rad(120)][np.deg2rad(27.5)][np.deg2rad(1.0)] # 1.31669345e-03  1.38535068e-03  2.63351951e-03 -3.39897923e-03, the first template without _ extension and also is _filtered
templ_array = np.zeros((8,512))
for i in range(8):
    templ_array[i] = templ[i]
templ = scipy.signal.resample(templ_array,5120,axis=1)


print(templ.shape)
path = '/Users/astrid/Desktop/st61_deeplearning/data/2ndpass'
noise = np.load(os.path.join(path, f"stn61_2of4trigger_noise_all_shuffledAA.npy"))
# noise = scipy.signal.resample(noise,2560,axis=2)
# np.save(path, f"stn61_2of4trigger_noise_all_shuffledAA_upsampled_2560.npy",noise)
print(noise.shape)
# # print(noise.shape[0])
# n1 = noise[45449]
# n2 = noise[57537]
# n = np.array((n1,n2))


signal = np.load(os.path.join("/Users/astrid/Desktop/st61_deeplearning/data/2ndpass/", "stn61_simulation_noise_trace.npy"))
# signal = scipy.signal.resample(signal,2560,axis=2)
# np.save(os.path.join("/Users/astrid/Desktop/st61_deeplearning/data/2ndpass/", "stn61_simulation_noise_trace_upsampled_2560.npy"),signal)

effcut_97 = np.load('/Users/astrid/Desktop/st61_deeplearning/data/extract_97_confidence_curve/97p_confidence_curve_neutrino_efficiency_2.npy')
red_dots = np.load('/Users/astrid/Desktop/st61_deeplearning/data/extract_97_confidence_curve/red_points_dipole_cut_53_SNR_chi.npy')


# lz_calc_vals = np.load('/Users/astrid/Desktop/st61_deeplearning/data/stn61_data_snr_chi_all_evts_updown_cut.npy',allow_pickle=True)

# x = scipy.signal.resample(n1,2560,axis=1)
# print(x.shape)
# plt.scatter(np.linspace(1,256,2560),x[0])
# plt.scatter(np.linspace(1,256,256),n1[0])
# plt.show()

def correl(data):
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
        # mx_corr = np.max([np.max(abs(xcorr0)),np.max(abs(xcorr1)),np.max(abs(xcorr2)),np.max(abs(xcorr3))])
        # print(mx_corr)

        max_ch_val = 0
        for ch in range(4):
            trace = ary[ch]
            mx_intermed = np.max(abs(trace))   # units in mV
            if mx_intermed>max_ch_val:
                max_ch_val=mx_intermed
        max_snr = max_ch_val/0.0105
        # print(mx_corr,i)
        ary_vals[i] = np.array((max_snr,mx_corr))
    return ary_vals

nary = correl(noise)
sary = correl(signal)

# import collections
# print(collections.Counter(np.around(lz_calc_vals[0],5)) == collections.Counter(np.around(ary_vals[:,0],5)))


fig, ax = plt.subplots()
yspace = np.linspace(0, 1, 100)
xspace = np.logspace(np.log10(3.0), np.log10(200), 100)

s = ax.hist2d(sary[:,0],sary[:,1],label=f'signal: {signal.shape[0]}',bins=(xspace,yspace),cmap=plt.cm.Blues,norm=colors.LogNorm())
n = ax.hist2d(nary[:,0],nary[:,1],label=f'all events: {noise.shape[0]}',cmin = 1,bins=(xspace,yspace),cmap=plt.cm.viridis,norm=colors.LogNorm())

plt.scatter(red_dots[:,0],red_dots[:,1],color='red',s=30,label='94.8% dipole cut: 53 events')
ax.scatter(nary[45449,0],nary[45449,1], s=80,marker='^',color='white')
ax.scatter(nary[57537,0],nary[57537,1], s=80,marker='^',color='white')


plt.plot(effcut_97[:,0],effcut_97[:,1],linestyle='dashed',color='black',linewidth=4,label='97.1% neutrino efficiency')

plt.ylabel(r'$\chi_{LPDA}$',fontsize=18)
plt.xlabel(r'SNR ( max(|amplitude|) / $V^{noise}_{RMS}$ )',fontsize=18)
plt.xscale('log')
plt.ylim(0,1)
plt.xticks(size=18)
plt.yticks(size=18)


cb = fig.colorbar(n[3],ax=ax)
cb.set_label('noise event density',fontsize=18)

# print(s)
cb2 = fig.colorbar(s[3],ax=ax)
cb2.set_label('signal event density',fontsize=18)

# handles, labels = ax.get_legend_handles_labels()
# new_handles = [Line2D([], [], c=h.get_edgecolor(), linestyle=h.get_linestyle()) for h in handles]
# plt.legend(loc='upper center', handles=new_handles, labels=labels, fontsize=18)
plt.legend(fontsize=18)
plt.show()
    
