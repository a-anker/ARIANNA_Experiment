from matplotlib import pyplot as plt
from scipy.signal import correlate
from numpy import save, load
import numpy as np
import os
import time

templ = np.load('/Volumes/External/ML_paper/cross_correlation_study/trimmed100_data_signal_LPDA_ch0_0.036sa_10.0vrms_no_noise.npy')[0, :]

path = "/Volumes/External/arianna_data"
signal = np.load(os.path.join(path, "trimmed100_data_signal_3.6SNR_1ch_0001.npy"))
noise = np.zeros((500000, 100), dtype=np.float32)
for i in range(6, 11):
    noise[(i - 6) * 100000:(i - 6 + 1) * 100000] = np.load(os.path.join(path, f"trimmed100_data_noise_3.6SNR_1ch_{i:04d}.npy")).astype(np.float32)

print(signal.shape)
print(noise.shape)


def correl_calc(data, save_path):
    xcorr_array = np.zeros((data.shape[0]))
    for i in range(len(data)):
        # print(i)
        # first value is stationary and then second input is convolved over the first. First value is first val in template and last value in 2nd input
        xcorr = correlate(templ, data[i], mode='full', method='auto')  # / (np.sum(templ ** 2) * np.sum(data[i] ** 2)) ** 0.5
        print((xcorr[1], templ, data[i]))
        xcorrpos = np.argmax(np.abs(xcorr))
        print(xcorrpos)
        xcorr = xcorr[xcorrpos]
        xcorr_array[i] = xcorr
        time.sleep(5)
    # np.save(save_path, xcorr_array)


# correl_calc(signal, '/Users/astrid/Desktop/ML_paper/cross_correlation_study/signal_correlation')
# correl_calc(noise, '/Users/astrid/Desktop/ML_paper/cross_correlation_study/noise_correlation')


def plot_correl(sig_corr, noise_corr):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlabel('correlation', fontsize=17)
    plt.ylabel('events', fontsize=17)
    plt.xticks(size=15)
    plt.yticks(size=15)

    # plt.yscale('log')

    weights = 1 * np.ones_like(sig_corr) / len(sig_corr)
    ax.hist(np.abs(sig_corr), bins=40, range=(0, 1), histtype='step', color='blue', linestyle='solid',
            label=f'signal', linewidth=1.5, weights=weights)

    weights = 1 * np.ones_like(noise_corr) / len(noise_corr)
    ax.hist(np.abs(noise_corr), bins=40, range=(0, 1), histtype='step', color='red', linestyle='solid',
            label=f'noise', linewidth=1.5, weights=weights)
    plt.legend()
    plt.show()


sig_corr = np.load('/Volumes/External/ML_paper/cross_correlation_study/signal_correlation.npy')
noise_corr = np.load('/Volumes/External/ML_paper/cross_correlation_study/noise_correlation.npy')

# calc efficiency for 1 cut value
# count = 0
# for val in np.abs(sig_corr):
#     if val >= 0.5:
#         count += 1
# print(f'signal efficiency: {count / len(sig_corr)}')
# count = 0
# for val in np.abs(noise_corr):
#     if val < 0.5:
#         count += 1
# print(f'noise efficiency: {count / len(noise_corr)}')

# plot_correl(sig_corr, noise_corr)

# plot all efficiency values


def cnn():
    n_dpt = 800
    ary = np.zeros((2, n_dpt))
    vals = np.zeros((n_dpt))  # array of threshold cuts
    vals[:n_dpt] = np.linspace(0, 1, n_dpt)
    # vals[n_dpt:] = np.linspace(0.9, 1, n_dpt)
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

    # plt.plot(ary[1][1:], ary[0][1:], c=colors, label=label)
    return ary[1][1:], ary[0][1:]


x, y = cnn()
ary = np.array((x, y))
print(ary.shape)
np.save('/Volumes/External/ML_paper/cross_correlation_study/signal_vs_noise_eff_values2', ary)
plt.plot(x, y, label='', linewidth=3)


plt.legend(loc='lower left')
plt.yscale('log')
# plt.xlim(0.91, 1)
# plt.ylim(1, 10**6)
plt.grid(True, 'major', 'both', linewidth=0.5)
plt.xlabel('signal efficiency', fontsize=15)
plt.ylabel('noise reduction factor', fontsize=15)
plt.show()
