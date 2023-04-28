import glob
import os
from NuRadioReco.modules.io import eventReader
from NuRadioReco.framework.parameters import ARIANNAParameters as ARIpar
import numpy as np
from numpy import save, load
import matplotlib.pyplot as plt


def main(noise_files, run_num):
    print(f'running file {run_num}')

    reader = eventReader.eventReader()
    reader.begin(noise_files)
    all_evts = np.zeros((150000, 4, 256))
    rate_ind_vals = np.zeros((150000, 4))
    count = 0
    
    for i, event in enumerate(reader.run()):
        for station_object in event.get_stations():
            if station_object.has_triggered() is True:
                station = event.get_station(18)
                seq = station.get_ARIANNA_parameter(ARIpar.seq_num)
                if seq is None:
                    print('none')
                    continue
                else:
                    for j, channel in enumerate(station.iter_channels()):
                        all_evts[count, j] = np.array([channel.get_trace()])
                    count += 1

    all_evts_final = all_evts[0:count]
    save(f'/Volumes/External/wind_events/arianna_data_st18-19/st18/all_st18_data/all_traces/traces_station_18_run_00{run_num}.npy', all_evts_final)

for i in range(14, 272):
    run_num = f"{i:03d}"
    fname = f'/Volumes/External/wind_events/arianna_data_st18-19/st18/all_st18_data/station_18_run_00{run_num}.root.nur'
    npy_name = f'/Volumes/External/wind_events/arianna_data_st18-19/st18/all_st18_data/no_thresh/rates_info_avg2seq_station_18_run_00{run_num}_4vals.npy'
    if os.path.isfile(fname) and os.path.isfile(npy_name):
        noise_files = glob.glob(fname)
        main(noise_files, run_num)
    else:
        continue

