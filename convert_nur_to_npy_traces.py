import glob
import os
from NuRadioReco.modules.io import eventReader
from NuRadioReco.framework.parameters import ARIANNAParameters as ARIpar
import numpy as np
from numpy import save, load
import matplotlib.pyplot as plt

######################################################################################################################
"""
This script takes in a range of .nur files with different run number labels (00000,00001,....00009),
extracts the traces/waveforms for all channels and event within the run number, and save them to a numpy array.
"""
######################################################################################################################
PathToARIANNA = os.environ['ARIANNA_Experiment']
chan_num = 4 #the number of channels in the input data used

def convert_to_numpy(data_file, run_num):
    print(f'running file {run_num}')

    reader = eventReader.eventReader()
    reader.begin(data_file)
    all_evts = np.zeros((150000, chan_num, 256))
    rate_ind_vals = np.zeros((1000000, chan_num))
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
    save(PathToARIANNA + f'/traces_station_18_run_00{run_num}.npy', all_evts_final)
    
def main():
    for i in range(10):
        run_num = f"{i:03d}"
        fname = PathToARIANNA + f'/station_18_run_00{run_num}.root.nur'
        if os.path.isfile(fname):
            noise_files = glob.glob(fname)
            convert_to_numpy(noise_files, run_num)
        else:
            continue

if __name__== "__main__":
    main()
