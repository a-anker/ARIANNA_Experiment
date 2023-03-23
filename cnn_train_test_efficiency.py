import os
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D

"""
This script trains a basic CNN with function train_cnn and then plots the efficiency curve with the function efficiency_curve

The parameters to set are the path to input data, the noise and signal files, and the output name for the model that will be trained and tested
"""

path = "/arianna_data"
noise = np.load(os.path.join(path, "noise.npy")) #input a subset of the data here so that you can validate on the other set
signal = np.load(os.path.join(path, "signal.npy")) #make sure the signal and noise subset of data are the same size
model_name = 'trained_CNN_1l-10-8-10_do0.5_mp10_fltn_sigm'


if signal.ndim==2:
    signal = np.reshape(signal, (signal.shape[0], 1, signal.shape[1]))
    noise = np.reshape(noise, (noise.shape[0], 1, noise.shape[1]))


def train_cnn():
  #make signal the same shape as the noise data, if needed
  # signal = np.vstack((signal,signal,signal,signal))
  # signal = signal[0:noise.shape[0]]

  print(signal.shape)
  print(noise.shape)

  x = np.vstack((noise, signal)) 

  n_samples = x.shape[2]
  n_channels = x.shape[1]
  x = np.expand_dims(x, axis=-1)
  y = np.vstack((np.zeros((noise.shape[0], 1)), np.ones((signal.shape[0], 1))))
  s = np.arange(x.shape[0])
  np.random.shuffle(s)
  x = x[s]
  y = y[s]
  print(x.shape)

  BATCH_SIZE = 32
  EPOCHS = 100

  callbacks_list = [
  keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

  model = Sequential()
  model.add(Conv2D(10, (8, 10), activation='relu', input_shape=(n_channels, n_samples, 1)))
  model.add(Dropout(0.5))
  model.add(MaxPooling2D(pool_size=(1, 10)))
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))
  model.compile(optimizer='Adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  model.summary()

  #input the path and file you'd like to save the model as (in h5 format)
  model.save(f'{model_name}.h5')



def efficiency_curve(h5_name, n_dpt, colors):

    
    n_signal = signal.shape[0]
    n_noise = noise.shape[0]
    x = np.vstack((signal, noise))
    x = np.expand_dims(x, axis=-1)
    # x = np.swapaxes(x, 1, 2)

    y = np.zeros((x.shape[0], 2))
    y[:n_signal, 1] = 1
    y[n_signal:, 0] = 1

    model = keras.models.load_model(f'{path}/{h5_name}.h5')
    y_pred = model.predict(x)
    print(y_pred)

    ary = np.zeros((2, n_dpt * 2))
    vals = np.zeros((2 * n_dpt))  # array of threshold cuts
    vals[:n_dpt] = np.linspace(0, 0.9, n_dpt) #doing this in two steps gives more detail in the higher cut values which is usually where the detail is needed
    vals[n_dpt:] = np.linspace(0.9, 1, n_dpt)
    for i, threshold in enumerate(vals):

        eff_signal = np.sum((y_pred[:signal.shape[0], 0] > threshold) == True) / n_signal
        eff_noise = np.sum((y_pred[signal.shape[0]:, 0] > threshold) == False) / n_noise

        if(eff_noise < 1):
            reduction_factor = (1 / (1 - eff_noise))
            ary[0][i] = reduction_factor
        else:
            reduction_factor = (n_noise)
            ary[0][i] = reduction_factor
        ary[1][i] = eff_signal

    return ary[1][1:], ary[0][1:]



def main():

    train_cnn()

    x1, y1 = efficiency_curve(h5_name=model_name, n_dpt = 500,colors='blue')
    plt.plot(x1[0::10], y1[0::10], label='cnn', linewidth=3) #syntax [0::10] plots every 10 events to give a smoother curve
    plt.legend(loc='lower left')
    plt.yscale('log')
    plt.xlim(0.91, 1.1)
    plt.ylim(1, 10**6)
    plt.grid(True, 'major', 'both', linewidth=0.5)
    plt.xlabel('signal efficiency', fontsize=15)
    plt.ylabel('noise reduction factor', fontsize=15)
    plt.show()

 
if __name__== "__main__":
    main()

