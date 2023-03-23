import numpy as np
import keras
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from matplotlib import pyplot as plt


"""
This script trains a basic FCNN with function train_nn and then plots the efficiency curve with the function efficiency_curve

The parameters to set are the path to input data, the noise and signal files, and the output name for the model that will be trained and tested.

The model parameters in the function train_nn can also be changed to create a different model
"""

path = "/arianna_data"
noise = np.load(os.path.join(path, "noise.npy")) #input a subset of the data here so that you can validate on the other set
signal = np.load(os.path.join(path, "signal.npy")) #make sure the signal and noise subset of data are the same size
model_name = 'trained_NN_1ch_1l_128n_10e_bs32_lr0.0008'
model_path = "/h5_model_path"


if signal.ndim==2:
    signal = np.reshape(signal, (signal.shape[0], 1, signal.shape[1]))
    noise = np.reshape(noise, (noise.shape[0], 1, noise.shape[1]))

signal = np.reshape(signal, (signal.shape[0], signal.shape[1]*signal.shape[1])) #creates one dimentional array (flattening it) to use for training
noise = np.reshape(noise, (noise.shape[0], noise.shape[1]*noise.shape[1]))


def train_nn():

    x = np.vstack((noise, signal))  
    in_dim = x.shape[1]
    y = np.vstack((np.zeros((noise.shape[0], 1)), np.ones((signal.shape[0], 1))))

    s = np.arange(x.shape[0])
    np.random.shuffle(s)

    x = x[s]
    y = y[s]

    BATCH_SIZE = 32
    EPOCHS = 10
    in_nodes = 128 #can change this to create a bigger or smaller hidden layer

    model = Sequential()
    model.add(Dense(in_nodes, input_dim=in_dim))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.0008)  # default 0.001
    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x, y, validation_split=.1, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)  # training on the data
    model.summary()

    model.save(f'{model_path}/{model_name}.h5')
 


def efficiency_curve_nn(h5_name, n_dpt, colors):

    n_signal = signal.shape[0]
    n_noise = noise.shape[0]
 
    x = np.vstack((noise, signal)) 
    in_dim = x.shape[1]
    y = np.vstack((np.zeros((noise.shape[0], 1)), np.ones((signal.shape[0], 1))))

    model = keras.models.load_model(f'{model_path}/{h5_name}.h5')
    y_pred = model.predict(x)

    ary = np.zeros((2, n_dpt * 2))
    vals = np.zeros((2 * n_dpt))  # array of threshold cuts
    vals[:n_dpt] = np.linspace(0, 0.9, n_dpt) #syntax [0::10] plots every 10 events to give a smoother curve
    vals[n_dpt:] = np.linspace(0.9, 1, n_dpt)

    for i, threshold in enumerate(vals):
        eff_noise = np.sum((y_pred[:noise.shape[0], 0] > threshold) == False) / n_noise
        eff_signal = np.sum((y_pred[noise.shape[0]:, 0] > threshold) == True) / n_signal
        if(eff_noise < 1):
            reduction_factor = (1 / (1 - eff_noise))
            ary[0][i] = reduction_factor
        else:
            reduction_factor = (n_noise)
            ary[0][i] = reduction_factor
        ary[1][i] = eff_signal

    return ary[1][1:], ary[0][1:]



def main():

    train_nn()

    x1, y1 = efficiency_curve_nn(h5_name=model_name, n_dpt = 500,colors='blue')
    plt.plot(x1[0::10], y1[0::10], label='nn', linewidth=3) #syntax [0::10] plots every 10 events to give a smoother curve
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

