import os
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D

######################################################################################################################
"""
This script trains a basic CNN and then plots the accuracy and validaiton accuracy on one plot and the loss and validation loss on one plot

The parameters to set are the path to input data, the noise and signal files, and the output name and path for the model that will be trained.

Here a model is trained on two dimensional data, and refer to nn_train_test_efficiency.py for training on data that is channels*samples=1 dimension
"""
######################################################################################################################
path = "/arianna_data"
noise = np.load(os.path.join(path, "noise.npy")) #input a subset of the data here so that you can validate on the other set
signal = np.load(os.path.join(path, "signal.npy")) #make sure the signal and noise subset of data are the same size
model_name = "trained_CNN_1l-10-8-10_do0.5_mp10_fltn_sigm" #name without .h5 at the end. Make sure to include relevant training info
model_path = "/h5_model_path"

# for the deep learning filter simulated data, this is how it is read in:
# signal = np.load(os.path.join(path, f"data_signal_{ch}ch_0000.npy"))
# # remove nans channels
# for i in range(signal.shape[1]):
#     mask = np.ones(signal.shape[0], dtype=np.bool)
#     for i in np.argwhere(np.sum(np.isnan(signal), axis=2)[:, i] > 0)[:, 0]:
#         mask[i] = False
#     signal = signal[mask]
# signal = signal[:, 0:ch, 8:-8]
# noise = np.zeros((600000, ch, 256), dtype=np.float32)
# for i in range(6):
#     noise[(i) * 100000:(i + 1) * 100000] = np.load(os.path.join(path, f"data_noise_{ch}ch_3.6SNR_{i:04d}.npy")).astype(np.float32)
# noise = noise[:, 0:ch, 8:-8]  # removing bins where data has stop glitch and only select channels
# if signal.ndim==2:
#   signal = np.reshape(signal, (signal.shape[0], 1, signal.shape[1]))
#   noise = np.reshape(noise, (noise.shape[0], 1, noise.shape[1]))

#make signal the same shape as the noise data, if needed
# signal = np.vstack((signal,signal,signal,signal))
# signal = signal[0:noise.shape[0]]

print(signal.shape)
print(noise.shape)

x = np.vstack((noise, signal))  
__, n_channels, n_samples = x.shape
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
keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)] #this stops the training whent he val loss doesn't decrease for 4 epochs

model = Sequential()
model.add(Conv2D(10, (8, 10), activation='relu', input_shape=(n_channels, n_samples, 1)))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(1, 10)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(x, y, validation_split=.1, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)  # training on the data
model.summary()

#print accuracy and loss plots after training:
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model 1chan, 64nodes')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.close()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model 1chan, 64nodes')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#input the path and file you'd like to save the model as (in h5 format)
model.save(f'{model_path}/{model_name}.h5')
