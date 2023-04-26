import os
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Reshape, Dense
from sklearn.model_selection import KFold

######################################################################################################################
"""
This script trains a specified number of networks using n-fold CV. Change the model parameters in training() for training a particula network.
This script should be used before test_cnn_5foldCV.py to get the trained models with the correct training/testing data sets.

load_data_kfold() creates the data set folds, trainnig() trains the models and saves them to the specified name,
and the main() function calls both the load_data_kfold() and training() functions
"""
######################################################################################################################
PathToARIANNA = os.environ['ARIANNA_Experiment']
n = np.load(PathToARIANNAData + '/data/noise.npy') #input a subset of the data here so that you can validate on the other set
s = np.load(PathToARIANNAData + '/data/signal.npy') #make sure the signal and noise subset of data are the same size
#actual model name is specified below when loading the model
model_path = PathToARIANNAData + '/models_h5_files/'
CV_num = 5 #this value can be changed to have the data split into more or less cross validiation groups
CV_size = 24000 #this gives the amount of data in each fold of the CV

if s.ndim==2: # for data of shape (event_num, samples), this reformats it into 3 dimensions
    s = np.reshape(s, (s.shape[0], 1, s.shape[1]))
    n = np.reshape(n, (n.shape[0], 1, n.shape[1]))

#this shuffles the data sets and then keeps track of the shuffle by saving as numpy file. This is needed to keep track of shuffled data in test_cnn_5foldCV.py.
# s = np.arange(signal_no_shuf.shape[0])
# np.random.shuffle(s)
# signal = signal_no_shuf[s]
# noise = noise_no_shuf[s]
# np.save('/Volumes/External/arianna_data/random_shuff_index_for_trimmed100_ABC.py',s)


def load_data_kfold(s, n, k, d_size):
    s_tot = np.zeros((k,d_size,100))
    n_tot = np.zeros((k,d_size,100))
    for i in range(k):
      s_tot[i] = s[(i)*d_size : (i + 1)*d_size]
    for i in range(k):
      n_tot[i] = n[(i)*d_size : (i + 1)*d_size]
    
    folds = list(KFold(n_splits=5).split(s_tot,n_tot))
    return folds, s_tot,n_tot

  
def training():
  n_samples = noise.shape[2]
  ch = noise.shape[1]
  BATCH_SIZE = 32
  EPOCHS = 150
  callbacks_list = [
  keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)]

  for j, (tr_idx,val_idx) in enumerate(Folds):
    print('\nFold ',j)
    s = signal[tr_idx] #(4,24k,100)
    s = np.reshape(s,(s.shape[0]*s.shape[1],s.shape[2],1,1))
    n = noise[tr_idx] #(4,24k,100)
    n = np.reshape(n,(n.shape[0]*n.shape[1],n.shape[2],1,1))
    x = np.vstack((n,s))
    y = np.vstack((np.zeros((n.shape[0], 1)), np.ones((s.shape[0], 1))))
  
    model = Sequential()
    model.add(Conv2D(5, (10, 1), activation='relu', input_shape=(n_samples * ch, 1, 1)))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(10, 1)))
    model.add(Reshape((np.prod(model.layers[-1].output_shape[1:]),)))  # equivalent to Flatten
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x,
              y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.2,
              verbose=1,
              callbacks=callbacks_list)

    model.save(model_path + f'trained_CNN_100samp_1L5-10_pat8_lr0.001_do0.5_5fold_24k_each_omitted_3rdtrial_datashuffdABC_{val_idx}.h5')


def main():
  Folds, signal, noise = load_data_kfold(s, n, CV_num, CV_size)
  training()

  
if __name__== "__main__":
    main()
