
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import glob
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils
from sklearn.model_selection import KFold, train_test_split

path = "/Volumes/External/arianna_data/"
ch = 1

# signal = np.load(os.path.join(path, f"data_signal_{ch}ch_0000.npy"))
s1 = np.load(os.path.join(path, "trimmed100_data_signal_3.6SNR_1ch_0000.npy"))
s2 = np.load(os.path.join(path, "trimmed100_data_signal_3.6SNR_1ch_0001.npy"))
signal = np.vstack((s1,s2))

sshuf = np.load('/Volumes/External/arianna_data/random_shuff_index_for_trimmed100_signal_0and1_ABC.npy')
signal = signal[sshuf]

# s = np.arange(signal_no_shuf.shape[0])
# np.random.shuffle(s)
# signal = signal_no_shuf[s]
# np.save('/Volumes/External/arianna_data/random_shuff_index_for_trimmed100_signal_0and1_ABC.py',s)


noise = np.zeros((200000, 100), dtype=np.float32)
for i in range(2):
  noise[(i) * 100000:(i + 1) * 100000] = np.load(os.path.join(path, f"trimmed100_data_noise_3.6SNR_1ch_{i:04d}.npy")).astype(np.float32)
nshuf = np.load('/Volumes/External/arianna_data/random_shuff_index_for_trimmed100_noise_0and1_ABC.npy')
noise = noise[nshuf]

# s = np.arange(noise_no_shuf.shape[0])
# np.random.shuffle(s)
# noise = noise_no_shuf[s]
# np.save('/Volumes/External/arianna_data/random_shuff_index_for_trimmed100_noise_0and1_ABC.py',s)


def load_data_kfold(k,d_size):
    s_tot = np.zeros((k,d_size,100))
    n_tot = np.zeros((k,d_size,100))
    for i in range(k):
      s_tot[i] = signal[(i)*d_size : (i + 1)*d_size]
    for i in range(k):
      n_tot[i] = noise[(i)*d_size : (i + 1)*d_size]
    
    folds = list(KFold(n_splits=5).split(s_tot,n_tot))
    return folds, s_tot,n_tot
Folds, signal, noise = load_data_kfold(5,24000)

n_samples = noise.shape[2]
BATCH_SIZE = 32
EPOCHS = 100
callbacks_list = [
keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)]

print(signal.shape)
print(noise.shape)

def training():

  for j, (tr_idx,val_idx) in enumerate(Folds):
    print('\nFold ',j)

    s = signal[tr_idx] #(4,24k,100)
    s = np.reshape(s,(s.shape[0]*s.shape[1],s.shape[2],1,1))
    n = noise[tr_idx] #(4,24k,100)
    n = np.reshape(n,(n.shape[0]*n.shape[1],n.shape[2],1,1))

    x = np.vstack((n,s))
    y = np.vstack((np.zeros((n.shape[0], 1)), np.ones((s.shape[0], 1))))
  
    print(x.shape,y.shape)
    model = Sequential()
    model.add(Conv2D(5, (10, 1), activation='relu', input_shape=(n_samples * ch, 1, 1)))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(10, 1)))
    model.add(Reshape((np.prod(model.layers[-1].output_shape[1:]),)))  # equivalent to Flatten
    model.add(Dense(1, activation='sigmoid'))
    # opt = keras.optimizers.Adam()  # default 0.001
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x,
              y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.2,
              verbose=1,
              callbacks=callbacks_list)

    # model.summary()
    model.save(f'/Volumes/External/ML_paper/5fold_cross_val/trainings_pat8/trained_CNN_100samp_1L5-10_pat8_lr0.001_do0.5_5fold_24k_each_omitted_3rdtrial_datashuffdABC_{val_idx}.h5')


training()
