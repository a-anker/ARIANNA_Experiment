from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import numpy as np
import sherpa # pip install parameter-sherpa

#refer to https://parameter-sherpa.readthedocs.io/en/latest/ for more info on this python package

def main():
    noise = np.load(f'/arianna_data/noise.py')
    signal = np.load(f'/arianna_data/signal.py')

    print(signal.shape)
    print(noise.shape)

    x = np.vstack((noise, signal)) 
    n_samples = x.shape[2]
    n_channels = x.shape[1]
    x = np.expand_dims(x, axis=-1)
    y = np.vstack((np.zeros((noise.shape[0], 1)), np.ones((signal.shape[0], 1))))
    r = np.arange(x.shape[0])
    np.random.shuffle(r)
    x = x[r]
    y = y[r]
    print(x.shape)

    ###sherpa###
    """in this example, there is a baseline CNN chosen with 2 hidden layers.
    The first layer is scanned over a varying kernel size from 5-100 and number of kernels from 1-20. 
    The second hidden layer is scanned over 5-50 kernel size and 1-10 kernels. 
    Lastly, the amount of epochs are varied between 5-100. """
    
    # there are more options for adding different kinds of parameters such as Continuous values, but Discrete should be used mainly
    parameters = [sherpa.Discrete('units1', [5, 100]),
    sherpa.Discrete('units2', [5, 50]),
    sherpa.Discrete(name='epochs', range=[5,100]),
    sherpa.Discrete(name='filts1', range=[1,20]),
    sherpa.Discrete(name='filts2', range=[1,10])]
    
    """alg sets how many Random iterations will be done between the parameters set above and how each parameter will be chosen, randomly in this case. 
    This is a good starting search technique especially when varying many parameters at once, but for 1-2 varying parameters, GridSearch is also a useful function."""
    alg = sherpa.algorithms.RandomSearch(max_num_trials=50)
    study = sherpa.Study(parameters=parameters,
                         algorithm=alg,
                         lower_is_better=True)

    for trial in study: 
        model = Sequential()
        model.add(Conv2D(trial.parameters['filts1'], (n_channels, trial.parameters['units1']), activation='relu', input_shape=(n_channels, n_samples, 1)))
        model.add(Conv2D(trial.parameters['filts2'], (1, trial.parameters['units2']), activation='relu', input_shape=(n_channels, n_samples, 1)))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])

        model.fit(x, y, epochs=trial.parameters['epochs'], batch_size=32,validation_split=0.2,
                  callbacks=[study.keras_callback(trial, objective_name='val_loss')],verbose=0)
        study.finalize(trial)
        print(study.get_best_result())
    study.save('/arianna_sherpa_study')
    print(study.get_best_result())

    
if __name__== "__main__":
    main()
