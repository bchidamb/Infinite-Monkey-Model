import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from process_data import *


def model(n_lstm, shape_in, n_out):
    '''
    Arguments:
        n_lstm  - the number of LSTM units in the first layer
        shape_in - the input shape of the data (2-tuple)
        n_out   - the number of categories in the output
    
    Return:
        model - A fresh Keras Sequential model for training on sequence data
    '''
    model = Sequential()
    model.add(LSTM(n_lstm, input_shape=shape_in))
    model.add(Dense(n_out, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    

def generate(model, n, seed):
    '''
    Arguments:
        model - a trained Keras Sequential model for sequence prediction
        n - the length of the generated sample (not including seed length)
        seed - the seed to initialize the model
    
    Return:
        gen - The generated string (including seed)
    '''
    def sample(pdf):
        return np.random.choice(np.arange(len(pdf)), p=np.array(pdf) / np.sum(pdf))
    
    gen = [int_to_onehot(ord(c), 128) for c in seed]
    window = len(seed)
    for _ in range(n):
        prev = np.array([gen[-window:]])
        gen.append(int_to_onehot(sample(model.predict(prev)[0]), 128))
        
    print([c for c in gen if 1 not in c])
    return ''.join([chr(c.index(1)) for c in gen if 1 in c])

    
X, Y = character_onehot(40, 5)
print('no. examples:', len(Y))

character_model = model(100, np.shape(X[0]), 128)
character_model.fit(X, Y, epochs=20, batch_size=5)

seed = "shall i compare thee to a summer’s day?\n"

print('Generating string...')
print(generate(character_model, 100, seed))
