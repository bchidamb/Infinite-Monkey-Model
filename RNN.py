import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from process_data import *
from time import time


def base_model(n_lstm, shape_in, n_out):
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
    
    
def multilayer_model(n_lstm, shape_in, n_out):
    '''
    Arguments:
        n_lstm  - the number of LSTM units in each hidden layer
        shape_in - the input shape of the data (2-tuple)
        n_out   - the number of categories in the output
    
    Return:
        model - A fresh Keras Sequential model for training on sequence data
        Note: this model has 3 hidden LSTM layers
    '''
    model = Sequential()
    model.add(LSTM(n_lstm, input_shape=shape_in, return_sequences=True))
    model.add(LSTM(n_lstm, return_sequences=True))
    model.add(LSTM(n_lstm))
    model.add(Dense(n_out, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    

def generate(model, n, seed, temp=1.0):
    '''
    Arguments:
        model - a trained Keras Sequential model for sequence prediction
        n - the length of the generated sample (not including seed length)
        seed - the seed to initialize the model
        temp - temperature parameter which controls the variance of the output
    
    Return:
        gen - The generated string (including seed)
    '''
    def sample(pdf):
        exp_pdf = np.exp(np.log(pdf) / temp)
        return np.random.choice(np.arange(len(pdf)), p=np.array(exp_pdf) / np.sum(exp_pdf))
    
    gen = [int_to_onehot(ord(c), 128) for c in seed]
    window = len(seed)
    for _ in range(n):
        prev = np.array([gen[-window:]])
        gen.append(int_to_onehot(sample(model.predict(prev)[0]), 128))
        
    # print([c for c in gen if 1 not in c])
    return ''.join([chr(c.index(1)) for c in gen if 1 in c])

    
def perplexity(model, X, Y):
    '''
    Arguments:
        model - the trained Keras Sequential model to be evaluated
        X, Y - a test-set of onehot encoded sequences
    
    Return:
        the perplexity metric on this dataset, defined as
        PP(w) = (product_i of P(w_i)) ^ (-1/N)
    '''
    P = np.array([max(row) for row in (model.predict(X) * Y) if max(row) > 0.0])
    return np.prod(P ** (-1 / len(P)))
    
    
X, Y = character_onehot(40, 3)
X_test, Y_test = character_onehot(40, 3, 1)
print('no. examples:', len(Y))

'''
Epochs = [200, 500, 1000, 2000]
Batch_sizes = [200, 500, 1000, 2000]

for ep, bs in zip(Epochs, Batch_sizes):

    start = time()

    character_model = model(100, np.shape(X[0]), 128)
    character_model.fit(X, Y, epochs=ep, batch_size=bs, verbose=0)
    
    end = time()

    print('Configuration: %d epochs, %d batch size' % (ep, bs))
    print('Training time: %.3f' % (end - start))
    print('Perplexity score:', perplexity(character_model, X, Y), '\n')

'''
ep = 125
bs = 100

start = time()

character_model = multilayer_model(100, np.shape(X[0]), 128)
character_model.fit(X, Y, epochs=ep, batch_size=bs, verbose=1)

end = time()

print('Configuration: %d epochs, %d batch size' % (ep, bs))
print('Training time: %.3f' % (end - start))
print('Train Perplexity score:', perplexity(character_model, X, Y))
print('Test Perplexity score:', perplexity(character_model, X_test, Y_test), '\n')

seed = "shall i compare thee to a summer’s day?\n"
temps = [0.25, 0.75, 1.0, 1.5]

print('Generating strings...')
for t in temps:
    print('Temperature', t)
    print(generate(character_model, 500, seed, temp=t))


