import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from process_data import *
from time import time
from RNN import *


def wordpairs_model(vocab_size, latent_factors):
    '''
    Arguments:
        vocab_size -    The number of unique words in our list; also the input 
                        and output dimensions of this model
        num_latent_factors 
                   -    The number of dimensions used to represent each word
    
    Return:
        model - A fresh Keras Sequential model for training on word pairs
    '''
    model = Sequential()
    model.add(Dense(latent_factors, input_dim=vocab_size))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
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
    
    
def word2vec_repr(word, weights, latent_factors):
    '''
    Arguments:
        word - An integer representing the word being represented
        weights - The weights of the latent factor model. Dimension is
        vocab_size X latent_factors
    
    Return:
        The vector representation of this word. Represented as a Numpy array 
        of length (latent_factors + 1)
    '''
    return weights[word]

        
def generate2(model, n, seed, weights, latent_factors, word_list, temp=1.0):
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
    
    gen = [word2vec_repr(i, weights, latent_factors) for i in seed]
    gen_ids = seed[:]
    window = len(seed)
    for _ in range(n):
        prev = np.array([gen[-window:]])
        next = sample(model.predict(prev)[0])
        gen_ids.append(next)
        gen.append(word2vec_repr(next, weights, latent_factors))
        
    # print([c for c in gen if 1 not in c])
    return ' '.join([word_list[w] for w in gen_ids])
    
    

# Build latent factor representation
latent_factors = 100

word_list, seq = build_wordlist()
print(np.shape(word_list), np.shape(seq))
X_w2v, Y_w2v = wordpair_onehot(word_list, seq, s=3)
print(len(X_w2v), len(Y_w2v))

word2vec_model = wordpairs_model(len(word_list), latent_factors)
word2vec_model.fit(X_w2v, Y_w2v, epochs=15)

weights = word2vec_model.layers[0].get_weights()[0]
print(np.shape(weights))

# Train LSTM using embedded word representations (embedded word model)
ep = 10
bs = 100

X_rnn, Y_rnn = word_examples(word_list, seq, n=40, s=2, ds=0, onehot=False)
X_rnn_emb = np.array([[word2vec_repr(w, weights, latent_factors) for w in seq] for seq in X_rnn])
print(np.shape(X_rnn_emb))

word_model = multilayer_model(50, np.shape(X_rnn_emb[0]), len(word_list))
word_model.fit(X_rnn_emb, Y_rnn, epochs=ep, batch_size=bs, verbose=1)

n = 100
seed = seq[:40]
temps = [0.25, 0.75, 1.0, 1.5]

for temp in temps:
    print('Temperature', temp)
    print(generate2(word_model, n, seed, weights, latent_factors, word_list, temp=temp))

X_rnn_test, Y_rnn_test = word_examples(word_list, seq, n=40, s=2, ds=1, onehot=False)
X_rnn_emb_test = np.array([[word2vec_repr(w, weights, latent_factors) for w in seq] for seq in X_rnn_test])

print('Model 1 Train Perplexity score:', perplexity(word_model, X_rnn_emb, Y_rnn))
print('Model 1 Test Perplexity score:', perplexity(word_model, X_rnn_emb_test, Y_rnn_test))


# Train LSTM using onehot encoded words (naive word model)
ep = 10
bs = 100

word_list, seq = build_wordlist()
print(np.shape(word_list), np.shape(seq))

X_rnn, Y_rnn = word_examples(word_list, seq, n=40, s=2, ds=0, onehot=True)

word_model2 = multilayer_model(100, np.shape(X_rnn[0]), len(word_list))
word_model2.fit(X_rnn, Y_rnn, epochs=ep, batch_size=bs, verbose=1)

X_rnn_test, Y_rnn_test = word_examples(word_list, seq, n=40, s=2, ds=1, onehot=True)

print('Model 2 Train Perplexity score:', perplexity(word_model2, X_rnn, Y_rnn))
print('Model 2 Test Perplexity score:', perplexity(word_model2, X_rnn_test, Y_rnn_test))


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


ep = 125
bs = 100

# start = time()



# end = time()

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
'''

