import numpy as np
import os

def basic_tokenized():
    '''
    Arguments:
        (none)
        
    Return:
        word_dict   - A list of words where the index corresponding to a word
                      is its word index
        seqs_padded - A list of equal length lists where each sublist is a list
                      of word indices representing a line of Shakespeare. 
                      Note: an index of -1 corresponds to end-padding
    '''
    word_list = []
    seqs = []
    
    # Load data into sequences
    with os.path.join('data', 'shakespeare.txt') as f:
        for line in f:
            seqs.append([])
            raw = line.strip().split()
            # Skip lines that aren't actually part of the Sonnets
            if len(raw) < 2:
                continue
            # If we encounter a new word for the first time, add it to word_list
            for word in raw:
                if word not in word_list:
                    word_list.append(word)
                seqs[-1].append(word_list.index(word))
                
    # Pad sequences that are less than the actual length
    max_length = max([len(seq) for seq in seqs])
    seqs_padded = [seq + [-1] * (max_length - len(seq)) for seq in seqs]
    
    return word_list, seqs_padded
    
    
def int_to_onehot(n, n_max):
    # Return n encoded as a one-hot vector of length n_max.
    return [0 if n != i else 1 for i in range(n_max)]
    
def character_onehot(n=40, s=5):
    '''
    Arguments:
        n - the number of characters per example
        s - the spacing between successive examples
        
    Return:
        X - A list of examples of length n
        Y - A list of characters that immediately follow 
        Note: both X and Y are one-hot encoded ascii-values
    '''
    
    data = ''
    # Load lines of actual Shakespeare into a single string
    with open(os.path.join('data', 'shakespeare.txt')) as f:
        for line in f:
            if len(line.strip().split()) < 2:
                continue
            else:
                data += line
    
    X = []
    Y = []
    
    # to convert back, use ''.join(chr(i) for i in L), stackoverflow.com/questions/180606
    ascii_values = [ord(c) for c in data] 
    
    # Generate samples of size n where successive samples are shifted by s
    for i in range(len(data) // s):
        if s * i + n < len(data):
            X.append([int_to_onehot(a, 128) for a in ascii_values[s*i:s*i+n]])
            Y.append(int_to_onehot(ascii_values[s*i+n], 128))
        
    return np.array(X), np.array(Y)
    