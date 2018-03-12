########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5
########################################

from HMM import *
from process_data import *
import matplotlib.pyplot as plt
import numpy as np
import re
from wordcloud import WordCloud
from matplotlib import animation
from matplotlib.animation import FuncAnimation

# Wordcloud functions
def mask():
    # Parameters.
    r = 128
    d = 2 * r + 1

    # Get points in a circle.
    y, x = np.ogrid[-r:d-r, -r:d-r]
    circle = (x**2 + y**2 <= r**2)

    # Create mask.
    mask = 255 * np.ones((d, d), dtype=np.uint8)
    mask[circle] = 0

    return mask

def text_to_wordcloud(text, max_words=50, title='', show=True):
    plt.close('all')

    # Generate a wordcloud image.
    wordcloud = WordCloud(random_state=0,
                          max_words=max_words,
                          background_color='white',
                          mask=mask()).generate(text)

    # Show the image.
    if show:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=24)
        plt.show()

    return wordcloud

def states_to_wordclouds(hmm, obs_map, max_words=50, show=True):
    # Initialize.
    M = 100000
    n_states = len(hmm.A)
    obs_map_r = obs_map_reverser(obs_map)
    wordclouds = []

    # Generate a large emission.
    emission, states = hmm.generate_emission(M)

    # For each state, get a list of observations that have been emitted
    # from that state.
    obs_count = []
    for i in range(n_states):
        obs_lst = np.array(emission)[np.where(np.array(states) == i)[0]]
        obs_count.append(obs_lst)

    # For each state, convert it into a wordcloud.
    for i in range(n_states):
        obs_lst = obs_count[i]
        sentence = [obs_map_r[j] for j in obs_lst]
        sentence_str = ' '.join(sentence)

        wordclouds.append(text_to_wordcloud(sentence_str, max_words=max_words, title='State %d' % i, show=show))

    return wordclouds

# HMM VISUALIZATION FUNCTION
####################

def visualize_sparsities(hmm, O_max_cols=50, O_vmax=0.1):
    plt.close('all')
    plt.set_cmap('viridis')

    # Visualize sparsity of A.
    plt.imshow(hmm.A, vmax=1.0)
    plt.colorbar()
    plt.title('Sparsity of A matrix')
    plt.show()

    # Visualize parsity of O.
    plt.imshow(np.array(hmm.O)[:, :O_max_cols], vmax=O_vmax, aspect='auto')
    plt.colorbar()
    plt.title('Sparsity of O matrix')
    plt.show()

def unsupervised_learning(n_states, N_iters, printPhrase=True, visSparsity=True):
    '''
    Trains an HMM using supervised learning on the file 'ron.txt' and
    prints the results.

    Arguments:
        n_states:   Number of hidden states that the HMM should have.
    '''
    #genres, genre_map = Utility.load_ron_hidden()

    #word_list, seqs, syllable_counts = basic_tokenized()
    word_list, seqs, syllable_counts, set_rhymes = advanced_tokenized()
    print('number unique words:', len(word_list))
    #print('first 5 lines:', seqs[:100])
    #print(syllable_counts['thy'])

    # Train the HMM.
    """
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of lists.
    """
    seqs.reverse()
    HMM = unsupervised_HMM(seqs, len(word_list), n_states, N_iters)
    
    # visualize the sparsities of the matrices 
    if visSparsity:
        visualize_sparsities(HMM, O_max_cols=50, O_vmax=0.1) 

    sonnet = []
    for i in range(14):
        sonnet.append(generate_phrase(HMM, word_list, syllable_counts, set_rhymes, printPhrase))

    #print(HMM.sonnet)

    return sonnet

def generate_phrase(HMM, word_list, syllable_counts, set_rhymes, printPhrase=True):

    # Naive Method
    #line_length = [8,9,10,11,12] # Account for punctuation in basic model
    #emission, lists = HMM.generate_emission(random.choice(line_length))

    # With syllable counting
    emission, lists = HMM.generate_line(word_list, syllable_counts, set_rhymes)
    #print(emission)

    line = []
    for i in range(len(emission)):
        aNum = emission[i]
        line.append(word_list[aNum])
    line.reverse()
    line[0] = line[0].capitalize()

    punctuation = ['.', '!', ',', '?', ':', ';', '(', ')','#']

    phrase = " ".join(line)
    for p in punctuation:
        phrase = phrase.replace(' ' + p, p)
        phrase = phrase.replace('#', '')

    if printPhrase:
        print(phrase)
    return phrase

    # Print the transition matrix.
    # print("Transition Matrix:")
    # print('#' * 70)
    # for i in range(len(HMM.A)):
    #     print(''.join("{:<12.3e}".format(HMM.A[i][j]) for j in range(len(HMM.A[i]))))
    # print('')
    # print('')
    #
    # # Print the observation matrix.
    # print("Observation Matrix:  ")
    # print('#' * 70)
    # for i in range(len(HMM.O)):
    #     print(''.join("{:<12.3e}".format(HMM.O[i][j]) for j in range(len(HMM.O[i]))))
    # print('')
    # print('')



if __name__ == '__main__':
    print('')
    print('')
    print('#' * 70)
    print("{:^70}".format("Running Code For Basic HMM"))
    print('#' * 70)
    print('')
    print('')
    
    # Then predict some stuff using the observation and transition matrices
    #print('HMM:', HMM)
    # do some data visualiztion

    sonnetStr = ''
    # trying it out with several generated poems so we get better word clouds    
    for i in range(2):
        sonnet = unsupervised_learning(7, 100, False, False)
        for phrase in sonnet:
            sonnetStr += phrase
            
    sonnet = unsupervised_learning(7, 100)
    
    for phrase in sonnet:
        sonnetStr += phrase
    
    text = open(os.path.join(os.getcwd(), 'data/shakespeare.txt')).read()
    #print('text', type(text), text)
    wordcloud = text_to_wordcloud(text, title='Original Sonnet Data')
    wordcloud1 = text_to_wordcloud(sonnetStr, title='Data from Several Generated Poem')
