########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5
########################################

from HMM import *
from process_data import *

def unsupervised_learning(n_states, N_iters):
    '''
    Trains an HMM using supervised learning on the file 'ron.txt' and
    prints the results.

    Arguments:
        n_states:   Number of hidden states that the HMM should have.
    '''
    #genres, genre_map = Utility.load_ron_hidden()

    word_list, seqs, syllable_counts = basic_tokenized()
    word_list, seqs, syllable_counts = advanced_tokenized()
    print('number unique words:', len(word_list))
    #print('first 5 lines:', seqs[:100])
    #print(syllable_counts['thy'])

    # Train the HMM.
    """
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of lists.
    """
    HMM = unsupervised_HMM(seqs, len(word_list), n_states, N_iters)

    for i in range(14):
        generate_phrase(HMM, word_list, syllable_counts)


def generate_phrase(HMM, word_list, syllable_counts):

    # Naive Method
    #emission, lists = HMM.generate_emission(10)

    # With syllable counting
    emission, lists = HMM.generate_line(word_list, syllable_counts)
    #print(emission)

    line = []
    for aNum in emission:
        line.append(word_list[aNum])
    phrase = " ".join(line)
    print(phrase)

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

    HMM = unsupervised_learning(4, 100)
    # Then predict some stuff using the observation and transition matrices
