import numpy as np
import os
import copy

def basic_tokenized():
    '''
    Counts the number of unique words in a given input text
    And converts the text into sequences of numbers that correspond to words

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
    f = open('data/shakespeare.txt', 'r')
    #with os.path.join('data', 'shakespeare.txt') as f:
    for line in f:

        raw = line.strip().split()
        # Skip lines that aren't actually part of the Sonnets
        if len(raw) < 2:
            continue
            # If we encounter a new word for the first time, add it to word_list

        seqs.append([])
        for word in raw:
            if word not in word_list:
                word_list.append(word)
            seqs[-1].append(word_list.index(word))

    #f.close()

    # Create a list of words and their possible syllable counts
    f2 = open('data/Syllable_dictionary.txt', 'r')
    syllable_counts = {}
    for line in f2:
        raw = line.strip().split()
        #word_index = word_list.index(raw[0])
        syllable_counts[raw[0]] = raw[1:]

    print('len(word_list)', len(word_list), word_list[0])
    print('len(seqs)', len(seqs), seqs[0])

    return word_list, seqs, syllable_counts

def stripPunctuation(string):
    lst = ['.', '!', ',', '?',':', ';', '(', ')']
    string1 = ''
    for s in string:
        if s not in lst:
            string1 += s

    return string1

# account for rhymes 
def advanced_tokenized():
    '''
    Counts the number of unique words in a given input text
    And converts the text into sequences of numbers that correspond to words

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
    list_rhymes = [] # list of lists, such that each list has rhymes
    # indices for the rhyme scheme that shakespeare uses:
    # abab, cdcd, efef, gg
    rhymeInd = [[0,2], [1,3], [4,6], [5,7], [8,10], [9,11], [12,13]]

    # Load data into sequences
    f = open('data/shakespeare.txt', 'r')
    #with os.path.join('data', 'shakespeare.txt') as f:
    countLine = 0 # count which line we are on for each sonnet
    lastWordsLst = [] # get the list of last words in each line

    for line in f:

        raw = stripPunctuation(line.strip()).split()
        #print('raw:', raw)
        # Skip lines that aren't actually part of the Sonnets
        if len(raw) < 2:
            # if we get enough lines for our sonnet:
            if countLine == 14:
                rhymesLine = []
                # make a list of the rhymes we found in our sonnet
                for pair in rhymeInd:
                    #print('pair', pair)
                    #print('lastWordsLst', len(lastWordsLst), lastWordsLst)
                    rhymesLine.append([lastWordsLst[pair[0]], lastWordsLst[pair[1]]])

                # go through all of the rhymes that we got from the current sonnet
                # and add them to the total number of rhymes
                list_rhymes, rhymesLine = joinRhymeFamily(list_rhymes, rhymesLine)

                # go through the list of rhymes and consolidate the rhyme families if
                # several different families are actually just the same
                list_rhymes = consolidateRhymeFamily(list_rhymes)

            countLine = 0
            lastWordsLst = []
            #print('rhyme scheme assimilated!')
            continue

        # If we encounter a new word for the first time, add it to word_list
        seqs.append([])
        for word in raw:
            if word not in word_list:
                word_list.append(word)
            seqs[-1].append(word_list.index(word))

        #print('last word seq', seqs[-1][-1])
        lastWordsLst.append(seqs[-1][-1]) # append the last word of the sequence to lastWordLst
        #print('lastWordsLst update', lastWordsLst)
        countLine += 1

    #f.close()

    # Create a list of words and their possible syllable counts
    f2 = open('data/Syllable_dictionary.txt', 'r')
    syllable_counts = {}
    for line in f2:
        raw = line.strip().split()
        #word_index = word_list.index(raw[0])
        syllable_counts[raw[0]] = raw[1:]

    #print('len(word_list)', len(word_list), word_list[0])
    #print('len(seqs)', len(seqs), seqs[0])
    print('list of rhymes:', len(list_rhymes))
    for line in list_rhymes:
        print([word_list[x] for x in line])
        print(line)

    return word_list, seqs, syllable_counts


# go through all of the rhymes that we've found so far total and add our rhymes to them
# we're adding rhymes to list_rhymes
def joinRhymeFamily(list_rhymes, rhymesLine):
    
    for rhymeLocal in rhymesLine:
        flagRhyme = True # if we didn't find a rhyme family
        for rhymeTotal in list_rhymes:
            if rhymeLocal[0] in rhymeTotal and rhymeLocal[1] not in rhymeTotal:
                rhymeTotal.append(rhymeLocal[1])
                flagRhyme = False
                
            elif rhymeLocal[1] in rhymeTotal and rhymeLocal[0] not in rhymeTotal:
                rhymeTotal.append(rhymeLocal[0])
                flagRhyme = False

            elif rhymeLocal[1] in rhymeTotal and rhymeLocal[0] in rhymeTotal:
                flagRhyme = False

        #print('rhymesLine', len(rhymesLine))
        #print('list_rhymes', len(list_rhymes))
        # if we don't find a rhyme family, start a new one
        if flagRhyme:
            list_rhymes.append(rhymeLocal)

    #print('joined rhyme fam')
    return list_rhymes, rhymesLine

# the point of this is to make sure that we consolidate all of 
# the rhyme families in discovery of new similar rhymes
# check to make sure that we don't have two different lists 
# with the same rhyme family
def consolidateRhymeFamily(list_rhymes):

    tempList_rhymes = []
    # keeps track of what indices we've been through in 
    # list_rhymes and where they are in tempList_rhymes
    indicDict = {} 
    for i in range(len(list_rhymes) -1):
        for j in range(1, len(list_rhymes)):
            flagGetOuti = False
            for k in range(len(list_rhymes[i])):
                # if i has already been added to tempList get out
                if flagGetOuti:
                    break
                if list_rhymes[i][k] in list_rhymes[j]:
                    flagGetOuti = True
                    # new element in tempList_rhymes
                    if i not in indicDict and j not in indicDict:
                        #diff_sets = set(list_rhymes[i]) - set(list_rhymes[j])
                        uniqueEl = list(set(list_rhymes[i] + list_rhymes[j]))
                        tempList_rhymes.append(uniqueEl)
                        indicDict[i] = len(tempList_rhymes) -1
                        indicDict[j] = len(tempList_rhymes) -1
                        if len(uniqueEl) != len(set(uniqueEl)):
                            print('1 repeats occurred here')
                            print('uniqueEl', uniqueEl)

                    # updating element in tempList_rhymes, with j
                    elif i in indicDict and j not in indicDict:
                        #diff_sets = set(tempList_rhymes[indicDict[i]]) - set(list_rhymes[j])
                        uniqueEl = list(set(tempList_rhymes[indicDict[i]] + list_rhymes[j]))
                        tempList_rhymes[indicDict[i]] = uniqueEl
                        indicDict[j] = indicDict[i]
                        if len(uniqueEl) != len(set(uniqueEl)):
                            print('2 repeats occurred here')
                            print('uniqueEl', uniqueEl)
                        
                    # updating element in tempList_rhymes, with i
                    elif j in indicDict and i not in indicDict:
                        diff_sets = set(tempList_rhymes[indicDict[j]]) - set(list_rhymes[i])
                        uniqueEl = tempList_rhymes[indicDict[j]] + list(diff_sets)
                        tempList_rhymes[indicDict[j]] = uniqueEl
                        indicDict[i] = indicDict[j]
                        if len(uniqueEl) != len(set(uniqueEl)):
                            print('3 repeats occurred here')

    #print('tempList_rhymes:')
    #for lst in tempList_rhymes:
    #    print(lst)
    #    if len(lst) is not len(set(lst)):
    #        print('repeats occurred here')

    # set list_rhymes such that all the rhymes are together and consolidated
    list_rhymes = copy.deepcopy(tempList_rhymes)
    #print('consolidated rhymes')
    return list_rhymes

def int_to_onehot(n, n_max):
    # Return n encoded as a one-hot vector of length n_max.
    return [0 if n != i else 1 for i in range(n_max)]

def character_onehot(n=40, s=5, ds=0):
    '''
    Arguments:
        n - the number of characters per example
        s - the spacing between successive examples
        ds - offset of each example (change from default to obtain unique validation sets)
        Note: 0 <= ds < s

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
        if s * i + n + ds < len(data):
            X.append([int_to_onehot(a, 128) for a in ascii_values[s*i+ds:s*i+n+ds]])
            Y.append(int_to_onehot(ascii_values[s*i+n+ds], 128))

    return np.array(X), np.array(Y)
