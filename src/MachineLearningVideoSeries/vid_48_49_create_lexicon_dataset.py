# So in order for tensor flow to work with this text, we need some way
# to translate each sentence into a fixed array. This can be done by
# creating a lexicon, this is a binary representation of the array.
'''
[chair, table, spoon, television]
I pulled the chair up from the table

[1 1 0 0]

So there we can have a
'''
import random

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 100000000


def create_lexacon(pos, neg):
    lexicon = []
    for file in [pos, neg]:
        with open(file, 'r') as f:
            contents = f.readlines()
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    # what it looks like: w_counts = {'the':43543, 'and':5466}

    l2 = []
    for w in w_counts:
        if 1000 > w_counts[w] > 30:
            l2.append(w)

    return l2


def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1

            features = list(features)
            featureset.append([features, classification])

    return featureset


def create_featuresets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexacon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1, 0])
    features += sample_handling('neg.txt', lexicon, [0, 1])
    random.shuffle(features)
#hi hhhhh
    # does tf.argmax([output]) == tf.argmax([expectations])
    # does tf.argmax([65463,654365]) == tf.argmax([1,0])
    # the goal is this:
    # does tf.argmax([99999999999999,000000000]) == tf.argmax([1,0])

    features = np.array(features)
    testing_size = int(test_size * len(features))

    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])

    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y

if __name__== '__main__':
    train_x, train_y, test_x, test_y = create_featuresets_and_labels('pos.txt','neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y ], f)


