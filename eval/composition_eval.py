# -*- coding: utf8 -*-
from composes.semantic_space.space import Space
from composes.utils import io_utils
from composes.similarity.cos import CosSimilarity
from composes.transformation.scaling.row_normalization import RowNormalization

import codecs
import sys, os
import numpy as np
import operator
import logging
import unittest
from scipy.spatial.distance import cdist
from pprint import pprint as pp

logging.getLogger("composes").setLevel(logging.WARNING)

def inspect_representations(path_composed_emb, output_path):
    print('Inspecting representations...')
    composed_space = Space.build(data=path_composed_emb, format='dm')
    f = codecs.open(output_path, 'w', 'utf8')
    word_list=[w for w in composed_space.get_row2id()]
    for j, w in enumerate(word_list):
        if j < 1000:
            neighbours = composed_space.get_neighbours(w, 10, CosSimilarity())

            f.write('Neighbours for ' + w + '\n')
            f.write("\n".join('%s %.6f' % x for x in neighbours))
            f.write('\n----------------------------\n')
    f.close()

def computeQuartiles(l):
    """Return sample quartiles.
    Method 1 from https://en.wikipedia.org/wiki/Quartile:
    Use the median to divide the ordered data set into two halves. Do not include the median in either half.
    The lower quartile value is the median of the lower half of the data. The upper quartile value is the median of the upper half of the data.
    """

    med = np.median(l)
    sortedL = np.sort(l)
    n = len(l)
    midIndex = n/2
    q1_list = sortedL[0: midIndex]
    if (n%2 == 0):
        q3_list = sortedL[midIndex:]
    else:
        q3_list = sortedL[midIndex+1:]
    q1 = np.median(q1_list)
    q3 = np.median(q3_list)
    return q1, med, q3

def computeRanks(composedSpace, observedSpace):
    """Ranks all the representations in the composed space with respect to 
    the representations in the observed space. Cut-off value 1000"
    """
    ranks = {}
    rankList = []

    composedWords = set(composedSpace.get_id2row())
    observedWords = observedSpace.get_id2row()
    neighbours = 1000

    for w_idx, word in enumerate(composedWords):
        vector = composedSpace.get_row(word)
        Y = 1 - cdist(vector.mat, observedSpace.get_cooccurrence_matrix().mat, 'cosine')
        nearest = Y.argmax()
        nearest_k_indices = np.argpartition(Y, tuple([-p for p in range(neighbours)]), axis=None)[-neighbours:]
        # pp([(observedWords[idx], Y[0][idx]) for idx in reversed(nearest_k_indices)])
        words = [observedWords[idx] for idx in reversed(nearest_k_indices)]
        wordRanks = {word:index+1 for index,word in enumerate(words)}
        # print(wordRanks)

        if (word in wordRanks):
            r = wordRanks[word]
            ranks[word] = r
            rankList.append(r)

        else:
            ranks[word] = 1000
            rankList.append(1000)

        if ((w_idx > 0) and (w_idx % 100 == 0)):
            print(w_idx)

    return rankList, ranks

def evaluateRank(wList, composedSpace, initialSpace):
    rankList, ranks = computeRanks(composedSpace, initialSpace)
    q1, q2, q3 = computeQuartiles(rankList)
    return q1, q2, q3, ranks

def printDictToFile(dictionary, fileName):
    outDict = codecs.open(fileName, mode='w', encoding='utf8')
    sorted_x = sorted(dictionary.items(), key=operator.itemgetter(1))
    for key, value in sorted_x:
        outDict.write("%s %d\n" % (key.decode('utf8'), value))
    outDict.close()

def printListToFile(save_list, fileName):
    out = codecs.open(fileName, mode='w', encoding='utf8')
    for value in save_list:
        out.write("%d\n" % (value))
    out.close()

def logResult(q1,q2,q3, filename):
    out = codecs.open(filename, mode='w', encoding='utf8')
    out.write("Q1: %d, Q2: %d, Q3: %d\n" % (q1, q2, q3))
    out.close()

def eval_on_file(path_composed_emb, path_observed_emb, save_path):
    raw_observed_space = Space.build(data=path_observed_emb, format='dm')
    observed_space = raw_observed_space.apply(RowNormalization('length'))
    observed_words = observed_space.get_id2row()
    print("Observed words, size: " + str(len(observed_words)) + ", first:")
    print(observed_words[:10])
    observed_words_set = set(observed_words)

    raw_composed_space = Space.build(data=path_composed_emb, format='dm')
    composed_space = raw_composed_space.apply(RowNormalization('length'))
    composed_words = composed_space.get_id2row()
    print("Composed words, size: " + str(len(composed_words)) + ", first:")
    print(composed_words[:10])

    # all composed words should be in the initial space
    for idx, word in enumerate(composed_words):
        assert(word in observed_words_set)

    q1, q2, q3, ranks = evaluateRank(composed_words, composed_space, observed_space)
    print("Q1: " + str(q1) + ", Q2: " + str(q2) + ", Q3: " + str(q3))

    printDictToFile(ranks, save_path + '_rankedCompounds.txt')
    
    sortedRanks = sorted(ranks.values())
    printListToFile(sortedRanks, save_path + '_ranks.txt')
    logResult(q1, q2, q3, save_path + '_quartiles.txt')

    return q1,q2,q3,ranks


class Test_evaluateRank(unittest.TestCase):
    def test_eval_on_file(self):
        path_observed = '/Users/corina/Work/git_own/gWordcomp/data/english_compounds_composition_dataset/embeddings/'+\
            'glove_encow14ax_enwiki_8B.400k_l2norm_axis01/glove_encow14ax_enwiki_8B.400k.l2norm_axis01.300d_cmh.dm'
        path_composed = '/Users/corina/Work/git_own/gWordcomp/acl_paper_models_submission/fullAdd_composition/'+\
            'model_FullAdd_tanh_adagrad_batch100_mse_2016-05-09_18-36_dev.pred'
        save_path = '/Users/corina/Work/git_own/gWordcomp/acl_paper_models_submission/fullAdd_composition/'+\
            'model_FullAdd_tanh_adagrad_batch100_mse_2016-05-09_18-36_dev_normalized'
        eval_on_file(path_composed, path_observed, save_path)
        
if __name__=="__main__":
    eval_on_file(sys.argv[1], sys.argv[2], sys.argv[3])
    # unittest.main()


