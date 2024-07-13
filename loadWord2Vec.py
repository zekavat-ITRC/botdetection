from __future__ import unicode_literals
import os
import pickle


import numpy as np
import gensim

import numpy as np
from glove import Glove


def loadModel():
    model = gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format('data\\insta_wchr.vec')
    num_features = model.vector_size
    #print(num_features)
    model_word_set = set(model.index2word)
    #print('Finished loading model')
    return model, num_features, model_word_set

def loadModel_glove():
    model = Glove.load('data\\instaGlove100.model')
  #  model = gensim.models.fasttext.FastText.load_fasttext_format('data\\instaGlove100.model')

    model_word_dict=model.dictionary
    model_word_set = list(model_word_dict.keys())

    #print(model.max_count)
    num_features = int(model.max_count)
    print(num_features)
    print('Finished loading model')
    return model, num_features, model_word_set, model.word_vectors

def make_agg_vec(words, model, num_features, model_word_set, filter_out=[]):
    """Create aggregate representation of list of words"""

    feature_vec = np.zeros((num_features,), dtype="float32")

    nwords = 0.

    for word in words:
        #if word not in filter_out:
            if word in model_word_set:
                nwords += 1
                feature_vec = np.add(feature_vec, model[word])
    #print(nwords)
    avg_feature_vec = feature_vec / nwords

    return avg_feature_vec

def make_agg_vec_glove(words, model, num_features, model_word_set, word_vec,filter_out=[]):
    """Create aggregate representation of list of words"""

    feature_vec = np.zeros((num_features,), dtype="float32")

    nwords = 0.

    for word in words:
        #if word not in filter_out:
            if word in model_word_set:
                nwords += 1
                feature_vec = np.add(feature_vec, word_vec[model.dictionary[word]])
   # print(nwords)
    avg_feature_vec = feature_vec / nwords

    return avg_feature_vec

#loadModel()
