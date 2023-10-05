import json
import collections
import argparse
import random
import numpy as np

from util import *

random.seed(42)

def extract_unigram_features(ex):
    """Return unigrams in the hypothesis and the premise.
    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
    Returns:
        A dictionary of BoW featurs of x.
    Example:
        "I love it", "I hate it" --> {"I":2, "it":2, "hate":1, "love":1}
    """
    # BEGIN_YOUR_CODE
    target = ex['sentence1'] + ex['sentence2']
    return collections.Counter(target)
    # END_YOUR_CODE

def extract_custom_features(ex):
    """Design your own features.
    """
    # BEGIN_YOUR_CODE
    remove = ['.', ',', '?', '!']
    sentence1 = [x.lower() for x in ex['sentence1']]
    sentence1 = [x for x in sentence1 if x not in remove]
    sentence2 = [x.lower() for x in ex['sentence2']]
    sentence2 = [x for x in sentence2 if x not in remove]
    target = sentence1 + sentence2
    return collections.Counter(target)
    # END_YOUR_CODE

def learn_predictor(train_data, valid_data, feature_extractor, learning_rate, num_epochs):
    """Running SGD on training examples using the logistic loss.
    You may want to evaluate the error on training and dev example after each epoch.
    Take a look at the functions predict and evaluate_predictor in util.py,
    which will be useful for your implementation.
    Parameters:
        train_data : [{gold_label: {0,1}, sentence1: [str], sentence2: [str]}]
        valid_data : same as train_data
        feature_extractor : function
            data (dict) --> feature vector (dict)
        learning_rate : float
        num_epochs : int
    Returns:
        weights : dict
            feature name (str) : weight (float)
    """
    # BEGIN_YOUR_CODE
    
    #preprocess data
    y_train = np.array([_['gold_label'] for _ in train_data])
    X_train = np.array([feature_extractor(_) for _ in train_data])
    
    #SGD
    w = dict()
    best_val_error = 1
    c = 0
    for _ in range(num_epochs):
        perm = np.random.permutation(y_train.shape[0])
        for i in perm:
            pred = predict(w, X_train[i])
            increment(w, X_train[i], -learning_rate*(pred-y_train[i]))
    
        predictor = lambda ex: 1 if dot(w, feature_extractor(ex)) > 0 else 0
        valid_err = evaluate_predictor([(ex, ex['gold_label']) for ex in valid_data], predictor)
        if valid_err < best_val_error:
            best_val_error = valid_err
            best_w = w
        else:
            c += 1
        if c == 3:
            break
    return best_w
    # END_YOUR_CODE

def count_cooccur_matrix(tokens, window_size=4):
    """Compute the co-occurrence matrix given a sequence of tokens.
    For each word, n words before and n words after it are its co-occurring neighbors.
    For example, given the tokens "in for a penny , in for a pound",
    the neighbors of "penny" given a window size of 2 are "for", "a", ",", "in".
    Parameters:
        tokens : [str]
        window_size : int
    Returns:
        word2ind : dict
            word (str) : index (int)
        co_mat : np.array
            co_mat[i][j] should contain the co-occurrence counts of the words indexed by i and j according to the dictionary word2ind.
    """
    # BEGIN_YOUR_CODE
    tokens = [word.lower() for word in tokens]
    uniq = set(tokens)
    word2ind = dict()
    word2ind = {word: i for i, word in enumerate(uniq)}
    tokens = [word2ind[t] for t in tokens]

    co_mat = np.zeros((len(uniq), len(uniq)))
    for i in range(len(tokens)):
        for j in range(max(0, i-window_size), min(len(tokens), i+window_size+1)):
            if i != j:
                co_mat[tokens[i]][tokens[j]] += 1
    
    return word2ind, co_mat
    # END_YOUR_CODE

def cooccur_to_embedding(co_mat, embed_size=50):
    """Convert the co-occurrence matrix to word embedding using truncated SVD. Use the np.linalg.svd function.
    Parameters:
        co_mat : np.array
            vocab size x vocab size
        embed_size : int
    Returns:
        embeddings : np.array
            vocab_size x embed_size
    """
    # BEGIN_YOUR_CODE
    U, S, V_t = np.linalg.svd(co_mat, full_matrices=False)

    U_trunc = U[:, :embed_size]
    S_trunc = np.diag(S[:embed_size])

    embeddings = np.dot(U_trunc, S_trunc)
    return embeddings
    # END_YOUR_CODE

def top_k_similar(word_ind, embeddings, word2ind, k=10, metric='dot'):
    """Return the top k most similar words to the given word (excluding itself).
    You will implement two similarity functions.
    If metric='dot', use the dot product.
    If metric='cosine', use the cosine similarity.
    Parameters:
        word_ind : int
            index of the word (for which we will find the similar words)
        embeddings : np.array
            vocab_size x embed_size
        word2ind : dict
        k : int
            number of words to return (excluding self)
        metric : 'dot' or 'cosine'
    Returns:
        topk-words : [str]
    """
    # BEGIN_YOUR_CODE
    word_embed = embeddings[word_ind]

    if metric == 'dot':
        score = np.dot(embeddings, word_embed)
    elif metric == 'cosine':
        score = np.dot(embeddings, word_embed)/ (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(word_embed))
    top_k = np.argsort(score)[::-1][1:k+1]

    ind2word = {v:k for k, v in word2ind.items()}

    return [ind2word[i] for i in top_k]
    # END_YOUR_CODE
