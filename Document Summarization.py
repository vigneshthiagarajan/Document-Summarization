# Databricks notebook source
# MAGIC %sh
# MAGIC curl -O https://projecttextsummarization.s3.amazonaws.com/Apple.txt

# COMMAND ----------

# MAGIC %sh
# MAGIC curl -O https://projecttextsummarization.s3.amazonaws.com/cosine_similarity_input_final_excel.txt

# COMMAND ----------

dir = '/tmp'

# COMMAND ----------

dbutils.fs.cp('file:/databricks/driver/Apple.txt', dir, recurse=True)
dbutils.fs.cp('file:/databricks/driver/cosine_similarity_input_final_excel.txt', dir, recurse=True)

# COMMAND ----------

# apple store
#  /FileStore/tables/Apple.txt
# terry collins
# /FileStore/tables/OneArticle.txt

# COMMAND ----------

import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)

# COMMAND ----------

import nltk
from nltk import tokenize
from nltk.cluster.util import cosine_distance
import numpy
import operator

# COMMAND ----------

nltk.download('all')

# COMMAND ----------

file = open('/dbfs/tmp/Apple.txt')
article = file.read()
print(article)
article_sentences = tokenize.sent_tokenize(article)

# COMMAND ----------

# stop words in english -  /FileStore/tables/english_stopwords.txt
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))


# COMMAND ----------

# method to check if the word is a stop word
def is_stop_word(word):
    # print("found")
    if word in stop_words:
        return True


# COMMAND ----------

def normalize(matrix):
    for idx in range(len(matrix)):
        matrix[idx] /= matrix[idx].sum()

    return matrix


# COMMAND ----------

def cosine_similarity_matrix(sentences):
    sentences_count = len(article_sentences)
    print("Sentence count", sentences_count)
    # Create an empty similarity matrix
    cosine_similarity_matrix = numpy.zeros((sentences_count, sentences_count))

    for i in range(sentences_count):
        for j in range(sentences_count):
            if i == j:
                continue

            sentence_one = [word.lower() for word in article_sentences[i]]
            sentence_two = [word.lower() for word in article_sentences[j]]

            all_words = list(set(sentence_one + sentence_two))
            count_of_words = len(all_words)

            vector_for_sent1 = [0] * count_of_words
            vector_for_sent2 = [0] * count_of_words

            # build the vector for the first sentence
            for word in sentence_one:
                if (is_stop_word(word)): continue
                vector_for_sent1[all_words.index(word)] = vector_for_sent1[all_words.index(word)] + 1

            # build the vector for the second sentence
            for word in sentence_two:
                if (is_stop_word(word)): continue
                vector_for_sent2[all_words.index(word)] = vector_for_sent1[all_words.index(word)] + 1

            cosine_similarity_matrix[i][j] = 1 - cosine_distance(vector_for_sent1, vector_for_sent2)

            # print the similarity matrix
    # print(cosine_similarity_matrix)

    # return the cosine similarity matrix
    return normalize(cosine_similarity_matrix)


# COMMAND ----------

def page_rank(cosine_similarity_matrix):
    jump_factor = 0.85
    error = 0.0001
    initial_page_rank = numpy.ones(len(cosine_similarity_matrix)) / len(cosine_similarity_matrix)

    while True:
        length_cosine_similarity = len(cosine_similarity_matrix)
        dot_product = cosine_similarity_matrix.T.dot(initial_page_rank)
        page_rank = numpy.ones(length_cosine_similarity) * (1 - jump_factor) / len(
            cosine_similarity_matrix) + jump_factor * dot_product

        if (abs(page_rank - initial_page_rank).sum()) <= error:
            return page_rank
        initial_page_rank = page_rank


# COMMAND ----------

def get_summary(sentences, top_n=5):
    sentence_ranks = page_rank(cosine_similarity_matrix(sentences))

    # printing page rank
    # print(sentence_ranks)

    # Sort the sentence ranks
    ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
    top_n_sentences = ranked_sentence_indexes[:top_n]
    summary = operator.itemgetter(*sorted(top_n_sentences))(sentences)
    return summary


# COMMAND ----------

# DECISION TREE

# COMMAND ----------

import numpy as np
from math import log
import random


# COMMAND ----------

# Partition the output label based on the value
def partition(x):
    partition_split = {}
    counter = 0

    for i in x:
        if i in partition_split:
            partition_split[i].append(counter)
        else:
            partition_split[i] = [counter]
        counter += 1

    return partition_split


# COMMAND ----------

def predict(x, tree):
    for split_criterion in tree:
        sub_tree = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if split_decision == (x[attribute_index] == attribute_value):
            if type(sub_tree) is dict:
                prediction = predict(x, sub_tree)
            else:
                prediction = sub_tree

    return prediction


# COMMAND ----------

# Calculates the entropy value of the output label
def entropy(y, w=None):
    if w is None:
        w = np.ones((len(y), 1), dtype=int)

    total_weight = np.sum(w)
    partition_dict = partition(y)
    curr_entropy = 0.

    for key in partition_dict.keys():
        split_w = []
        for indices in partition_dict[key]:
            split_w.append(w[indices])
        y_val = np.sum(split_w) / total_weight
        curr_entropy -= y_val * log(y_val, 2)

    return curr_entropy


# COMMAND ----------

# Calculates the mutual information factor for the given input and the respective output label
def mutual_information(x, y, w=None):
    if w is None:
        w = np.ones((len(y), 1), dtype=int)

    y_entropy = entropy(y, w)

    conditional_entropy = 0
    total_weight = np.sum(w)
    partition_split = partition(x)

    for j in partition_split.keys():
        split_y = []
        split_w = []
        for x in partition_split[j]:
            split_y.append(y[x])
            split_w.append(w[x])
        conditional_entropy += np.sum(split_w) * entropy(split_y, split_w)
    curr_info_gain = y_entropy - (conditional_entropy / total_weight)

    return curr_info_gain


# COMMAND ----------

# Decision tree implementation
def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, w=None):
    local_attribute_val_pairs = []
    if depth == 0:
        column_counter = 0
        for column in x.T:
            unique_values = np.unique(column)
            for i in unique_values:
                local_attribute_val_pairs.append(tuple((column_counter, i)))
            column_counter += 1
    else:
        local_attribute_val_pairs = attribute_value_pairs

    values, counts = np.unique(y, return_counts=True)

    if (depth == max_depth) or (len(values) == 1) or (len(local_attribute_val_pairs) == 0):
        return values[counts.argmax()]

    best_attribute = 0
    best_value = 0
    prev_best_info_gain = 0

    for attribute, value in local_attribute_val_pairs:
        x_part = (np.array(x)[:, attribute] == value).astype(int)
        curr_best_info_gain = mutual_information(x_part, y)
        if curr_best_info_gain > prev_best_info_gain:
            prev_best_info_gain = curr_best_info_gain
            best_attribute = attribute
            best_value = value

    partition_split = partition((np.array(x)[:, best_attribute] == best_value).astype(int))
    build_tree = {}

    for value, indices in partition_split.items():
        decision = bool(value)

        build_tree[(best_attribute, best_value, decision)] = id3(x.take(indices, axis=0)
                                                                 , y.take(indices, axis=0)
                                                                 , attribute_value_pairs=local_attribute_val_pairs
                                                                 , depth=depth + 1
                                                                 , max_depth=max_depth
                                                                 , w=w)
    return build_tree


# COMMAND ----------

# Calculates the error percentage for the predicted output label
def compute_error(y_true, y_pred, w=None):
    if w is None:
        w = np.ones((len(y_true), 1), dtype=int)
    error = []
    for i in range(len(y_true)):
        error.append(w[i] * (y_true[i] != y_pred[i]))

    return np.sum(error) / np.sum(w)


# COMMAND ----------

# Decision tree - Bagging implementation
def bagging(x, y, max_depth, num_trees):
    indices = []
    h_i = {}
    alpha_i = 1
    for t in range(num_trees):
        for k in range(len(y)):
            indices.append(random.randrange(len(y)))
        decision_tree = id3(x[indices], y[indices], max_depth=max_depth)
        h_i[t] = (alpha_i, decision_tree)

    return h_i


# COMMAND ----------

# Predicts the output label for the given X feature vector and the ensemble
def predict_example(x, h_ens):
    y_pred = []
    for row in x:
        pred = []
        n = 0
        for bag in h_ens.keys():
            alpha, tree = h_ens[bag]
            pred.append(alpha * predict(row, tree))
            n += alpha

        avg_value = np.sum(pred) / n
        if avg_value >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)

    return y_pred


# COMMAND ----------

# Applies the decision tree bagging implementation on the cosine similarity matrix of the input text
M = np.genfromtxt('/dbfs/tmp/cosine_similarity_input_final_excel.txt', missing_values=0, skip_header=0,
                  delimiter='	', dtype=float)
ytrn = M[:, 0]
Xtrn = M[:, 1:]

# Load the test data
M = np.genfromtxt('/dbfs/tmp/cosine_similarity_input_final_excel.txt', missing_values=0, skip_header=0,
                  delimiter='	', dtype=float)
ytst = M[:, 0]
Xtst = M[:, 1:]

print('Bagging parameters and test data error')

for num_trees in (5, 10):
    for max_depth in (15, 17):
        h_ens = bagging(Xtrn, ytrn, max_depth, num_trees)
        y_pred = predict_example(Xtst, h_ens)
        test_error = compute_error(ytst, y_pred)
        print('Number of trees: ', num_trees)
        print('Maximum depth of the tree: ', max_depth)
        print('Test Error = {0:4.2f}%'.format(test_error * 100))
        print('----------')

# COMMAND ----------

# display the result
for idx, summary in enumerate(get_summary(article_sentences)):
    print(summary)

# COMMAND ----------


