"""
Title: 20 Newsgroups Naive Bayes Implementation
Author: Andrew Sanders
"""
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import math

print('(0/6) loading files into memory...')
train_data = np.loadtxt('train.data', dtype=int)
print('(1/6) train.data loaded')
test_data = np.loadtxt('test.data', dtype=int)
print('(2/6) test.data loaded')
train_label = np.loadtxt('train.label', dtype=int)
print('(3/6) train.label loaded')
test_label = np.loadtxt('test.label', dtype=int)
print('(4/6) test.label loaded')
vocab = open('vocabulary.txt', ).read().splitlines()
print('(5/6) vocabulary.txt loaded')
newsgrouplabels = np.loadtxt('newsgrouplabels.map', dtype=str, usecols=[0])
print('(6/6) newsgrouplabels.map loaded')
stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
alpha = .1
vocab_count = len(vocab)

# calculate probabilty of labels
print('calculating probability of labels...')
labels_count = dict(zip(*np.unique(train_label, return_counts=True)))
labels_sum = sum(labels_count.values())

label_probability = {}  # 1-20
for i in labels_count:
    label_probability[i] = labels_count[i] / labels_sum

# number of times a word is in a label
print('calculating number of times a word is in a label...')
vocab_occurrence_matrix = np.full((vocab_count, 20), 0)  # [0-61187][0-19]
for word in train_data:
    vocab_occurrence_matrix[word[1] - 1][train_label[word[0] - 1] - 1] += word[2]

# calculate words per category
print('calculating words per category...')
words_per_category = {}  # 1-20
for i in range(1, 21):
    words_per_category[i] = 0

for word in train_data:
    words_per_category[train_label[word[0] - 1]] += word[2]

# calculate probabilty of word given label
print('calculating probabilty of word given label...')
probability_of_word_given_label = np.full((vocab_count, 20), 1 / vocab_count)  # [0-61187][0-19]
for word in range(vocab_count):
    if vocab[word] in stop_words:
        for i in range(20):
            probability_of_word_given_label[word][i] = alpha / vocab_count
    else:
        for label in range(20):
            probability_of_word_given_label[word][label] = (vocab_occurrence_matrix[word][label] + alpha) / (
                    words_per_category[label + 1] + vocab_count)


def getLabelProbability(label):
    """
    :param label: label to check probability [1-20]
    :return: label probability
    """
    return label_probability[label]


def getWordGivenLabelProbability(word, label):
    """
    :param word: word to check [0-61187]
    :param label: label to check probability of word [0-19]
    :return: Word Given Label probability
    """
    return probability_of_word_given_label[word][label]


def getLabelGivenWordsProbability(label, words):
    """
    :param label: whats the chance the words came from this label? [1-20]
    :param words: Words in question [list of [1-61188]]
    :return: Label Given Words probability
    """
    result = math.log(getLabelProbability(label))
    for word in words:
        result += math.log(getWordGivenLabelProbability(word - 1, label - 1))
    return result


td = pd.DataFrame(test_data, columns=['docID', 'WordID', 'Word Count'])
grouped_TD = td.groupby('docID').count()
numStart = 0


def argmax(documentid):
    """
    :param document: test_data DocID to test [1-7505]
    :return: label [1-21]
    """
    docwords = []
    global numStart
    numEnd = numStart + grouped_TD['Word Count'][documentid] - 1
    for index in range(numStart, numEnd + 1):
        for i in range(test_data[index][2]):
            docwords.append(test_data[index][1])
    numStart = numEnd + 1
    results = []
    for i in range(1, 21):
        results.append(getLabelGivenWordsProbability(i, docwords))
    return results.index(max(results)) + 1


document_predictions = []
for document in range(1, len(test_label) + 1):
    #print('count: ' + str(document))
    document_predictions.append(argmax(document))

print(document_predictions)

confusion_matrix = np.full((20, 20), 0)
count = 0
correct = 0
for x in test_label:
    confusion_matrix[document_predictions[count] - 1][int(x) - 1] += 1
    if document_predictions[count] == int(x):
        correct += 1
    count += 1

print('test accuracy: ' + str(correct / len(test_label)))

df_cm = pd.DataFrame(confusion_matrix, range(1, 21), range(1, 21))
plt.figure(figsize=(10, 8))
sn.set(font_scale=.8)
ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 8}, fmt='g')
ax.invert_yaxis()
ax.set(xlabel='Ground Truths', ylabel='Predictions')
plt.show()
