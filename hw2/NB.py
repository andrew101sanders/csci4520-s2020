import math

probability_matrix = []
totalCategory = []
uniqueCategory = []
totalWordsCategory = []
alpha = 1


def createEmptyMatrix(vocab_array_length, newslabels_array_length):
    matrix = []
    for x in range(vocab_array_length):
        matrix.append([])
        for y in range(newslabels_array_length):
            matrix[x].append(0)
    return matrix


vocab = open('vocabulary.txt').read().splitlines()

newslabels = open('newsgrouplabels.map').read().splitlines()
newslabels = [x.split(' ')[0] for x in newslabels]  # Strip the number

train_data = open('train.data').read().splitlines()
train_data = [x.split(' ') for x in train_data]

test_data = open('test.data').read().splitlines()
test_data = [x.split(' ') for x in test_data]

train_label = open('train.label').read().splitlines()

test_label = open('test.label').read().splitlines()

train_matrix = createEmptyMatrix(len(vocab), len(newslabels))

uniqueWords = len(vocab)

for x in range(len(newslabels)):
    totalCategory.append(0)
    uniqueCategory.append(x + 1)
    totalWordsCategory.append(0)

for x in uniqueCategory:
    for y in train_label:
        if int(x) == int(y):
            totalCategory[x - 1] += 1

cumulativeTotalCategory = sum(totalCategory)

for x in train_data:
    label = int(train_label[(int(x[0]) - 1)]) - 1
    word = int(int(x[1]) - 1)
    train_matrix[word][label] += int(x[2])

for x in train_matrix:
    count = 0
    for y in x:
        totalWordsCategory[count] += y
        count += 1

print(totalWordsCategory)
print(totalCategory)
print(cumulativeTotalCategory)
print(uniqueCategory)
print(uniqueWords)

countx = 0
for x in vocab:
    probability_matrix.append([])
    county = 0
    for y in newslabels:
        probability_matrix[countx].append(
            (int(train_matrix[countx][county]) + alpha) / (int(totalWordsCategory[county]) + uniqueWords))
        county += 1
    countx += 1


def probabilityofCategory(category):
    return totalCategory[category] / cumulativeTotalCategory


def probabiltyofWordGivenCategory(word, category):
    return probability_matrix[word - 1][category]


def probabilityofCategoryGivenWords(category, words):
    result = 1
    for word in words:
        result += math.log10(probabiltyofWordGivenCategory(word, category))
    return math.log10(probabilityofCategory(category)) + result


currentindex = 0


def argmax(document):
    wordlist = []
    probabilities = []
    leng = len(test_data)
    global currentindex
    while currentindex < leng:
        if int(test_data[currentindex][0]) == int(document):
            for x in range(int(test_data[currentindex][2])):
                wordlist.append(int(test_data[currentindex][1]))
        else:
            break
        currentindex += 1
    currentindex += 1

    for category in range(len(uniqueCategory)):
        probabilities.append(probabilityofCategoryGivenWords(category, wordlist))

    # return probabilities
    return probabilities.index(max(probabilities))


# print(probabilityofCategory(0))
document_predictions = []
count = 0
for document in test_label:
    count += 1
    document_predictions.append(argmax(count))

print(document_predictions)
print(document_predictions.__len__())

count = 0
correct = 0
for x in test_label:
    if document_predictions[count] + 1 == int(x):
        correct += 1
    count += 1

print('correct: ' + str(correct / len(test_label)))
