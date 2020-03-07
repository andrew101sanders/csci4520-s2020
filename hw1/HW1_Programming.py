from sklearn import tree
from sklearn import preprocessing
import graphviz
import pandas as pd

# Grabbing the data from the csv's
# Make sure to replace paths with the correct paths for your environment
train_x = pd.read_csv('c:/users/andrew/desktop/train.csv',
                      usecols=['Type', 'Price'],  # We only want these columns
                      skipinitialspace=True)
train_y = pd.read_csv('c:/users/andrew/desktop/train.csv',
                      usecols=['Category'],  # We only want these columns
                      skipinitialspace=True)
test_x = pd.read_csv('c:/users/andrew/desktop/test.csv',
                     usecols=['Type', 'Price'],  # We only want these columns
                     skipinitialspace=True)
test_y = pd.read_csv('c:/users/andrew/desktop/test.csv',
                     usecols=['Category'],  # We only want these columns
                     skipinitialspace=True)

# These arrays let the one hot encoder know basically how many indecies to use
Type = ['HipHop', 'Rock', 'Jazz']
Price = ['Cheap', 'Expensive']
Buy = ['No', 'Yes']

enc_x = preprocessing.OneHotEncoder(categories=[Type, Price], handle_unknown='ignore')
enc_y = preprocessing.OneHotEncoder(categories=[Buy], handle_unknown='ignore')
enc_test_x = preprocessing.OneHotEncoder(categories=[Type, Price], handle_unknown='ignore')
enc_test_y = preprocessing.OneHotEncoder(categories=[Buy], handle_unknown='ignore')

# Fitting the encoders the the specific arrays
enc_x.fit(train_x)
enc_y.fit(train_y)
enc_test_x.fit(test_x)
enc_test_y.fit(test_y)

# Encoding the arrays
train_x_array = enc_x.transform(train_x).toarray()
train_y_array = enc_y.transform(train_y).toarray()
test_x_array = enc_x.transform(test_x).toarray()
test_y_array = enc_y.transform(test_y).toarray()

# Training the Decision Tree
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(train_x_array, train_y_array)

# Testing its accuracy, and rounding it to 2 decimal places
print('Training Set Accuracy: ' + str(round(clf.score(train_x_array, train_y_array), 2)))
print('Testing Set Accuracy: ' + str(round(clf.score(test_x_array, test_y_array), 2)))
print(clf.predict([[0,1,0,1,0]]))

# Exporting the tree data
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=['HipHop', 'Rock', 'Jazz', 'Cheap', 'Expensive'])

# Reformatting the tree data
dot_data = dot_data.replace('True', 'asdf')
dot_data = dot_data.replace('False', 'True')
dot_data = dot_data.replace('asdf', 'False')
dot_data = dot_data.replace('<=', '>')

# Creating the pdf
graph = graphviz.Source(dot_data)
graph.render('HW1')
