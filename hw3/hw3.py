from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

datafile_x, datafile_y, X_train, X_test, y_train, y_test, Log_Reg = None, None, None, None, None, None, None


def getData():
    global datafile_x, datafile_y, X_train, X_test, y_train, y_test
    datafile = pd.read_csv('adult.data', skipinitialspace=True)
    datafile_x = pd.get_dummies(datafile.iloc[:, :-1])
    datafile_y = datafile.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(datafile_x, datafile_y.values.ravel(), test_size=0.2)


def trainModel():
    global Log_Reg
    Log_Reg = SGDClassifier(max_iter=2000, loss='log')
    Log_Reg.fit(X_train, y_train)


def getAccuracy():
    print(Log_Reg.score(X_train, y_train))
    print(Log_Reg.score(X_test, y_test))


getData()
trainModel()
getAccuracy()
