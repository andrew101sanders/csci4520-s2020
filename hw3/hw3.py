from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

X_categories = ['age', 'type_employer', 'fnlwgt', 'education', 'education_num', 'marital', 'occupation',
                'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hr_per_week', 'country']
X_categorical_data = ['type_employer', 'education', 'marital', 'occupation', 'relationship', 'race', 'sex',
                      'country']
X_continuous_data = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hr_per_week']
Y_categories = 'income'
ss = StandardScaler()


def getData():
    datafile = pd.read_csv('adult.data')
    x_data = pd.get_dummies(datafile.iloc[:, :-1])
    x_data[X_continuous_data] = ss.fit_transform(x_data[X_continuous_data])

    X_categories_plus_dummy = list(x_data.columns)
    datafile_y = datafile[Y_categories].to_numpy().reshape(-1, 1)
    y_data = LabelBinarizer().fit_transform(datafile_y).ravel()
    return train_test_split(x_data, y_data, test_size=0.2), X_categories_plus_dummy


def trainModels(X_train, y_train):
    models = []
    modelnames = []

    default = SGDClassifier(loss='log')
    default.fit(X_train, y_train)
    models.append(default)
    modelnames.append('default')

    alpha1 = SGDClassifier(loss='log', alpha=1)
    alpha1.fit(X_train, y_train)
    models.append(alpha1)
    modelnames.append('alpha1')

    alpha01 = SGDClassifier(loss='log', alpha=.1)
    alpha01.fit(X_train, y_train)
    models.append(alpha01)
    modelnames.append('alpha01')

    alpha001 = SGDClassifier(loss='log', alpha=.01)
    alpha001.fit(X_train, y_train)
    models.append(alpha001)
    modelnames.append('alpha001')

    alpha0001 = SGDClassifier(loss='log', alpha=.001)
    alpha0001.fit(X_train, y_train)
    models.append(alpha0001)
    modelnames.append('alpha0001')

    alpha000001 = SGDClassifier(loss='log', alpha=.00001)
    alpha000001.fit(X_train, y_train)
    models.append(alpha000001)
    modelnames.append('alpha000001')

    penaltyl1 = SGDClassifier(loss='log', penalty='l1')
    penaltyl1.fit(X_train, y_train)
    models.append(penaltyl1)
    modelnames.append('penaltyl1')

    penaltyelasticnet = SGDClassifier(loss='log', penalty='elasticnet')
    penaltyelasticnet.fit(X_train, y_train)
    models.append(penaltyelasticnet)
    modelnames.append('penaltyelasticnet')

    learningrate_constant_eta0_1 = SGDClassifier(loss='log', learning_rate='constant', eta0=1)
    learningrate_constant_eta0_1.fit(X_train, y_train)
    models.append(learningrate_constant_eta0_1)
    modelnames.append('learningrate_constant_eta0_1')

    learningrate_constant_eta0_01 = SGDClassifier(loss='log', learning_rate='constant', eta0=.1)
    learningrate_constant_eta0_01.fit(X_train, y_train)
    models.append(learningrate_constant_eta0_01)
    modelnames.append('learningrate_constant_eta0_01')

    learningrate_constant_eta0_001 = SGDClassifier(loss='log', learning_rate='constant', eta0=.01)
    learningrate_constant_eta0_001.fit(X_train, y_train)
    models.append(learningrate_constant_eta0_001)
    modelnames.append('learningrate_constant_eta0_001')

    learningrate_constant_eta0_0001 = SGDClassifier(loss='log', learning_rate='constant', eta0=.001)
    learningrate_constant_eta0_0001.fit(X_train, y_train)
    models.append(learningrate_constant_eta0_0001)
    modelnames.append('learningrate_constant_eta0_0001')

    return models, modelnames


def getAccuracy(model, X_train, X_test, y_train, y_test):
    scrtr = 0
    scrte = 0
    for i in range(100):
        scrtr += model.score(X_train, y_train)
        scrte += model.score(X_test, y_test)
    # print('\ttraining accuracy: ' + str(scrtr/100))
    print('\ttesting accuracy: ' + str(scrte / 100))
    return scrte / 100


def predictsingle(model, columns, input):
    # use this to predict custom inputs

    # input = [92, 'Self-emp-not-inc', 183710, '9th', 5, 'Married-civ-spouse', 'Farming-fishing', 'Husband', 'White', 'Male',
    #       0, 0, 40, 'United-States']
    test = pd.DataFrame(input, columns=X_categories)

    test = pd.get_dummies(test)

    # There is probably a better way of doing this
    df = pd.DataFrame(np.zeros([1, len(columns)]), columns=columns)
    df += test
    df = df.fillna(0)
    df[X_continuous_data] = ss.transform(df[X_continuous_data])

    # outputstring = 'For the values: \n'
    # for i in test:
    #     outputstring += '{}: {}\n'.format(i, test.at[0, i])
    # outputstring += 'The model predict this person is making {} than 50k'.format(
    #     'more' if model.predict(df) == 0 else 'less')
    # print(outputstring)
    return model.predict_proba(df)[0][0]


def confusionMatrix(model, x_test, y_test):
    disp = plot_confusion_matrix(model, x_test, y_test, cmap=plt.cm.Blues, values_format='')
    plt.xticks([0, 1], ['<=50k', '>50k'])
    plt.yticks([0, 1], ['<=50k', '>50k'])
    plt.show()


def scatterplot(model, columns):
    # Used for testing custom input
    sns.set(style='white')
    f, ax = plt.subplots(figsize=(8, 6))
    grid = []
    dataposx = []
    dataposy = []
    for i in range(0, 31):
        grid.append([])
        for j in range(0, 31):
            dataposx.append(i)
            dataposy.append(j)
            grid[i].append(
                [25, 'Private', 30771, 'Some-college', 10, 'Married-civ-spouse', 'Adm-clerical', 'Wife',
                 'Black', 'Female',
                 i*100, j*100, 25, 'United-States'])
    probabilities = []
    for i in range(0, 31):
        probabilities.append([])
        for j in range(0, 31):
            probabilities[i].append(predictsingle(model, columns, [grid[i][j]]))
    con = ax.contour(range(0, 31), range(0, 31), np.reshape(probabilities,[31,31]), cmap="Greys", vmin=0, vmax=.6, levels=[.5])

    ax.scatter(dataposx, dataposy, c=sum(probabilities,[]), s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(0, 30), ylim=(0, 30),
           xlabel="Capital gain", ylabel="Capital loss")
    plt.show()


def scatterplot2(model, columns, x_train):
    # used for testing the training data
    sns.set(style='white')
    f, ax = plt.subplots(figsize=(8, 6))
    probs = model.predict_proba(x_train)
    probs = [i[0] for i in probs]
    x = np.linspace(-5, 5, 100)
    a = -model.coef_[0][x_train.columns.get_loc('age')] / model.coef_[0][x_train.columns.get_loc('hr_per_week')]
    y = a * x - (model.intercept_[0]) / model.coef_[0][x_train.columns.get_loc('hr_per_week')]

    plt.plot(x, y, 'k-')
    ax.scatter(x_train['age'], x_train['hr_per_week'], c=probs, s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlabel="Age", ylabel="Hours per week")
    plt.show()


split, columns = getData()
array_of_models, array_of_modelnames = trainModels(split[0], split[2])
modelacc = []
for i in range(len(array_of_models)):
    print('Testing model: ' + str(array_of_modelnames[i]))
    modelacc.append(getAccuracy(array_of_models[i], split[0], split[1], split[2], split[3]))

most_accurate_model = modelacc.index(max(modelacc))
print('the highest accuracy model is: ' + str(array_of_modelnames[most_accurate_model]))
# predictsingle(array_of_models[0], columns)
# confusionMatrix(array_of_models[most_accurate_model], split[1], split[3])
scatterplot2(array_of_models[most_accurate_model], columns, split[0])
