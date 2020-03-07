from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

datafile_x = pd.get_dummies(pd.read_csv('adult.data',
                                        usecols=['age', 'type_employer', 'fnlwgt', 'education', 'education_num',
                                                 'marital',
                                                 'occupation', 'relationship', 'race', 'sex', 'capital_gain',
                                                 'capital_loss',
                                                 'hr_per_week', 'country'],
                                        skipinitialspace=True))
datafile_y = pd.read_csv('adult.data',
                         usecols=['income'],
                         skipinitialspace=True)
X_train, X_test, y_train, y_test = train_test_split(datafile_x, datafile_y.values.ravel(), test_size=0.2)
Log_Reg = SGDClassifier(max_iter=2000, loss='log')
Log_Reg.fit(X_train, y_train)
print(Log_Reg.score(X_train, y_train))
print(Log_Reg.score(X_test, y_test))
