import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

#print data.head()
#print data.shape
#print data.describe()

'''
    NOTE:
    This data set has the following set of features 
    quality - This is the target
    fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
    free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
'''

# Splitting the data into training and test set
y = data.quality
X = data.drop('quality', axis=1)

#print y.head()
#print X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# Standardization

#scaler = preprocessing.StandardScaler().fit(X_train)

#X_train_scaled = scaler.transform(X_train)

#print X_train_scaled.mean(axis = 0)
#print X_train_scaled.std(axis = 0)

#X_test_scaled = scaler.transform(X_test)

# Make a pipeline with the preprocessing shown above and model
pipeline = make_pipeline(preprocessing.StandardScaler(),
                         RandomForestRegressor(n_estimators=100))

#print pipeline.get_params()
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                    'randomforestregressor__max_depth': [None, 5, 3, 1]}


clf = GridSearchCV(pipeline, hyperparameters, cv=10)
 
# Fit and tune model
clf.fit(X_train, y_train)

# Evaluate on the test data
pred = clf.predict(X_test)

#check the results
print r2_score(y_test, pred)
print mean_squared_error(y_test, pred)
