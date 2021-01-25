import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score


# def createSubSamples(data):
#     # arange the dataframe only with unique values
#     uniqueDataFrame = data.sample(frac=.63)
#
#     nDataFrame = len(data)
#     nUniqueDataFrame = len(uniqueDataFrame)
#     duplicateSample = uniqueDataFrame.sample(n=(nDataFrame - nUniqueDataFrame), replace=True)
#
#     finalSample = pd.concat([uniqueDataFrame, duplicateSample])
#     return finalSample


def CategoricalToNumerical(dataframe):
    dataframe.pclass = dataframe.pclass.map({'crew': 0, '1st': 1, '2nd': 2, '3rd': 3})
    dataframe.gender = dataframe.gender.map({'male': 1, 'female': 0})
    dataframe.age = dataframe.age.map({'adult': 1, 'child': 0})
    dataframe.survived = dataframe.survived.map({'yes': 1, 'no': 0})
    return dataframe


def AdaBoost(data, test):
    # initial same weights to each record in the database
    data["weight"] = (1 / len(data))
    sample = data.sample(len(data), replace=True, weights=data['weight'])
    # print(sample)
    features = ['pclass', 'age', 'gender']
    x = sample[features]
    y = sample.survived
    clf = tree.DecisionTreeClassifier()
    clf.max_depth = 1
    clf.criterion = 'entropy'
    clf = clf.fit(x, y)
    prediction = clf.predict(x)
    sample["prediction"] = prediction
    print(sample)


    return prediction
