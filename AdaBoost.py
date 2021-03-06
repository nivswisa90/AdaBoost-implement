import pandas as pd
from sklearn import tree
import numpy as np
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
    df = data.copy()
    df["weight"] = (1 / len(data))
    hypothesis = list()
    alpha_list = list()
    features = ['pclass', 'age', 'gender']
    x = data[features]
    y = data.survived
    for i in range(3):
        clf = tree.DecisionTreeClassifier()
        clf.max_depth = 1
        clf.criterion = 'entropy'
        clf = clf.fit(x, y)
        hypothesis.append(clf)

        prediction = clf.predict(x)
        # new column name prediction
        df["prediction"] = prediction

        # check the where the prediction got wrong predict
        df['wrong predict'] = np.where(df["prediction"] != data["survived"], 1, 0)

        # calculate error ratio
        calculateError = np.sum(df['weight'] * df['wrong predict'])
        if calculateError > 0.5:
            pass
        calculateBeta = calculateError / (1 - calculateError)
        calculateAlpha = np.log(1 / calculateBeta) * 0.5
        # If prediction is True weight*beta else weight*alpha

        df['weight'] = np.where(df["wrong predict"] == 0, df['weight'] * calculateBeta, df['weight'] * calculateAlpha)

        # Normalize the weights
        weight_sum = sum(df['weight'])
        df['weight'] = df['weight'] / weight_sum

    final_prediction = []
    for (j, i) in enumerate(hypothesis):
        final_prediction.append(alpha_list[j] * i.predict(test))

    predictions = [sum(x) for x in zip(*np.array(final_prediction))]

    for i in range(len(predictions)):
        predictions[i] = 'yes' if predictions[i] == 0 else 'no'

    predict_decision = pd.DataFrame(predictions, columns=['prediction'])
    titanic_test_final_data = pd.concat([test, predict_decision], axis=1)

    print(titanic_test_final_data)
    print("Successfully predicted data with accuracy " + "= "
          + str(accuracy_score(titanic_test_final_data.survived, titanic_test_final_data.prediction) * 100) + ' %')

    return df
