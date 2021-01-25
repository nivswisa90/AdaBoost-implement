# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from AdaBoost import *
import pandas as df
import numpy as np


def main():
    dataSet = pd.read_csv('titanikData.csv')
    dataSetTest = pd.read_csv('titanikTest.csv', names=['pclass', 'age', 'gender', 'survived'])
    dataSet = CategoricalToNumerical(dataSet)
    dataSetTest = CategoricalToNumerical(dataSetTest)
    finalDataSetTest = dataSetTest.drop('survived', axis=1)
    # uniqueDataFrame = dataSet.drop_duplicates()

    predictions = list()
    for i in range(3):
        # Create sub sample and convert it to DataFrame.
        # sample = createSubSamples(uniqueDataFrame)
        prediction = AdaBoost(dataSet, finalDataSetTest)
        predictions.append(prediction)
    # print(predictions)


if __name__ == '__main__':
    main()
