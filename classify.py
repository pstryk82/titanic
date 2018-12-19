import pandas as pd
import numpy as np
import toolkit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

dataframe = pd.read_csv('./dataset/train.csv')

# ignore following variables:
# Fare: doesn't seem to relate passengers
# Cabin: too many missing values
# Name: we have PassengerId anyway
dataframe.drop(labels=['Name', 'Fare', 'Cabin'], axis=1, inplace=True)


# remove passengers with empty age (age itself seems to be relevant, shouldn't ignore it)
dataframe.dropna(subset=['Age'], inplace=True)
# This is equivalent to:
# dataframe = dataframe[dataframe.Age > 0]


# assume missing Embarked means 'Southampton'
dataframe.fillna(value={'Embarked':'S'}, inplace=True)

# label encoding and one hot encoding categorical variables: Pclass, Sex, Ticket, Embarked
dataframe = pd.get_dummies(dataframe, columns=['Pclass', 'Sex', 'Ticket', 'Embarked'], drop_first=True)

# for now feature scaling seems unnecessary, but we'll add it later if it turns out to be required
X = dataframe.drop(labels=['PassengerId', 'Survived'], axis=1).values
y = dataframe._getitem_column('Survived').values


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

print(cm)




from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X, y = y, cv = 10)
print(accuracies.mean())
print(accuracies.std())

