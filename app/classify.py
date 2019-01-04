import pandas as pd
import toolkit
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import matplotlib.pyplot as plt
import statsmodels.tools as smtools
from ClassifierLibrary.DecisionTree import DecisionTreeClassifier
from ClassifierLibrary.RandomForest import RandomForestClassifier

dataframe = pd.read_csv('./dataset/train.csv')

# create additional variable indicating if the passenger was travelling alone
toolkit.set_passenger_was_alone(dataframe)


# ignore following variables:
# Fare: doesn't seem to relate passengers
# Cabin: too many missing values
# Name: we have PassengerId anyway
# Ticket: we get a better accuracy without it
dataframe.drop(labels=['Name', 'Fare', 'Cabin', 'Ticket'], axis=1, inplace=True)


# remove passengers with empty age (age itself seems to be relevant, shouldn't ignore it)
dataframe.dropna(subset=['Age'], inplace=True)
# This is equivalent to:
# dataframe = dataframe[dataframe.Age > 0]


# assume missing Embarked means 'Southampton'
dataframe.fillna(value={'Embarked':'S'}, inplace=True)

# label encoding and one hot encoding categorical variables: Pclass, Sex, Embarked
dataframe = pd.get_dummies(dataframe, columns=['Pclass', 'Sex', 'Embarked'], drop_first=True)

# for now feature scaling seems unnecessary, but we'll add it later if it turns out to be required

# extract dependent and independent variable matrices
X = dataframe.drop(labels=['PassengerId', 'Survived'], axis=1)
y = dataframe._getitem_column('Survived')



## Backward Elimination

# add a column of 1s to represent x0 variable (intercept)
X = smtools.add_constant(X)


# use Backward Elimination to get rid of insignificant variables
significance_level = 0.05
X = toolkit.backward_elimination_using_pvalues(X, y, significance_level)
# X = toolkit.backward_elimination_using_adjR2(X, y)



# Fitting Decision Tree Classification to the Training set
accuracies = {}
std = {}
classifier = DecisionTreeClassifier()
accuracies['Decision Tree'], std['Decision Tree'] = classifier.classify(X, y)



# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier()
accuracies['Random Forest'], std['Random Forest'] = classifier.classify(X, y)

print('accuracies: ', accuracies)
print('std: ', std)

