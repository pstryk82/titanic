import pandas as pd
import toolkit
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import matplotlib.pyplot as plt
import statsmodels.tools as smtools
from ClassifierLibrary.DecisionTree import DecisionTreeClassifier
from ClassifierLibrary.LogisticRegression import LogisticRegressionClassifier
from ClassifierLibrary.RandomForest import RandomForestClassifier
from app.DataPreprocessor import DataPreprocessor

dataframe = pd.read_csv('./dataset/train.csv')


# take all data to fill in missing values based on all population
test_dataframe = pd.read_csv('./dataset/test.csv')
all_passengers = dataframe.append(test_dataframe, ignore_index=True)


preprocessor = DataPreprocessor(all_passengers)
# try to fill in missing family data
preprocessor.assume_family_member()

# create additional variable for passenger's family size
preprocessor.set_family_size()

# try to best guess missing age data based on family information
preprocessor.assume_age()


# create additional variable indicating if the passenger was travelling alone
#preprocessor.set_passenger_was_alone(dataframe)




# ignore following variables:
# Fare: doesn't seem to relate passengers
# Cabin: too many missing values
# Name: we have PassengerId anyway
dataframe.drop(labels=['Name', 'Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)

# ignore variables that just served for deriving other variables
dataframe.drop(labels=['SibSp', 'Parch', 'Ticket'], axis=1, inplace=True)


# remove passengers with empty age (age itself seems to be relevant, shouldn't ignore it)
dataframe.dropna(subset=['Age'], inplace=True)
# This is equivalent to:
# dataframe = dataframe[dataframe.Age > 0]


# assume missing Embarked means 'Southampton'
dataframe.fillna(value={'Embarked':'S'}, inplace=True)

# label encoding and one hot encoding categorical variables: Pclass, Sex, Embarked
dataframe = pd.get_dummies(dataframe, columns=['Pclass', 'Sex'], drop_first=True)

# for now feature scaling seems unnecessary, but we'll add it later if it turns out to be required

# extract independent and dependent variable matrices
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

classifier = LogisticRegressionClassifier()
accuracies['Logistic Regression'], std['Logistic Regression'] = classifier.classify(X, y)

print('accuracies: ', accuracies)
print('std: ', std)

