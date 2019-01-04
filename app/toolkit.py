import statsmodels.formula.api as sm
import numpy

def set_passenger_was_alone(dataframe):
    dataframe.insert(len(dataframe.columns), 'Alone', numpy.nan)
    dataframe.loc[dataframe['SibSp'] > 0, 'Alone'] = 0
    dataframe.loc[dataframe['Parch'] > 0, 'Alone'] = 0

    tickets = dataframe['Ticket']

    dataframe.loc[dataframe.Ticket.isin(tickets[tickets.duplicated(keep=False)]), 'Alone'] = 0

    dataframe.fillna(value={'Alone': 1}, inplace=True)


def backward_elimination_using_pvalues(input_matrix, output_matrix, significance_level):
    ordinary_least_squares_regressor = sm.OLS(endog=output_matrix, exog=input_matrix).fit()

    while max(ordinary_least_squares_regressor.pvalues) > significance_level:
        max_pvalue_index = ordinary_least_squares_regressor.pvalues.idxmax()
        print('Dropping column: ', max_pvalue_index)
        input_matrix.drop(labels=[max_pvalue_index], axis=1, inplace=True)
        ordinary_least_squares_regressor = sm.OLS(endog=output_matrix, exog=input_matrix).fit()
        # print(ordinary_least_squares_regressor.summary())

    print('Input matrix final shape: ', input_matrix.shape)
    # print(ordinary_least_squares_regressor.summary())

    return input_matrix


def backward_elimination_using_adjR2(input_matrix, output_matrix):
    ordinary_least_squares_regressor = sm.OLS(endog=output_matrix, exog=input_matrix).fit()
    previous_adjR2 = -1
    adjR2 = ordinary_least_squares_regressor.rsquared_adj

    while adjR2 >= previous_adjR2:
        max_pvalue_index = ordinary_least_squares_regressor.pvalues.idxmax()
        print('Dropping column: ', max_pvalue_index)
        input_matrix.drop(labels=[max_pvalue_index], axis=1, inplace=True)
        ordinary_least_squares_regressor = sm.OLS(endog=output_matrix, exog=input_matrix).fit()
        previous_adjR2 = adjR2
        adjR2 = ordinary_least_squares_regressor.rsquared_adj
        # @todo need to restore recently deleted column because adjR2 is higher when it's in place

    # print(ordinary_least_squares_regressor.summary())
    print('Input matrix final shape: ', input_matrix.shape)
    print('adjR2:', adjR2, 'previous_adjR2: ', previous_adjR2)

    return input_matrix