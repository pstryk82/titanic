import numpy
# import pandas
from pandas import DataFrame

class DataPreprocessor:
    def __init__(self, dataframe: DataFrame):
        self.dataframe = dataframe

    def assume_age(self):

    # if passenger was alone, we can't tell anything
    # however, we can tell who his family members were based on ticket number and common surname
    # then we can check if passenger was somebody's spouse and assume his age based on spouse's
    # or if he was somebody's sibling, we can assume his age based on them


        """
            def set_passenger_was_alone(self, self.dataframe: DataFrame):
                self.dataframe.insert(len(self.dataframe.columns), 'Alone', numpy.nan)
                self.dataframe.loc[self.dataframe['SibSp'] > 0, 'Alone'] = 0
                self.dataframe.loc[self.dataframe['Parch'] > 0, 'Alone'] = 0

                tickets = self.dataframe['Ticket']

                self.dataframe.loc[self.dataframe.Ticket.isin(tickets[tickets.duplicated(keep=False)]), 'Alone'] = 0

                self.dataframe.fillna(value={'Alone': 1}, inplace=True)
        """

    def set_family_size(self):
        self.dataframe.insert(len(self.dataframe.columns), 'FamilySize', 0)
        self.dataframe.loc[:, 'FamilySize'] = self.dataframe['SibSp'] + self.dataframe['Parch'] + 1


    def assume_family_member(self):
        # if ticket number and surname is the same, we assume these are family members
        duplicated_tickets = self.dataframe['Ticket'].duplicated(keep=False)
        self.dataframe.insert(len(self.dataframe.columns), 'Surname', self.dataframe['Name'].str.split(',', expand=True)[0])
        self.dataframe.groupby(['Ticket', 'Surname']).size()  # what to do next with this?
        pass
        

    def resolve_surname(self):
        self.dataframe.insert(
            len(self.dataframe.columns),
            'Surname',
            self.dataframe['Name'].str.split(',', expand=True)[0]
        )
