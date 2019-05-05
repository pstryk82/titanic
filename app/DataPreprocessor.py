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

    # from the plots:
    #   if someone had 4+ SibSp, he/she was <20 years old
    #   if someone had 3+ Parch, he/she was 16-65 years old
        pass


    def set_family_size(self):
        self.dataframe.insert(len(self.dataframe.columns), 'FamilySize', 0)
        self.dataframe.loc[:, 'FamilySize'] = self.dataframe['SibSp'] + self.dataframe['Parch'] + 1


    def assume_family_member(self):
        # # if ticket number and surname is the same, we assume these are family members
        # duplicated_tickets = self.dataframe['Ticket'].duplicated(keep=False)
        # self.dataframe.insert(len(self.dataframe.columns), 'Surname', self.dataframe['Name'].str.split(',', expand=True)[0])
        # self.dataframe.groupby(['Ticket', 'Surname']).size()  # what to do next with this?
        pass
        

    def resolve_surname(self):
        self.dataframe.insert(
            len(self.dataframe.columns),
            'Surname',
            self.dataframe['Name'].str.split(',', expand=True)[0]
        )
