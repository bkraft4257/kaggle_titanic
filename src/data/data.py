import pandas as pd
from typing import Union
from pathlib import Path


class ExtractData:
    def __init__(self, filename: Union[str, Path]) -> None:
        """Extract Training Data from file or Path
        
        Arguments:
            filename {[str]} -- Filename of CSV data file containing data. 
        """

        self.filename = filename
        self.all_label_columns = ['survived']
        self.all_feature_columns = ['pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket',
                                    'fare', 'cabin', 'embarked']
        self.Xy_raw = None
        self.Xy = None

        self.extract()
        self.clean()

    def extract(self) -> pd.DataFrame:
        """
        Extracts data from a CSV file.  

        Returns:
            pd.DataFrame -- [description]
        """
        Xy_raw = pd.read_csv(self.filename)
        Xy_raw.columns = Xy_raw.columns.str.lower().str.replace(' ', '_')

        self.Xy_raw = Xy_raw.set_index('passengerid')

    def clean(self, drop_columns=None):
        """Clean data to remove missing data and "unnecessary" features.
        
        Arguments:
            in_raw_df {pd.DataFrame} -- Dataframe containing all columns and rows Kaggle Titanic Training Data set
        
        Keyword Arguments:
            drop_columns {[type]} -- [description] (default: {None})
        """

        if drop_columns is None:
            drop_columns = ['age', 'cabin', 'name', 'ticket']

        self.Xy = self.Xy_raw.drop(drop_columns, axis=1).dropna(axis=0, how='any')
