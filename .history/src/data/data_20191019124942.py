import pandas as pd
from typing import Union
from pathlib import Path
from nameparser import HumanName


class ExtractData:

    def __init__(self, filename: Union[str, Path], drop_columns=None):
        # """Extract Training Data from file or Path

        # Arguments:
        #     filename {[str]} -- Filename of CSV data file containing data.
        #     drop_columns -- Columns in dataframe that should be dropped.
        # """
        if drop_columns is None:
            drop_columns = ["age", "cabin", "name", "ticket"]

        self.filename = filename
        self.drop_columns = drop_columns
        self.all_label_columns = ["survived"]
        self.all_feature_columns = [
            "pclass",
            "name",
            "sex",
            "age",
            "sibsp",
            "parch",
            "ticket",
            "fare",
            "cabin",
            "embarked",
        ]
        self.Xy_raw = None
        self.extract_raw()

    def extract_raw(self):
        """
        Extracts data from a CSV file.  

        Returns:
            pd.DataFrame -- [description]
        """
        Xy_raw = pd.read_csv(self.filename)

        Xy_raw.columns = Xy_raw.columns.str.lower().str.replace(" ", "_")
        Xy_raw = Xy_raw.rename(columns={'age':'age_known'})
        Xy_raw["pclass"] = Xy_raw["pclass"].astype("category")
        self.Xy_raw = Xy_raw.set_index("passengerid")

class TransformData:
    
    title_translator = {
        "Mlle.": "Mrs.",
        "Mme.": "Mrs.",
        "Sir.": "Mr.",
        "Ms.": "Mrs.",
        "": "Mr.",
        "Col.": "Mr.",
        "Capt.": "Mr.",
        "Lady.": "Mrs.",
        "the Countess. of": "Mrs.",
    }

    def __init__(self, raw_data, adult_age_threshold_min = 13, drop_columns=None):
        # """Extract Training Data from file or Path

        # Arguments:
        #     filename {[str]} -- Filename of CSV data file containing data.
        #     drop_columns -- Columns in dataframe that should be dropped.
        # """
        if drop_columns is None:
            drop_columns = ["age", "cabin", "name", "ticket"]

        self.raw = raw_data
        self.adult_age_threshold_min = adult_age_threshold_min

        self.Xy = self.raw.Xy_raw.copy()
        self.extract_title()
        self.extract_last_name()
        self.extract_cabin_number()
        self.extract_cabin_prefix()
        self.estimate_age()
        self.calc_is_child()

    def calc_is_child(self):
        self.Xy['is_child'] = self.Xy.age < self.adult_age_threshold_min
        
    def extract_cabin_number(self):
        self.Xy['cabin_number'] = self.Xy.ticket.str.extract('(\d+)$')
    
    def extract_cabin_prefix(self):
        self.Xy['cabin_prefix'] = self.Xy.ticket.str.extract('^(.+) ')

    def extract_title(self):
        """[summary]
        """
        
        self.Xy["title"] = (
            self.Xy.name.apply(lambda x: HumanName(x).title)
            .replace(self.title_translator)
            .replace({"\.": ""}, regex=True)
        )

    def extract_last_name(self):
        self.Xy["last_name"] = self.Xy.name.apply(lambda x: HumanName(x).last)

    def clean(self,):
        """Clean data to remove missing data and "unnecessary" features.
        
        Arguments:
            in_raw_df {pd.DataFrame} -- Dataframe containing all columns and rows Kaggle Titanic Training Data set
        """
        self.Xy = self.Xy_raw.drop(self.drop_columns, axis=1)


    def estimate_age(self, groupby=['sex','title']):
        Xy_age_estimate = self.Xy.groupby(['sex','title']).age_known.mean().to_frame().round(1)
        Xy_age_estimate = Xy_age_estimate.rename(columns ={'age_known':'age_estimate'})
    
        out_df = self.Xy.reset_index().merge(Xy_age_estimate, on=['sex', 'title'])
        out_df['age'] = out_df['age_known'].fillna(out_df['age_estimate'])
    
    
        return out_df