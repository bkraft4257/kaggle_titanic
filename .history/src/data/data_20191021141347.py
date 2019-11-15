import pandas as pd
import numpy as np
from typing import Union
from pathlib import Path
from nameparser import HumanName


class ExtractData:
    def __init__(self, filename: Union[str, Path], age_bins = None, drop_columns=None):
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
        Xy_raw = Xy_raw.rename(columns={"age": "age_known"})
        Xy_raw["pclass"] = Xy_raw["pclass"].astype("category")
        self.Xy_raw = Xy_raw.set_index("passengerid")


class TransformData:

    title_translator = {
        "Mlle.": "Mrs.",
        "Mme.": "Mrs.",
        "Sir.": "Mr.",
        "Ms.": "Mrs.",
        "Rev.": "Mr.",
        "": "Mr.",
        "Col.": "Mr.",
        "Capt.": "Mr.",
        "Lady.": "Mrs.",
        "the Countess. of": "Mrs.",
    }

    def __init__(
        self,
        raw_data,
        adult_age_threshold_min=13,
        age_bins = None,
        Xy_age_estimate=None,
        drop_columns=None,
    ):
        # """Extract Training Data from file or Path

        # Arguments:
        #     filename {[str]} -- Filename of CSV data file containing data.
        #     drop_columns -- Columns in dataframe that should be dropped.
        # """

        if age_bins is None:
            age_bins = [0,10,20,30, 40, 50, 60, np.inf]

        if drop_columns is None:
            drop_columns = ["age", "cabin", "name", "ticket"]

        self.raw = raw_data
        self.adult_age_threshold_min = adult_age_threshold_min
        self.Xy_age_estimate = Xy_age_estimate
        self.age_bins = age_bins

        self.Xy = self.raw.Xy_raw.copy()
        self.extract_title()
        self.extract_last_name()
        self.extract_cabin_number()
        self.extract_cabin_prefix()
        self.estimate_age()
        self.calc_age_bins()
        self.calc_is_child()
        self.calc_is_travelling_alone()

    def calc_is_travelling_alone(self):
        self.Xy["is_travelling_alone"] = (self.Xy.sibsp == 0) & (self.Xy.parch == 0)

    def calc_is_child(self):
        self.Xy["is_child"] = self.Xy.age < self.adult_age_threshold_min

    def extract_cabin_number(self):
        self.Xy["cabin_number"] = self.Xy.ticket.str.extract("(\d+)$")

    def extract_cabin_prefix(self):
        self.Xy["cabin_prefix"] = self.Xy.ticket.str.extract("^(.+) ")

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

    def calc_age_bins(self):
        self.Xy['age_bin'] = pd.cut(self.Xy.age, bins=[0,10,20,30, 40, 50, 60, np.inf])

    def clean(self,):
        """Clean data to remove missing data and "unnecessary" features.
        
        Arguments:
            in_raw_df {pd.DataFrame} -- Dataframe containing all columns and rows Kaggle Titanic Training Data set
        """
        self.Xy = self.Xy_raw.drop(self.drop_columns, axis=1)

    def estimate_age(self, groupby_columns=["sex", "title"]):
        """[summary]
        
        Keyword Arguments:
            groupby {list} -- [description] (default: {['sex','title']})
        """

        if self.Xy_age_estimate is None:
            self.Xy_age_estimate = (
                self.Xy.groupby(groupby_columns).age_known.mean().to_frame().round(1)
            )

            self.Xy_age_estimate = self.Xy_age_estimate.rename(
                columns={"age_known": "age_estimate"}
            )

        out_df = self.Xy.reset_index().merge(self.Xy_age_estimate, on=groupby_columns)
        out_df["age"] = out_df["age_known"].fillna(out_df["age_estimate"])

        self.Xy = out_df

    def impute_missing_fare():
        pass

    def impute_missing_embark():
        pass