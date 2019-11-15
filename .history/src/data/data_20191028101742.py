import pandas as pd
import numpy as np
from typing import Union
from pathlib import Path
from nameparser import HumanName


class ExtractData:
    """Extract Titanic data from the Kaggle's train.csv file.
    """

    def __init__(self, filename: Union[str, Path], age_bins=None, drop_columns=None):
        """Extract Training Data from filename (string or Path object)

        Arguments:
            filename {[str]} -- Filename of CSV data file containing data.
            drop_columns -- Columns in dataframe that should be dropped.
        """
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

    """ 
    TransformData takes the raw extracted data cleans and creates new features before
    returning a new dataframe. 

    The training and test data contain the following:
        * 1 Lady. She was traveling with a sibling and no husband. Set title to Miss
        * 2 Mlle and 1 Mme. All 3 were 24 years old and travelling alone.  Retitled as Miss.
        * 1 Sir. Male 49 years old. Travelling with a sibling.
        * Revs were all males.
        * 8 Drs. (7 male, 1 female) changed to Mr. and Mrs. respectively.
    """

    title_translator = {
        "Mlle.": "Miss.",
        "Mme.": "Miss.",
        "Sir.": "Mr.",
        "Ms.": "Mrs.",
        "Rev.": "Mr.",
        "Col.": "Mr.",
        "Capt.": "Mr.",
        "Lady.": "Miss.",
        "the Countess. of": "Mrs.",
        "Dr.": np.nan,
    }

    def __init__(
        self,
        raw_data,
        adult_age_threshold_min=13,
        age_bins=None,
        fare_mode=None,
        embarked_mode=None,
        Xy_age_estimate=None,
        drop_columns=None,
    ):
        """Extract Training Data from file or Path

        Arguments:
             filename {[str]} -- Filename of CSV data file containing data.
             drop_columns -- Columns in dataframe that should be dropped.
        """

        if age_bins is None:
            age_bins = [0, 10, 20, 30, 40, 50, 60, np.inf]



        self.raw = raw_data
        self.adult_age_threshold_min = adult_age_threshold_min
        self.drop_columns = drop_columns
        self.π = Xy_age_estimate
        self.age_bins = age_bins

        self.Xy = self.raw.Xy_raw.copy()

        if fare_mode is None:
            fare_mode = self.Xy["fare"].mode()[0]

        if embarked_mode is None:
            embarked_mode = self.Xy["embarked"].mode()[0]

        self.fare_mode = fare_mode
        self.embarked_mode = embarked_mode

        self.impute_missing_fare()
        self.impute_missing_embarked()

        self.extract_title()
        self.extract_last_name()
        self.extract_cabin_number()
        self.extract_cabin_prefix()
        self.calc_family_size()
        self.estimate_age()
        self.calc_age_bins()
        self.calc_is_child()
        self.calc_is_travelling_alone()

        self.Xy = self.Xy.sort_index()

    def calc_family_size(self):
        """Create feature family size, which is the number of people (including
        self) that are traveling together. 
        """
        self.Xy["family_size"] = self.Xy.sibsp + self.Xy.parch + 1
        
    def calc_is_travelling_alone(self):
        """Create Boolean feature if passenger is travelling alone. (True=Traveling alone, False=Traveling in group)
        """
        self.Xy["is_travelling_alone"] = self.Xy["family_size"] == 1

    def calc_is_child(self):
        """Calculate Boolean feature if passenger is a child as determined by the self.adult_age_threshold_min
        """
        self.Xy["is_child"] = self.Xy.age < self.adult_age_threshold_min

    def extract_cabin_number(self):
        """ 
        Extracts cabin number from ticket. 
        """
        self.Xy["cabin_number"] = self.Xy.ticket.str.extract("(\d+)$")

    def extract_cabin_prefix(self):
        """Extracts cabin prefix from ticket.
        """
        self.Xy["cabin_prefix"] = self.Xy.ticket.str.extract("^(.+) ")

    def extract_title(self):
        """Extract title from the name using nameparser.

        If the Title is empty then we will fill the title with either Mr or Mrs depending upon the sex.  This 
        is adequate for the train and holdout data sets.  The title being empty only occurs for passenger 1306
        in the holdout data set.  A more appropriate way to do this is to check on the sex and age to correctly 
        assign the title
        """
        title = (
            self.Xy.name.apply(lambda x: HumanName(x).title)
            .replace(self.title_translator)
            .replace({"\.": ""}, regex=True)
            .replace({"": np.nan})
            .fillna(self.Xy["sex"])
            .replace({"female": "Mrs", "male": "Mr"})
        )

        self.Xy["title"] = title

    def extract_last_name(self):
        "Extracts last name from name feature using nameparser."
        self.Xy["last_name"] = self.Xy.name.apply(lambda x: HumanName(x).last)

    def calc_age_bins(self):
        """Calculates age bins. 
        """
        self.Xy["age_bin"] = pd.cut(self.Xy.age, bins=self.age_bins)

    def clean(self,):
        """Clean data to remove missing data and "unnecessary" features.
        
        Arguments:
            in_raw_df {pd.DataFrame} -- Dataframe containing all columns and rows Kaggle Titanic Training Data set
        """
        self.Xy = self.Xy_raw.drop(self.drop_columns, axis=1)

    def estimate_age(self, groupby_columns=["sex", "title"]):
        """Estimate age of passenger when age is unknown.   The age will be estimated according to the 
        group as specified in the groupby_columns. 
        
        Keyword Arguments:
            groupby_columns {list} -- [description] (default: {["sex", "title"]})
        """

        if self.Xy_age_estimate is None:
            self.Xy_age_estimate = (
                self.Xy.groupby(groupby_columns).age_known.mean().to_frame().round(1)
            )

            self.Xy_age_estimate = self.Xy_age_estimate.rename(
                columns={"age_known": "age_estimate"}
            )

        out_df = (
            self.Xy.reset_index()
            .merge(self.Xy_age_estimate, on=groupby_columns)
            .set_index("passengerid")
        )

        out_df["age"] = out_df["age_known"].fillna(out_df["age_estimate"])

        self.Xy = out_df

    def impute_missing_fare(self):
        """Imputes missing fare based upon only the most frequent fare. This 
        could be improved by looking at additional features. In particular, 
        the number of passengers in the party and pclass. 
        """
        self.Xy["fare"] = self.Xy["fare"].fillna(self.fare_mode)

    def impute_missing_embarked(self):
        """Imputes missing embarkment location based upon the most frequent 
        place to embark.
        """
        self.Xy["embarked"] = self.Xy["embarked"].fillna(self.embarked_mode)
