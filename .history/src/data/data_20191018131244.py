import pandas as pd
from typing import Union
from pathlib import Path
from nameparser import HumanName


class ExtractData:

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

    def __init__(self, filename: Union[str, Path], drop_columns=None) -> None:
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
        self.Xy = None

        self.extract_raw()
        self.extract_title()
        self.extract_last_name()
        self.clean()

    def extract_raw(self) -> pd.DataFrame:
        """
        Extracts data from a CSV file.  

        Returns:
            pd.DataFrame -- [description]
        """
        Xy_raw = pd.read_csv(self.filename)
        Xy_raw.columns = Xy_raw.columns.str.lower().str.replace(" ", "_")

        Xy_raw["pclass"] = Xy_raw["pclass"].astype("category")
        self.Xy_raw = Xy_raw.set_index("passengerid")

    def extract_title(self):
        self.Xy["title"] = (
            self.Xy.name.apply(lambda x: HumanName(x).title)
            .replace(self.title_translator)
            .replace({"\.": ""}, regex=True)
        )

    def extract_title(self):
        self.Xy["last_name"] = self.Xy.name.apply(lambda x: HumanName(x).last)

    def clean(self,):
        """Clean data to remove missing data and "unnecessary" features.
        
        Arguments:
            in_raw_df {pd.DataFrame} -- Dataframe containing all columns and rows Kaggle Titanic Training Data set
        """
        self.Xy = self.Xy_raw.drop(self.drop_columns, axis=1)
