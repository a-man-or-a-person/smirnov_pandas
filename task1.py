import pandas as pd
import numpy as np


class Preparation:
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path)
        self._sep_obj_and_num()

    def _sep_obj_and_num(self) -> None:
        self.num_features = [
            col
            for col in self.df.columns
            if (self.df[col].dtype != "object" and len(self.df[col].unique()) >= 25)
            or col in ("day", "month", "year")
        ]
        self.obj_features = [
            col for col in self.df.columns if col not in self.num_features
        ]

    def fill_nan(self) -> None:
        for col in self.num_features:
            self.df[col] = self.df[col].fillna(self.df[col].median())

    def mean(self) -> str:
        return self.df[self.num_features].mean().to_string()

    def prep_obj_features(self) -> None:
        self.df = pd.get_dummies(data=self.df, columns=self.obj_features)
        self._sep_obj_and_num()

    def create_dttm(self):
        self.df[["day", "month", "year"]] = self.df["Date"].str.split("/", expand=True)
        for col in ["day", "month", "year"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        self.df = self.df.drop("Date", axis=1)
        self._sep_obj_and_num()

    def del_high_corr_features(self):
        corr_matrix = self.df[self.num_features].corr().abs()
        corr_features_triu = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape)).astype(bool)
        )
        corr_features = []
        for col in corr_features_triu.columns:
            if any((corr_features_triu[col] > 0.5) & (corr_features_triu[col] != 1)):
                corr_features.append(col)

        existing_corr_features = [col for col in corr_features if col in self.df.columns]
        self.df.drop(columns=existing_corr_features, inplace=True)
        self._sep_obj_and_num()

    def __str__(self):
        print(self.obj_features, self.num_features)
        return self.df.head().to_string()


pre = Preparation("BikeData.csv")
pre.create_dttm()
pre.del_high_corr_features()
print(pre.mean())
print(pre)
