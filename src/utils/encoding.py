from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class OneHotEncoderHandler(object):

    def __init__(self, handle_unknown="ignore"):
        self.enc = OneHotEncoder(handle_unknown=handle_unknown)

    def fit(self, data):
        self.columns_binned = list(data)
        self.enc.fit(data.astype(str))
        self.columns_dummies = list(pd.get_dummies(data).columns)

    def transform(self, data):
        return pd.DataFrame(self.enc.transform(data.astype(str)).toarray().astype(int), columns=self.columns_dummies)
