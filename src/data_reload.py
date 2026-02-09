import pandas as pd

def load_dataset():
    df = pd.read_csv("data/processed_property.csv")

    feature_cols = ['Area', 'Sale_Year', 'Sale_Month']
    target_col = 'Price_per_unit'

    X = df[feature_cols]
    y = df[target_col]

    return X, y
