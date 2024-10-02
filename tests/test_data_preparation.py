import pandas as pd
from src.data_preparation import load_and_clean_data

def test_load_and_clean_data():
 X, y = load_and_clean_data("data/raw/occupancy_data.csv")
 assert len(X) == len(y)
 assert not X.isnull().values.any()
