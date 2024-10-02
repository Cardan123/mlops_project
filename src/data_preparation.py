import pandas as pd
from sklearn.model_selection import train_test_split
import dvc.api

data_url = dvc.api.get_url(path='data/raw/occupancy_data.csv')

def load_and_clean_data(data_url):
 df = pd.read_csv(data_url)
 df = df.dropna()
 X = df.drop('Occupancy', axis=1)
 y = df['Occupancy']
 return X, y

def split_data(X, y):
 return train_test_split(X, y, test_size=0.2, random_state=42)

def save_processed_data(X_train, X_test, y_train, y_test):
 X_train.to_csv('data/processed/X_train.csv', index=False)
 X_test.to_csv('data/processed/X_test.csv', index=False)
 y_train.to_csv('data/processed/y_train.csv', index=False)
 y_test.to_csv('data/processed/y_test.csv', index=False)
 
if __name__ == "__main__":
 X, y = load_and_clean_data(data_url)
 X_train, X_test, y_train, y_test = split_data(X, y)
 save_processed_data(X_train, X_test, y_train, y_test)
