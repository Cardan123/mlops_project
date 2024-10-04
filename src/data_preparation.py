import pandas as pd
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv

# Load the environment variables
stage = os.getenv('ENV', 'dev')
load_dotenv(f".env.{stage}")

class DataPreparation:
    def __init__(self, input_path, output_dir):
        self.input_path = input_path
        self.output_dir = output_dir

    def load_data(self):
        return pd.read_csv(self.input_path)

    def preprocess_data(self, df):
        if 'Date' in df.columns and 'Time' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df.drop(['Date', 'Time'], axis=1, inplace=True)

        if 'Datetime' in df.columns:
            df['hour'] = df['Datetime'].dt.hour
            df['day'] = df['Datetime'].dt.day
            df['month'] = df['Datetime'].dt.month
            df.drop(['Datetime'], axis=1, inplace=True)

        df = df.select_dtypes(include=[float, int])

        X = df.drop('Room_Occupancy_Count', axis=1)
        y = df['Room_Occupancy_Count']
        return X, y

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def save_data(self, X_train, X_test, y_train, y_test):
        X_train.to_csv(f'{self.output_dir}/X_train.csv', index=False)
        X_test.to_csv(f'{self.output_dir}/X_test.csv', index=False)
        y_train.to_csv(f'{self.output_dir}/y_train.csv', index=False)
        y_test.to_csv(f'{self.output_dir}/y_test.csv', index=False)

if __name__ == "__main__":
    input_path = os.getenv("RAW_DATA_PATH")
    output_dir = os.getenv("PROCESSED_DATA_PATH")

    dp = DataPreparation(input_path, output_dir)
    df = dp.load_data()
    X, y = dp.preprocess_data(df)
    X_train, X_test, y_train, y_test = dp.split_data(X, y)
    dp.save_data(X_train, X_test, y_train, y_test)
