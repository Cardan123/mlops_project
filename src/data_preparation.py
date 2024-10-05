import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse
import yaml

class DataPreparation:
    def __init__(self, input_path, output_dir, params):
        self.input_path = input_path
        self.output_dir = output_dir
        self.params = params

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
        test_size = self.params['prepare_data']['test_size']
        random_state = self.params['prepare_data']['random_state']
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def save_data(self, X_train, X_test, y_train, y_test):
        os.makedirs(self.output_dir, exist_ok=True)
        X_train.to_csv(f'{self.output_dir}/X_train.csv', index=False)
        X_test.to_csv(f'{self.output_dir}/X_test.csv', index=False)
        y_train.to_csv(f'{self.output_dir}/y_train.csv', index=False)
        y_test.to_csv(f'{self.output_dir}/y_test.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data")
    parser.add_argument("--input_path", type=str, help="Path to raw data")
    parser.add_argument("--output_dir", type=str, help="Directory for processed data")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    with open(args.params, 'r') as fd:
        params = yaml.safe_load(fd)

    dp = DataPreparation(args.input_path, args.output_dir, params)
    df = dp.load_data()
    X, y = dp.preprocess_data(df)
    X_train, X_test, y_train, y_test = dp.split_data(X, y)
    dp.save_data(X_train, X_test, y_train, y_test)
