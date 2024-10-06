import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import argparse
import yaml

class DataPreparation:
    def __init__(self, input_path, output_dir, params):
        # Initialize the DataPreparation class with input file path, output directory, and parameters
        self.input_path = input_path
        self.output_dir = output_dir
        self.params = params

    def load_data(self):
        # Load data from a CSV file and return it as a DataFrame
        return pd.read_csv(self.input_path)

    def create_time_of_day_column(self, df):
        # Create a 'Time_of_Day' column based on the 'Date' and 'Time' columns
        if 'Date' in df.columns and 'Time' in df.columns:
            # Combine 'Date' and 'Time' columns into a single 'Datetime' column
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            # Drop original 'Date' and 'Time' columns as they are no longer needed
            df.drop(['Date', 'Time'], axis=1, inplace=True)

        if 'Datetime' in df.columns:
            # Extract hour from the 'Datetime' column
            df['hour'] = df['Datetime'].dt.hour
            # Create 'Time_of_Day' categories based on the hour
            df['Time_of_Day'] = pd.cut(df['hour'], bins=[0, 6, 12, 17, 22, 24], 
                                        labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Night'], 
                                        include_lowest=True, ordered=False)
            # Drop 'Datetime' and 'hour' columns as they are no longer needed
            df.drop(['Datetime', 'hour'], axis=1, inplace=True)
            # Move 'Time_of_Day' to the first column
            day_time = df.pop('Time_of_Day')
            df.insert(0, 'Time_of_Day', day_time)

        return df
    
    def drop_unnecessary_columns(self, df, columns):
        # Remove specified unnecessary columns from the DataFrame
        df.drop(columns=columns, axis=1, inplace=True)
        return df
        
    def encode_labels(self, X_train, X_test):
        # Encode categorical labels using LabelEncoder
        encoder = LabelEncoder()
        # Fit the encoder on the training set and transform both training and testing sets
        X_train['Time_of_Day'] = encoder.fit_transform(X_train['Time_of_Day'])
        X_test['Time_of_Day'] = encoder.transform(X_test['Time_of_Day'])

        return X_train, X_test, encoder
    
    def standardize_data(self, X_train, X_test):
        # Standardize numerical columns in the training and testing datasets
        numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
        
        scaler = StandardScaler()
        
        # Scale only the numeric columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
        X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])
        
        return X_train_scaled, X_test_scaled, scaler

    def split_data(self, df, target_column, test_size=0.2, random_state=42):
        # Split the DataFrame into training and testing sets
        test_size = self.params['prepare_data']['test_size']
        random_state = self.params['prepare_data']['random_state']

        X = df.drop(columns=[target_column])  # Features
        y = df[target_column]  # Target variable
        # Use train_test_split to create training and testing datasets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def save_data(self, X_train, X_test, y_train, y_test):
        # Save processed datasets to CSV files in the specified output directory
        os.makedirs(self.output_dir, exist_ok=True)  # Create output directory if it doesn't exist
        X_train.to_csv(f'{self.output_dir}/X_train.csv', index=False)
        X_test.to_csv(f'{self.output_dir}/X_test.csv', index=False)
        y_train.to_csv(f'{self.output_dir}/y_train.csv', index=False)
        y_test.to_csv(f'{self.output_dir}/y_test.csv', index=False)

    def process_data(self, df, target_column):
        # Process the data by creating new features, dropping unnecessary columns, splitting data, 
        # standardizing it, and encoding labels
        df = self.create_time_of_day_column(df)  # Create 'Time_of_Day' column
        df = self.drop_unnecessary_columns(df, ['S3_Temp'])  # Drop unnecessary columns (Due to high correlation)
        X_train, X_test, y_train, y_test = self.split_data(df, target_column)  # Split data
        X_train, X_test, scaler = self.standardize_data(X_train, X_test)  # Standardize data
        X_train, X_test, encoder = self.encode_labels(X_train, X_test)  # Encode categorical labels
        self.save_data(X_train, X_test, y_train, y_test)  # Save processed data


if __name__ == "__main__":
    # Entry point of the script
    parser = argparse.ArgumentParser(description="Prepare data")  # Create argument parser
    parser.add_argument("--input_path", type=str, help="Path to raw data")  # Input data path argument
    parser.add_argument("--output_dir", type=str, help="Directory for processed data")  # Output directory argument
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to params.yaml")  # Parameters file argument
    args = parser.parse_args()  # Parse the arguments

    # Load parameters from YAML file
    with open(args.params, 'r') as fd:
        params = yaml.safe_load(fd)

    dp = DataPreparation(args.input_path, args.output_dir, params)  # Create an instance of DataPreparation
    df = dp.load_data()  # Load the raw data
    dp.process_data(df, target_column='Room_Occupancy_Count')  # Process the data
