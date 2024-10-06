import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

    def create_time_of_day_column(self, df):
        if 'Date' in df.columns and 'Time' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
            df.drop(['Date', 'Time'], axis=1, inplace=True)

        if 'Datetime' in df.columns:
            df['hour'] = df['Datetime'].dt.hour
            df['Time_of_Day'] = pd.cut(df['hour'], bins = [0,6,12,17,22,24], labels = ['Night','Morning','Afternoon','Evening','Night'], include_lowest=True, ordered = False)
            df.drop(['Datetime', 'hour'], axis=1, inplace=True)
            day_time = df.pop('Time_of_Day')
            df.insert(0, 'Time_of_Day', day_time)

        return df
    
    def drop_unnecessary_columns(self, df, columns):
        #Removing columns with high correlation (> 0.9)
        df.drop(columns=columns,axis=1,inplace=True)
        return df
        
    def encode_labels(self, X_train, X_test):
        encoder = LabelEncoder()
        X_train['Time_of_Day'] = encoder.fit_transform(X_train['Time_of_Day'])
        X_test['Time_of_Day'] = encoder.transform(X_test['Time_of_Day'])

        return X_train, X_test, encoder
    
    def standardize_data(self, X_train, X_test):
        numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns
        
        scaler = StandardScaler()
        
        # Escalar solo las columnas numéricas
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
        X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])
        
        return X_train_scaled, X_test_scaled, scaler

    def split_data(self, df, target_column, test_size=0.2, random_state=42):
        test_size = self.params['prepare_data']['test_size']
        random_state = self.params['prepare_data']['random_state']

        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def save_data(self, X_train, X_test, y_train, y_test):
        os.makedirs(self.output_dir, exist_ok=True)
        X_train.to_csv(f'{self.output_dir}/X_train.csv', index=False)
        X_test.to_csv(f'{self.output_dir}/X_test.csv', index=False)
        y_train.to_csv(f'{self.output_dir}/y_train.csv', index=False)
        y_test.to_csv(f'{self.output_dir}/y_test.csv', index=False)

    def process_data(self, df, target_column):
        # Crear columna Time_of_Day
        df = self.create_time_of_day_column(df)
        
        # Eliminar columnas innecesarias (Date y Time)
        df = self.drop_unnecessary_columns(df, ['S3_Temp'])
        
        # Dividir en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = self.split_data(df, target_column)
        
        # Estandarizar los datos ya divididos para evitar data leakage
        X_train, X_test, scaler = self.standardize_data(X_train, X_test)
        
        # Codificar las etiquetas de la nueva columna Time_of_Day
        X_train, X_test, encoder = self.encode_labels(X_train, X_test)

        # Guardar los datos procesados
        dp.save_data(X_train, X_test, y_train, y_test)


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
    dp.process_data(df, target_column='Room_Occupancy_Count')