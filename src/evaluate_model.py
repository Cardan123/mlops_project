import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import os
from dotenv import load_dotenv

stage = os.getenv('ENV', 'dev')
load_dotenv(f".env.{stage}")

class ModelEvaluator:
    def __init__(self, X_test_path, y_test_path, model_path):
        self.X_test_path = X_test_path
        self.y_test_path = y_test_path
        self.model_path = model_path

    def load_data(self):
        X_test = pd.read_csv(self.X_test_path)
        y_test = pd.read_csv(self.y_test_path)
        return X_test, y_test

    def load_model(self):
        return mlflow.sklearn.load_model(self.model_path)

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def log_metrics(self, mse, r2):
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

if __name__ == "__main__":
    X_test_path = os.getenv("PROCESSED_DATA_PATH") + "/X_test.csv"
    y_test_path = os.getenv("PROCESSED_DATA_PATH") + "/y_test.csv"
    model_path = os.getenv("MODEL_URI", "models:/RandomForestClassifier/latest")
    
    me = ModelEvaluator(X_test_path, y_test_path, model_path)
    X_test, y_test = me.load_data()
    model = me.load_model()
    mse, r2 = me.evaluate_model(model, X_test, y_test)
    me.log_metrics(mse, r2)
