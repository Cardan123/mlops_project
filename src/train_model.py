import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv

stage = os.getenv('ENV', 'dev')
load_dotenv(f".env.{stage}")

class ModelTrainer:
    def __init__(self, X_train_path, y_train_path, experiment_name="My RandomForest Experiment"):
        self.X_train_path = X_train_path
        self.y_train_path = y_train_path
        self.experiment_name = experiment_name

    def load_data(self):
        X_train = pd.read_csv(self.X_train_path)
        y_train = pd.read_csv(self.y_train_path)
        return X_train, y_train

    def train_model(self, X_train, y_train):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.values.ravel())
        return model

    def log_model_and_metrics(self, model, X_train, y_train):
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run() as run:
            input_example = X_train.head(1)

            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("random_state", model.random_state)

            y_pred_train = model.predict(X_train)

            feature_importances = model.feature_importances_
            for i, importance in enumerate(feature_importances):
                mlflow.log_metric(f"feature_importance_{i}", importance)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="random_forest_model",
                input_example=input_example,
                registered_model_name="RandomForestClassifier"
            )
            
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/random_forest_model",
                name="RandomForestClassifier",
                
            )

if __name__ == "__main__":
    X_train_path = os.getenv("PROCESSED_DATA_PATH") + "/X_train.csv"
    y_train_path = os.getenv("PROCESSED_DATA_PATH") + "/y_train.csv"

    mt = ModelTrainer(X_train_path, y_train_path)
    X_train, y_train = mt.load_data()
    model = mt.train_model(X_train, y_train)

    mt.log_model_and_metrics(model, X_train, y_train)
