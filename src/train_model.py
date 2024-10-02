import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow

X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

mlflow.set_tracking_uri('http://mlflow_server:5000')
mlflow.set_experiment('occupancy_classification')

with mlflow.start_run():
 model = RandomForestClassifier()
 model.fit(X_train, y_train)
 mlflow.sklearn.log_model(model, "model")
 mlflow.log_metric("accuracy", model.score(X_train, y_train))