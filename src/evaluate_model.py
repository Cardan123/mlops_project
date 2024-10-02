import pandas as pd
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn

X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')
model = mlflow.sklearn.load_model("models:/occupancy_classification/production")
predictions = model.predict(X_test)
report = classification_report(y_test, predictions)
print(report)