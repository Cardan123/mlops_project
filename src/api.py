from fastapi import FastAPI
import mlflow
import pandas as pd

app = FastAPI()
model = mlflow.sklearn.load_model("models:/occupancy_classification/production")
@app.post("/predict")
def predict(data: dict):
 df = pd.DataFrame([data])
 prediction = model.predict(df)
 return {"prediction": prediction[0]}