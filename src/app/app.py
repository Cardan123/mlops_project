from flask import Flask, request, jsonify
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from collections import deque
import threading
from monitoring import background_monitoring_task, buffer, buffer_lock

app = Flask(__name__)

def load_latest_model():
    model_name = "BestModel"
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        return model
    except MlflowException as e:
        print(f"Error loading model: {e}")
        return None

model = load_latest_model()

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please check the MLflow Model Registry."}), 500
    
    try:
        data = request.json
        df = pd.DataFrame([data]) 
        
        predictions = model.predict(df)
        
        with buffer_lock:
            buffer.append(df)

        return jsonify({"predictions": predictions.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    monitoring_thread = threading.Thread(target=background_monitoring_task, daemon=True)
    monitoring_thread.start()

    app.run(host="0.0.0.0", port=5000)
