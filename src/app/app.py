from flask import Flask, request, jsonify
import pandas as pd
import mlflow.pyfunc
import mlflow
import threading
import os
import random
from dotenv import load_dotenv
from monitoring import MonitoringTask
from datetime import datetime

app = Flask(__name__)

stage = os.getenv('ENV', 'dev')
load_dotenv(f".env.{stage}")

mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
mlflow.set_tracking_uri(mlflow_tracking_uri)

model_lock = threading.Lock()
model = None

monitoring_task = None
monitoring_lock = threading.Lock()
is_monitoring_running = False

def load_latest_production_model():
    model_name = "BestModel"
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return None
        latest_version = max(versions, key=lambda v: int(v.version))
        model_uri = f"models:/{model_name}/{latest_version.version}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        return loaded_model
    except mlflow.exceptions.MlflowException:
        return None
    except Exception:
        return None

def initialize_model():
    global model
    with model_lock:
        model = load_latest_production_model()

def generate_random_data():
    return {
        "S1_Temp": round(random.uniform(24.0, 26.0), 2),
        "S2_Temp": round(random.uniform(24.0, 26.0), 2),
        "S3_Temp": round(random.uniform(24.0, 26.0), 2),
        "S4_Temp": round(random.uniform(24.0, 26.0), 2),
        "S1_Light": random.randint(30, 150),
        "S2_Light": random.randint(30, 150),
        "S3_Light": random.randint(30, 150),
        "S4_Light": random.randint(30, 150),
        "S1_Sound": round(random.uniform(0, 1), 2),
        "S2_Sound": round(random.uniform(0, 1), 2),
        "S3_Sound": round(random.uniform(0, 1), 2),
        "S4_Sound": round(random.uniform(0, 1), 2),
        "S5_CO2": random.randint(300, 400),
        "S5_CO2_Slope": round(random.uniform(0, 1), 2),
        "S6_PIR": random.randint(0, 1),
        "S7_PIR": random.randint(0, 1),
        "Room_Occupancy_Count": random.randint(0, 5)
    }

def complete_data_with_random_values(data):
    default_data = generate_random_data()
    for key in default_data:
        if key not in data:
            data[key] = default_data[key]
    return data

def add_time_of_day_column(df):
    current_hour = datetime.now().hour
    if 0 <= current_hour < 6:
        time_of_day = 'Night'
    elif 6 <= current_hour < 12:
        time_of_day = 'Morning'
    elif 12 <= current_hour < 18:
        time_of_day = 'Afternoon'
    else:
        time_of_day = 'Evening'

    time_of_day_mapping = {'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3}
    df['Time_of_Day'] = time_of_day_mapping.get(time_of_day, 3)

    return df

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please check the MLflow Model Registry."}), 500

    try:
        data = request.json
        if not data:
            data = generate_random_data()
        else:
            data = complete_data_with_random_values(data)

        df = pd.DataFrame([data])
        df = add_time_of_day_column(df)
        df.drop(columns=['S3_Temp', 'Room_Occupancy_Count'], inplace=True, errors='ignore')
        df['Time_of_Day'] = df['Time_of_Day'].astype('int64')
        required_columns = ['S1_Temp', 'S2_Temp', 'S4_Temp', 'S1_Light', 'S2_Light', 'S3_Light', 'S4_Light',
                            'S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound', 'S5_CO2', 'S5_CO2_Slope', 
                            'S6_PIR', 'S7_PIR']
        for col in required_columns:
            if col in df.columns:
                df[col] = df[col].astype('float64')

        with model_lock:
            predictions = model.predict(df)

        monitoring_task.data_monitor.add_live_data(df)

        return jsonify({"input_data": data, "predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/reload_model', methods=['POST'])
def reload_model():
    try:
        new_model = load_latest_production_model()
        if new_model is not None:
            with model_lock:
                global model
                model = new_model
            return jsonify({"status": "Model reloaded successfully."}), 200
        else:
            return jsonify({"error": "Failed to load the new model."}), 500
    except Exception as e:
        return jsonify({"error": f"Exception during model reload: {str(e)}"}), 500

@app.route('/monitoring_status', methods=['GET'])
def monitoring_status():
    with monitoring_lock:
        status = "running" if is_monitoring_running else "not running"
    return jsonify({"status": f"Monitoring thread is {status}"}), 200

@app.route('/buffer_size', methods=['GET'])
def buffer_size():
    if monitoring_task and monitoring_task.data_monitor:
        buffer_length = len(monitoring_task.data_monitor.buffer)
    else:
        buffer_length = 0
    return jsonify({"buffer_size": buffer_length}), 200

@app.route('/drift_metrics', methods=['GET'])
def drift_metrics():
    if monitoring_task and monitoring_task.data_monitor:
        history = monitoring_task.data_monitor.get_drift_history()
        serialized_history = [
            {
                "timestamp": entry["timestamp"].isoformat(),
                "metrics": entry["metrics"]
            }
            for entry in history
        ]
    else:
        serialized_history = []
    return jsonify({"drift_history": serialized_history}), 200

@app.route('/trigger_drift_analysis', methods=['POST'])
def trigger_drift_analysis():
    try:
        if not monitoring_task or not monitoring_task.data_monitor:
            return jsonify({"error": "Monitoring task is not initialized."}), 500

        drift_metrics = monitoring_task.data_monitor.check_data_drift(force=True)
        
        if drift_metrics:
            significant_drift = monitoring_task.data_monitor.detect_significant_drift(drift_metrics)

            if significant_drift:
                mlflow.set_experiment(monitoring_task.experiment_name)
                with mlflow.start_run():
                    for col, metrics in significant_drift.items():
                        mlflow.log_metric(f"{col}_KL_Divergence", metrics["KL_Divergence"])
                        mlflow.log_metric(f"{col}_KS_Statistic", metrics["KS_Statistic"])
                return jsonify({
                    "status": "Data drift detected and logged",
                    "significant_drift": significant_drift
                }), 200
            else:
                return jsonify({
                    "status": "No significant data drift detected",
                    "drift_metrics": drift_metrics
                }), 200
        else:
            return jsonify({"status": "Not enough data for drift analysis"}), 400

    except Exception as e:
        return jsonify({"error": f"Drift analysis failed: {str(e)}"}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500
    try:
        model_uri = model.metadata.get("artifact_path", "Unknown")
        return jsonify({"model_uri": model_uri}), 200
    except AttributeError:
        return jsonify({"model_info": "Model loaded successfully."}), 200

def start_monitoring():
    global monitoring_task
    global is_monitoring_running

    X_train_path = os.getenv('X_TRAIN_PATH', '../../data/processed/X_train.csv')

    monitoring_task = MonitoringTask(
        X_train_path=X_train_path,
        experiment_name="Monitoring",
        buffer_size=100,
        drift_threshold=0.05,
        interval=300,
        num_bins=50
    )

    def run_monitoring():
        global is_monitoring_running
        with monitoring_lock:
            is_monitoring_running = True
        try:
            monitoring_task.start()
            monitoring_task.thread.join()
        except Exception:
            print("Monitoring task failed.")
        finally:
            with monitoring_lock:
                is_monitoring_running = False

    monitoring_thread = threading.Thread(target=run_monitoring, daemon=True)
    monitoring_thread.start()

if __name__ == "__main__":
    initialize_model()
    start_monitoring()
    app.run(host="0.0.0.0", port=5000)
