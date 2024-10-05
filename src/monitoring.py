import pandas as pd
import mlflow
import numpy as np
from scipy.stats import entropy
from collections import deque
import threading
import time

buffer = deque(maxlen=100) 
buffer_lock = threading.Lock()

def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    return entropy(p, q)

def check_data_drift(X_train, X_live):
    drift_metrics = {}
    for column in X_train.columns:
        if column in X_live.columns:
            kl_div = kl_divergence(X_train[column], X_live[column])
            drift_metrics[column] = kl_div
    return drift_metrics

X_train = pd.read_csv("data/processed/X_train.csv")

def background_monitoring_task():
    while True:
        time.sleep(5)  
        with buffer_lock:
            if len(buffer) == buffer.maxlen:
                X_live_buffer = pd.concat(list(buffer), ignore_index=True)

                data_drift_metrics = check_data_drift(X_train, X_live_buffer)
                
                mlflow.set_experiment("Monitoring")
                with mlflow.start_run():
                    mlflow.log_metrics(data_drift_metrics)
                
                print(f"Data drift metrics logged: {data_drift_metrics}")
