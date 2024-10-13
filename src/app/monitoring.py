import pandas as pd
import mlflow
import numpy as np
from scipy.stats import entropy, ks_2samp
from collections import deque
import threading
import time
import os

class DataDriftMonitor:
    def __init__(self, X_train_path, buffer_size=100, drift_threshold=0.05, num_bins=50):
        self.X_train = pd.read_csv(X_train_path)
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()
        self.drift_threshold = drift_threshold
        self.drift_history = []
        self.num_bins = num_bins
        self.bins = {}

        for column in self.X_train.columns:
            min_val = self.X_train[column].min()
            max_val = self.X_train[column].max()
            if min_val == max_val:
                min_val -= 0.5
                max_val += 0.5
            self.bins[column] = np.linspace(min_val, max_val, self.num_bins + 1)

    def add_live_data(self, live_data):
        with self.buffer_lock:
            self.buffer.append(live_data)

    def kl_divergence(self, p, q, column):
        p_hist, _ = np.histogram(p, bins=self.bins[column], density=True)
        q_hist, _ = np.histogram(q, bins=self.bins[column], density=True)
        p_hist = np.where(p_hist == 0, 1e-10, p_hist)
        q_hist = np.where(q_hist == 0, 1e-10, q_hist)
        return entropy(p_hist, q_hist)

    def ks_statistic(self, p, q):
        statistic, _ = ks_2samp(p, q)
        return statistic

    def check_data_drift(self, force=False):
        drift_metrics = {}
        with self.buffer_lock:
            buffer_length = len(self.buffer)
            if buffer_length == self.buffer.maxlen or force:
                if buffer_length == 0:
                    return drift_metrics
                X_live_buffer = pd.concat(list(self.buffer), ignore_index=True)
                for column in self.X_train.columns:
                    if column in X_live_buffer.columns:
                        p = self.X_train[column].dropna().values
                        q = X_live_buffer[column].dropna().values
                        kl_div = self.kl_divergence(p, q, column)
                        ks_stat = self.ks_statistic(p, q)
                        drift_metrics[column] = {
                            "KL_Divergence": kl_div,
                            "KS_Statistic": ks_stat
                        }
                self.drift_history.append({
                    "timestamp": pd.Timestamp.now(),
                    "metrics": drift_metrics
                })
        return drift_metrics

    def detect_significant_drift(self, drift_metrics):
        significant_drift = {}
        for col, metrics in drift_metrics.items():
            if (metrics["KL_Divergence"] > self.drift_threshold) or (metrics["KS_Statistic"] > self.drift_threshold):
                significant_drift[col] = metrics
        return significant_drift

    def get_drift_history(self):
        return self.drift_history

class MonitoringTask:
    def __init__(self, X_train_path, experiment_name="Monitoring", buffer_size=100, drift_threshold=0.05, interval=300, num_bins=50):
        self.data_monitor = DataDriftMonitor(X_train_path, buffer_size, drift_threshold, num_bins)
        self.experiment_name = experiment_name
        self.interval = interval
        self.stop_event = threading.Event()

    def background_task(self):
        while not self.stop_event.is_set():
            time.sleep(self.interval)
            drift_metrics = self.data_monitor.check_data_drift()
            if drift_metrics:
                significant_drift = self.data_monitor.detect_significant_drift(drift_metrics)
                if significant_drift:
                    mlflow.set_experiment(self.experiment_name)
                    with mlflow.start_run():
                        for col, metrics in significant_drift.items():
                            mlflow.log_metric(f"{col}_KL_Divergence", metrics["KL_Divergence"])
                            mlflow.log_metric(f"{col}_KS_Statistic", metrics["KS_Statistic"])

    def start(self):
        self.thread = threading.Thread(target=self.background_task, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()
