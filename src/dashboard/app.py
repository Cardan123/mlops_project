import streamlit as st
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking import MlflowClient
import os
import numpy as np
from dotenv import load_dotenv
import requests

THEME_COLOR = "#FFA07A"
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:5000')

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

st.set_page_config(layout="wide")
st.title("🚀 MLflow Experiment Dashboard")

def get_experiments(client):
    return client.search_experiments()

def get_experiment_id(client, experiment_name):
    return client.get_experiment_by_name(experiment_name).experiment_id

def get_runs(client, experiment_id):
    return client.search_runs(experiment_ids=[experiment_id])

def is_experiment_type(experiment_name, type_name):
    return type_name in experiment_name

def section_divider():
    st.markdown("---")

def format_run_data(runs, is_training_experiment):
    runs_data = []
    for run in runs:
        run_data = {
            'Run ID': run.info.run_id,
            'Model Name': run.info.run_name if is_training_experiment else "Best Model",
            'Status': run.info.status,
            'Start Time': run.info.start_time,
            'End Time': run.info.end_time,
            'Duration (s)': (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else None,
        }
        run_data.update(run.data.metrics)
        runs_data.append(run_data)
    return pd.DataFrame(runs_data)

def fetch_drift_data():
    try:
        response = requests.get(f'{API_BASE_URL}/drift_metrics')
        if response.status_code == 200:
            drift_history = response.json().get('drift_history')
            if drift_history:
                drift_data = []
                for entry in drift_history:
                    timestamp = entry['timestamp']
                    for sensor, metrics in entry['metrics'].items():
                        kl_divergence = metrics.get("KL_Divergence", np.nan)
                        ks_statistic = metrics.get("KS_Statistic", np.nan)
                        drift_data.append({
                            "timestamp": timestamp,
                            f"{sensor}_KL_Divergence": kl_divergence,
                            f"{sensor}_KS_Statistic": ks_statistic
                        })
                return pd.DataFrame(drift_data)
            else:
                st.warning("No drift history available.")
        else:
            st.error(f"Error fetching drift metrics: {response.status_code}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def plot_metrics_comparison(filtered_runs_df, selected_models, metrics_selected):
    fig, ax = plt.subplots(figsize=(12, 6))
    for metric in metrics_selected:
        for model in selected_models:
            model_runs = filtered_runs_df[filtered_runs_df['Model Name'] == model]
            sns.lineplot(data=model_runs, x='Run ID', y=metric, ax=ax, marker='o', label=f"{model} - {metric}")
    plt.xticks(rotation=45)
    plt.title(f"Metrics Comparison for Selected Models", color=THEME_COLOR)
    plt.legend()
    st.pyplot(fig)

def plot_drift_over_time(drift_df, variables_selected):
    fig, ax = plt.subplots(figsize=(12, 6))
    for var in variables_selected:
        sns.lineplot(data=drift_df, x='timestamp', y=var, ax=ax, marker='o', label=var)
    plt.xticks(rotation=45)
    plt.title("Drift Metrics Over Time for Selected Variables", color=THEME_COLOR)
    plt.legend()
    st.pyplot(fig)

experiments = get_experiments(client)
experiment_names = [exp.name for exp in experiments]

st.sidebar.header("⚙️ Dashboard Controls")
experiment_selected = st.sidebar.selectbox('Select Experiment', experiment_names)
experiment_id = get_experiment_id(client, experiment_selected)

is_training_experiment = is_experiment_type(experiment_selected, "Training")
is_monitoring_experiment = is_experiment_type(experiment_selected, "Monitoring")

runs = get_runs(client, experiment_id)
runs_df = format_run_data(runs, is_training_experiment)

st.header(f"📊 Runs in Experiment: {experiment_selected}")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Run Summary")
    st.write(f"Total Runs: {len(runs_df)}")
    st.dataframe(runs_df.style.format(precision=2))

if is_training_experiment:
    unique_models = runs_df['Model Name'].unique()
    selected_models = st.sidebar.multiselect('Select Models for Comparison', unique_models)

    if selected_models:
        filtered_runs_df = runs_df[runs_df['Model Name'].isin(selected_models)]
        st.subheader(f"🧐 Selected Models: {', '.join(selected_models)}")
        st.dataframe(filtered_runs_df)
else:
    filtered_runs_df = runs_df

section_divider()
st.subheader("📈 Metrics Comparison")
with st.expander("View or Compare Metrics", expanded=True):
    metrics_columns = [col for col in runs_df.columns if col not in ['Run ID', 'Model Name', 'Status', 'Start Time', 'End Time', 'Duration (s)']]
    metrics_selected = st.multiselect('Select Metrics to Compare', metrics_columns)

    if metrics_selected:
        if is_training_experiment and selected_models:
            plot_metrics_comparison(filtered_runs_df, selected_models, metrics_selected)
        elif not is_training_experiment:
            plot_metrics_comparison(filtered_runs_df, ["Best Model"], metrics_selected)

if is_monitoring_experiment:
    section_divider()
    st.subheader("🌡️ Data Drift Summary")
    drift_df = fetch_drift_data()

    if drift_df is not None and not drift_df.empty:
        st.write("Heatmap of KL Divergence across variables:")
        kl_columns = [col for col in drift_df.columns if "KL_Divergence" in col]
        kl_means = drift_df[kl_columns].mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap([kl_means], annot=True, cmap="YlOrRd", ax=ax)
        ax.set_xticklabels(kl_columns, rotation=45)
        plt.title("Mean KL Divergence across Variables", color=THEME_COLOR)
        st.pyplot(fig)

        section_divider()
        st.subheader("📈 Drift Evolution Over Time")
        variables_selected = st.multiselect('Select Variables to Compare Drift Over Time', kl_columns)
        if variables_selected:
            plot_drift_over_time(drift_df, variables_selected)
    else:
        st.warning("No drift data available for visualization.")

section_divider()
st.subheader("📝 Detailed Run Information")
with st.expander("Expand to view detailed run data", expanded=False):
    for run in runs:
        if (is_training_experiment and run.info.run_name in selected_models) or not is_training_experiment:
            st.markdown(f"#### Run ID: {run.info.run_id}")
            st.write(f"**Status**: {run.info.status}")
            st.write(f"**Start Time**: {run.info.start_time}")
            st.write(f"**End Time**: {run.info.end_time}")
            st.write(f"**Duration (s)**: {(run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else 'Ongoing'}")
            st.write("### Metrics:")
            for metric_name, metric_value in run.data.metrics.items():
                st.write(f"- {metric_name}: {metric_value}")
            st.write("### Parameters:")
            for param_name, param_value in run.data.params.items():
                st.write(f"- {param_name}: {param_value}")
            st.write("### Artifacts:")
            artifacts = client.list_artifacts(run.info.run_id)
            if artifacts:
                for artifact in artifacts:
                    st.write(f"- {artifact.path}")

section_divider()
st.subheader("💾 Download Run Data")
with st.expander("Download CSV of Run Data", expanded=False):
    st.download_button(label="Download Run Data as CSV", data=runs_df.to_csv(index=False), mime="text/csv")
