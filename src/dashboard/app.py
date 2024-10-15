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
import json

THEME_COLOR = "#FFA07A"
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:5000')

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

st.set_page_config(layout="wide", page_title="MLflow & API Interface", page_icon="🚀")

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #FFA07A;'>MLflow Experiment Dashboard & API Interface</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Track MLflow experiments and interact with the deployed API for predictions and drift analysis</h4>", unsafe_allow_html=True)
st.markdown("---")

def get_experiments(client):
    return client.search_experiments()

def get_experiment_id(client, experiment_name):
    return client.get_experiment_by_name(experiment_name).experiment_id

def get_runs(client, experiment_id):
    return client.search_runs(experiment_ids=[experiment_id])

def is_experiment_type(experiment_name, type_name):
    return type_name in experiment_name

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

def fetch_recent_predictions():
    try:
        response = requests.get(f'{API_BASE_URL}/recent_predictions')
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching recent predictions: {response.status_code}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    return []

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

def get_predictions(input_data=None):
    try:
        if input_data:
            response = requests.post(f'{API_BASE_URL}/predict', json=input_data)
        else:
            response = requests.post(f'{API_BASE_URL}/predict', json={})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching predictions: {response.status_code}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

def plot_confidence_intervals(predictions_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(data=predictions_df, x='timestamp', y='predictions', label='Predictions', ax=ax)
    
    ax.fill_between(
        predictions_df['timestamp'],
        predictions_df['confidence_interval_lower'],
        predictions_df['confidence_interval_upper'],
        color='b', alpha=0.2, label='Confidence Interval'
    )
    
    plt.xticks(rotation=45)
    plt.title("Predictions with Confidence Intervals", color=THEME_COLOR)
    plt.legend()
    st.pyplot(fig)

def confidence_level(row):
    width = row['confidence_interval_width']
    if width < 0.1:
        return ['background-color: #7CFC00'] * len(row) 
    elif width < 0.5:
        return ['background-color: '] * len(row)
    else:
        return ['background-color: #FF4500'] * len(row)
        
tabs = st.sidebar.radio("Select Tab", ['MLflow Dashboard', 'API Interface'])

if tabs == 'MLflow Dashboard':
    st.header("📊 MLflow Experiment Dashboard")
    st.markdown("<h5 style='color: #4CAF50;'>Track your experiment progress, compare runs, and visualize drift metrics</h5>", unsafe_allow_html=True)
    
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

            st.subheader("📈 Drift Evolution Over Time")
            variables_selected = st.multiselect('Select Variables to Compare Drift Over Time', kl_columns)
            if variables_selected:
                plot_drift_over_time(drift_df, variables_selected)
        else:
            st.warning("No drift data available for visualization.")

    recent_predictions = fetch_recent_predictions()

    if is_monitoring_experiment:
        st.subheader("🔮 Recent Predictions")
        if not recent_predictions:
            st.warning("No recent predictions available")

        predictions_df = pd.DataFrame(recent_predictions)

        predictions_df['predictions'] = predictions_df['predictions'].apply(lambda x: x[0] if isinstance(x, list) else x)

        predictions_df['confidence_interval_lower'] = predictions_df['confidence_interval'].apply(
            lambda x: x['lower'][0] if isinstance(x['lower'], list) else x['lower'])
        predictions_df['confidence_interval_upper'] = predictions_df['confidence_interval'].apply(
            lambda x: x['upper'][0] if isinstance(x['upper'], list) else x['upper'])

        predictions_df['confidence_interval_lower'] = pd.to_numeric(predictions_df['confidence_interval_lower'], errors='coerce')
        predictions_df['confidence_interval_upper'] = pd.to_numeric(predictions_df['confidence_interval_upper'], errors='coerce')

        if not pd.api.types.is_datetime64_any_dtype(predictions_df['timestamp']):
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'], errors='coerce')

        predictions_df = predictions_df.dropna(subset=['confidence_interval_lower', 'confidence_interval_upper', 'predictions', 'timestamp'])

        st.dataframe(predictions_df)

        plot_confidence_intervals(predictions_df)

    st.subheader("📝 Detailed Run Information")
    with st.expander("Expand to view detailed run data", expanded=False):
        for run in runs:
            if (is_training_experiment and run.info.run_name in selected_models) or not is_training_experiment:
                st.markdown(f"#### Run ID: {run.info.run_id}")
                st.write(f"**Status**: {run.info.status}")
                st.write(f"**Start Time**: {pd.to_datetime(run.info.start_time, unit='ms')}")
                st.write(f"**End Time**: {pd.to_datetime(run.info.end_time, unit='ms') if run.info.end_time else 'Ongoing'}")
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

    st.subheader("💾 Download Run Data")
    with st.expander("Download CSV of Run Data", expanded=False):
        st.download_button(label="Download Run Data as CSV", data=runs_df.to_csv(index=False), mime="text/csv")

elif tabs == 'API Interface':
    st.header("🔌 Flask API Interface")
    st.markdown("<h5 style='color: #4CAF50;'>Interact with your deployed API for model predictions and data monitoring</h5>", unsafe_allow_html=True)
    
    with st.expander("📊 Model Predictions", expanded=True):
        input_data = st.text_area("Input Data (Optional, leave blank for random data)", height=100)
        st.markdown('<small>Provide input in JSON format or leave blank to generate random data</small>', unsafe_allow_html=True)
        
        if input_data:
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                st.error("Invalid JSON format. Please provide a valid JSON.")
                input_data = None
        
        if st.button("Get Prediction"):
            prediction = get_predictions(input_data)
            if prediction:
                st.write("Prediction Result:", prediction)



    with st.expander("🔮 Recent Predictions", expanded=True):
        if st.button("Fetch Recent Predictions"):
            recent_predictions = fetch_recent_predictions()
            
            if recent_predictions:
                predictions_df = pd.DataFrame(recent_predictions)

                predictions_df['confidence_interval_lower'] = predictions_df['confidence_interval'].apply(lambda x: x['lower'][0])
                predictions_df['confidence_interval_upper'] = predictions_df['confidence_interval'].apply(lambda x: x['upper'][0])

                predictions_df['confidence_interval_width'] = predictions_df['confidence_interval_upper'] - predictions_df['confidence_interval_lower']

                styled_df = predictions_df.style.apply(confidence_level, axis=1)
                
                st.dataframe(styled_df)

    with st.expander("🌡️ Drift Metrics", expanded=True):
        if st.button("Fetch Drift Metrics"):
            drift_history = fetch_drift_data()
            if drift_history is not None:
                st.dataframe(drift_history)
