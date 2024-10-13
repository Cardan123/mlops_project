# **MLOps Project: End-to-End Machine Learning Pipeline with DVC and Remote Storage**

## **Overview**

This project is a full **MLOps pipeline** that uses **DVC** for tracking data and models, and **MLflow** for managing experiments and model lifecycle. It covers everything from data preparation, model training, and evaluation to deployment. It's built with environment-specific configurations for **development**, **staging**, and **production** environments. The project also includes a Flask API for serving model predictions and uses **Streamlit** to visualize experiment results and monitor the model.

## **Project Structure**

```bash
mlops_project/
├── .dvc/                   
├── data/                    
│   ├── raw/                
│   └── processed/          
├── models/                 
├── src/
│   ├── dashboard/           
│   │   ├── app.py          
│   │   ├── Dockerfile      
│   │   └── requirements.txt
│   ├── app/                 
│   │   ├── app.py          
│   │   ├── Dockerfile      
│   │   └── requirements.txt 
│   └── scripts/            
├── tests/                 
├── notebooks/             
├── .env.dev                
├── .env.staging            
├── .env.prod            
├── dvc.yaml                
├── docker-compose.yml       
└── requirements.txt         
```

## **Components**

### **1. Flask API**

A RESTful Flask API (`src/app/app.py`) serves the production-ready model for predictions. It interacts with the **MLflow Model Registry** to load and serve models. Core features include:

- **Prediction Endpoint**: Takes input data and returns predictions.
- **Model Reloading**: Automatically reloads the latest production model.
- **Monitoring**: Displays status of the monitoring service.

### **2. Streamlit Dashboard**

The **Streamlit app** (`src/dashboard/app.py`) lets you visualize experiment results and metrics from **MLflow**, monitor drift metrics, and track model performance. It connects to the Flask API for live data. Key features include:

- Comparing metrics between models.
- Visualizing experiment runs and model performance.
- Tracking data drift over time.

### **3. Data Version Control (DVC)**

**DVC** tracks the dataset and model artifacts. The pipeline defined in `dvc.yaml` ensures reproducibility and can push artifacts to a **remote storage** like S3 or Google Drive.

### **4. Docker Compose**

The project is fully containerized using Docker Compose to orchestrate:

- **Flask API**: Serves predictions from the latest model.
- **Streamlit Dashboard**: For MLflow experiment visualization.
- **MLflow Server**: Tracks experiments and manages the model registry.
- **PostgreSQL**: Stores MLflow tracking data.
- **MinIO**: Object storage for model artifacts.

## **Setup and Running**

### **1. Prerequisites**

- Docker & Docker Compose
- Python 3.8+ and `pip`
- DVC (`pip install dvc`)
- **Remote storage** set up for DVC (e.g., S3 or GDrive)

### **2. Clone the Repository**

```bash
git clone https://github.com/ArmandoDLaRosa/mlops_project.git
cd mlops_project
```

### **3. Virtual Environment Setup**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **4. Configure Environment Variables**

You'll need `.env` files for each environment. Example for **development** (`.env.dev`):

```bash
MLFLOW_TRACKING_URI=http://tracking_server:5001
MLFLOW_ARTIFACT_URI=http://localhost:9000/mlflowdev
MODEL_REGISTRY_STAGE="Development"
RAW_DATA_PATH=data/raw/Occupancy_Estimation.csv
PROCESSED_DATA_PATH=data/processed
```

Repeat for `.env.staging` and `.env.prod`.

### **5. Run the Project with Docker Compose**

#### **Development**:

```bash
docker-compose --env-file .env.dev up -d
```

#### **Staging**:

```bash
docker-compose --env-file .env.staging up -d
```

#### **Production**:

```bash
docker-compose --env-file .env.prod up -d
```

Streamlit will be available at `http://localhost:8501` and the Flask API at `http://localhost:5000`.

### **6. Test the API**

You can send a **POST** request to the `/predict` endpoint to test the API:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
    "S1_Temp": 25.3, "S2_Temp": 26.1, "S3_Temp": 24.8, "S4_Temp": 25.0,
    "S1_Light": 100, "S2_Light": 120, "S3_Light": 110, "S4_Light": 90,
    "S1_Sound": 0.5, "S2_Sound": 0.7, "S3_Sound": 0.3, "S4_Sound": 0.4,
    "S5_CO2": 380, "S5_CO2_Slope": 0.03, "S6_PIR": 1, "S7_PIR": 1
}' http://localhost:5000/predict
```

You'll receive a JSON response with the model’s predictions:

```json
{
    "input_data": {...},
    "predictions": [23.5]
}
```

### **7. DVC Pipeline Management**

The DVC pipeline is defined in `dvc.yaml`, which outlines the data preparation, model training, and evaluation steps.

To run the full DVC pipeline:

```bash
dvc repro
```

This will execute the following stages:

1. **Data Preparation**: Load, preprocess, and split the data.
2. **Model Training**: Train the model and log it with MLflow.
3. **Model Evaluation**: Evaluate the model and track metrics.

To push the artifacts to the remote storage:

```bash
dvc push
```

This stores data, models, and other artifacts in your remote DVC storage.

### **8. Shutting Down Services**

To stop all services:

```bash
docker-compose down
```

### **9. Data Drift Monitoring**

The **Streamlit Dashboard** allows you to track data drift over time. You can manually trigger drift analysis through the API:

```bash
curl -X POST http://localhost:5000/trigger_drift_analysis
```

### **10. MLflow Experiment Visualization**

The dashboard also provides an interface for comparing **MLflow experiment runs**, metrics, and models, helping you monitor your production model’s performance.
