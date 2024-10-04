# **MLOps Project - End-to-End Machine Learning Pipeline with DVC and Remote Storage**

## **Overview**

This project demonstrates a complete **MLOps pipeline** using **DVC** (Data Version Control) for tracking data and model artifacts, and **MLflow** for managing experiments and model lifecycle. It includes stages from data preparation, model training, and evaluation to model deployment, with environment-specific configurations for **development**, **staging**, and **production** environments. A Flask API is also provided to serve predictions, with everything containerized using Docker.

## **Project Structure**

```bash
mlops_project/
├── .dvc/                   
├── data/                   
│   ├── raw/                
│   └── processed/          
├── models/                 
├── src/
│   ├── app/                
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

### **1. Data Version Control (DVC)**
DVC is used to manage data and model artifacts. The pipeline stages are defined in `dvc.yaml`, allowing easy reproduction of the entire pipeline and pushing the artifacts to remote storage.

### **2. Data Preparation**
The `data_preparation.py` script loads, preprocesses, and splits the data. It outputs the train/test datasets, which are tracked by DVC in the `data/processed/` directory.

### **3. Model Training**
The `train_model.py` script trains a machine learning model using RandomForestRegressor, logs the model to **MLflow**, and registers it in the **MLflow Model Registry**. This is part of the DVC pipeline.

### **4. Model Evaluation**
The `evaluate_model.py` script evaluates the model on test data, logs evaluation metrics (e.g., MSE, R² score) to MLflow, and is tracked by DVC.

### **5. Flask API**
A Flask-based REST API (`app.py`) serves the model for predictions. The API loads the latest production model from the **MLflow Model Registry** and returns predictions as JSON.

### **6. Docker**
The project is containerized using **Docker Compose** to orchestrate the following services:
- **Flask App**: A REST API for predictions.
- **PostgreSQL**: Database to store MLflow tracking information.
- **MinIO**: Object storage to store model artifacts.
- **MLflow Server**: To track experiments and manage the model registry.

## **Setup and Running**

### **1. Prerequisites**
- Docker & Docker Compose installed.
- Python 3.8+ and `pip` installed.
- DVC installed (`pip install dvc`).
- **DVC remote** storage (e.g., S3, GDrive) is already set up.

### **2. Clone the Repository**

```bash
git clone https://github.com/ArmandoDLaRosa/mlops_project.git
cd mlops_project
```

### **3. Set Up Python Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```

### **4. Configure Environment Variables**
Create your own `.env` files for each environment. Example `.env.dev`:

```bash
# .env.dev
MLFLOW_TRACKING_URI=http://localhost:5001
MLFLOW_ARTIFACT_URI=http://localhost:9000/mlflow_dev
MODEL_REGISTRY_STAGE="Development"
RAW_DATA_PATH=data/raw/Occupancy_Estimation.csv
PROCESSED_DATA_PATH=data/processed
```

Repeat for `.env.staging` and `.env.prod`.

### **5. Running the Project with Docker Compose**

#### **Development:**
```bash
docker-compose --env-file .env.dev up -d
```

#### **Staging:**
```bash
docker-compose --env-file .env.staging up -d
```

#### **Production:**
```bash
docker-compose --env-file .env.prod up -d
```

### **6. Using DVC to Manage the Pipeline**
The DVC pipeline is defined in `dvc.yaml`. This file outlines the steps for data preparation, model training, and evaluation.

#### To run the full DVC pipeline:

```bash
dvc repro
```

This will execute the following stages:
1. **Data Preparation**: Loads, preprocesses, and splits the data.
2. **Model Training**: Trains the model and registers it with MLflow.
3. **Model Evaluation**: Evaluates the model and logs metrics.

#### Pushing Artifacts to Remote Storage:

Once the pipeline has completed successfully, push all data and artifacts to the remote storage using:

```bash
dvc push
```

This ensures that the raw data, processed data, trained models, and any other tracked files are stored in the configured DVC remote (Gdrive).

### **7. Training the Model**
The model training process can be run manually or as part of the DVC pipeline. If you just want to retrain the model:

```bash
dvc repro train
```

This will:
- Use the processed training data.
- Train the model and log it to MLflow.
- Register the trained model in the MLflow Model Registry.

### **8. Testing the API**
Once the model is trained and the Flask API is running, you can test it by sending a **POST** request to the `/predict` endpoint.

Example using `curl`:

```bash
curl -X POST -H "Content-Type: application/json" -d '[
    {"feature1": 2.3, "feature2": 3.5, "feature3": 4.7}
]' http://localhost:5000/predict
```

The response will include the model’s predictions:

```json
{
    "predictions": [23.5, 42.1, ...]
}
```

### **9. Managing DVC and MLflow**
- **DVC** is used to track dataset versions, models, and pipelines. You can push/pull your data and models to the remote storage easily.
  
  ```bash
  dvc add data/raw/Occupancy_Estimation.csv
  git add data/raw/Occupancy_Estimation.csv.dvc
  git commit -m "Added raw data to DVC"
  ```

  After a pipeline run, use `dvc push` to push changes to remote storage.

- **MLflow** is used to track experiments, models, and metrics. You can view your experiment results and model registry by accessing the MLflow UI at `http://localhost:5001`.

### **10. Shutting Down Services**
To stop the running services:

```bash
docker-compose down
```