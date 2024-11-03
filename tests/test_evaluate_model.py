from unittest.mock import patch, MagicMock
import pytest
from evaluate_model import ModelEvaluator

# Test for the load_data method.
# Verifies that the load_data method calls pandas' read_csv function
# and returns the expected X_test and y_test datasets without actual file I/O.
@patch('evaluate_model.mlflow')
def test_load_data(mock_mlflow):
    evaluator = ModelEvaluator(X_test_path="../data/processed/X_test.csv", y_test_path="../data/processed/y_test.csv", model_path="../models/best_model.pkl")
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = MagicMock()
        X_test, y_test = evaluator.load_data()
        mock_read_csv.assert_called()

# Test for the load_model method.
# Checks that load_model correctly calls joblib's load function to load a model
# without actually reading a model file, confirming the method’s functionality.
@patch('evaluate_model.joblib.load')
def test_load_model(mock_load):
    evaluator = ModelEvaluator(X_test_path="../data/processed/X_test.csv", y_test_path="../data/processed/y_test.csv", model_path="../models/best_model.pkl")
    mock_load.return_value = MagicMock()
    model = evaluator.load_model()
    mock_load.assert_called() 

# Test for the log_metrics method.
# Ensures that log_metrics properly logs performance metrics (MSE, R^2, MAE, RMSE)
# to MLflow, confirming that metrics are accurately logged.
@patch('evaluate_model.mlflow')
def test_log_metrics(mock_mlflow):
    evaluator = ModelEvaluator(X_test_path="../data/processed/X_test.csv", y_test_path="../data/processed/y_test.csv", model_path="../models/best_model.pkl")
    evaluator.log_metrics(mse=0.25, r2=0.9, mae=0.1, rmse=0.5)
    mock_mlflow.log_metric.assert_any_call("test_mse", 0.25)
    mock_mlflow.log_metric.assert_any_call("test_r2", 0.9)
    mock_mlflow.log_metric.assert_any_call("test_mae", 0.1)
    mock_mlflow.log_metric.assert_any_call("test_rmse", 0.5)
