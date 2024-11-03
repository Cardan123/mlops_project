import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from train_model import ModelTrainer
from test_data_preparation import DataPreparation

# Fixture for setting up a DataPreparation instance with test parameters.
# This ensures consistent test data without requiring actual files.
@pytest.fixture
def data_preparation_instance(tmp_path):
    params = {'prepare_data': {'test_size': 0.2, 'random_state': 42}}
    return DataPreparation(input_path="mock_input.csv", output_dir=tmp_path, params=params)

# Test for the load_data method in ModelTrainer.
# This test checks that load_data calls pandas' read_csv function for loading data,
# allowing us to verify that the method interacts correctly with file reading.
@patch('train_model.mlflow')
def test_load_data(mock_mlflow, tmp_path):
    trainer = ModelTrainer(X_train_path="../data/processed/X_train.csv", y_train_path="../data/processed/y_train.csv", params_path="../params.yaml")
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = MagicMock()
        X_train, y_train = trainer.load_data()
        mock_read_csv.assert_called() 

# Test for the train_and_evaluate method in ModelTrainer.
# Checks that train_and_evaluate successfully selects and evaluates models using GridSearchCV,
# then returns the best model with expected name and parameters.
@patch('train_model.GridSearchCV')
@patch.object(DataPreparation, 'load_data')
def test_train_and_evaluate(mock_load_data, mock_grid_search, data_preparation_instance):
    mock_load_data.return_value = (
        pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}),
        pd.DataFrame({'target': [0, 1, 0]})
    )

    mock_gs_instance = mock_grid_search.return_value
    mock_gs_instance.best_score_ = -0.5
    mock_gs_instance.best_estimator_ = MagicMock()
    mock_gs_instance.best_params_ = {"param1": 1, "param2": 2}
    mock_gs_instance.best_estimator_.predict.return_value = [0] * 8103 

    trainer = ModelTrainer(
        X_train_path="../data/processed/X_train.csv",
        y_train_path="../data/processed/y_train.csv",
        params_path="../params.yaml"
    )

    X_train, y_train = trainer.load_data() 
    models_params = {'LinearRegression': {'fit_intercept': [True, False]}}

    best_model, best_model_name, best_params, best_run_id = trainer.train_and_evaluate(
        X_train, y_train, models_params
    )

    assert best_model is not None
    assert best_model_name == "LinearRegression"
    assert best_params == {"param1": 1, "param2": 2}
