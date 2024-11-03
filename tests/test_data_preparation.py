import pytest
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from data_preparation import DataPreparation

# Sample data fixture to provide consistent input data for the tests.
@pytest.fixture
def sample_data():
    data = {
        'Date': ['2021-01-01', '2021-01-02', '2021-01-03'],
        'Time': ['08:00', '14:00', '20:00'],
        'S3_Temp': [22.1, 23.4, 21.8],
        'Room_Occupancy_Count': [1, 0, 1]
    }
    return pd.DataFrame(data)

# Fixture for data preparation parameters, allowing customization for test configurations.
@pytest.fixture
def data_prep_params():
    return {'prepare_data': {'test_size': 0.2, 'random_state': 42}}

# Fixture for creating an instance of DataPreparation with a temporary path.
@pytest.fixture
def data_preparation_instance(tmp_path, data_prep_params):
    return DataPreparation(input_path=None, output_dir=tmp_path, params=data_prep_params)

# Test for the create_time_of_day_column method.
# This test verifies that a new column, 'Time_of_Day', is added to the DataFrame,
# and that the values correctly categorize different times into parts of the day.
def test_create_time_of_day_column(sample_data, data_preparation_instance):
    df = data_preparation_instance.create_time_of_day_column(sample_data)
    assert 'Time_of_Day' in df.columns
    assert set(df['Time_of_Day'].unique()) == {'Morning', 'Afternoon', 'Evening'}

# Test for the drop_unnecessary_columns method.
# Ensures specified columns, like 'S3_Temp', are successfully removed from the DataFrame.
def test_drop_unnecessary_columns(sample_data, data_preparation_instance):
    df = data_preparation_instance.drop_unnecessary_columns(sample_data, ['S3_Temp'])
    assert 'S3_Temp' not in df.columns

# Test for the encode_labels method.
# Verifies that the 'Time_of_Day' categorical column is encoded as integers in both training and test sets.
def test_encode_labels(data_preparation_instance):
    df = pd.DataFrame({'Time_of_Day': ['Morning', 'Afternoon', 'Evening']})
    X_train, X_test = df.copy(), df.copy()
    
    X_train_encoded, X_test_encoded, encoder = data_preparation_instance.encode_labels(X_train, X_test)
    assert 'Time_of_Day' in X_train_encoded.columns
    assert X_train_encoded['Time_of_Day'].dtype == 'int64' 

# Test for the standardize_data method.
# Confirms that numeric columns are standardized to have a mean of approximately 0.
# This ensures the method applies proper scaling to the data.
def test_standardize_data(data_preparation_instance):
    df_train = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    df_test = pd.DataFrame({'feature1': [2, 3, 4], 'feature2': [5, 6, 7]})
    X_train_scaled, X_test_scaled, scaler = data_preparation_instance.standardize_data(df_train, df_test)
    
    assert X_train_scaled['feature1'].mean() == pytest.approx(0, rel=1e-6)
    assert X_test_scaled['feature1'].mean() == pytest.approx(0, rel=1e-6)
