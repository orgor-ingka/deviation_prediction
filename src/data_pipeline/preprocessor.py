import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

def preprocess_data(raw_data_file_path: str = 'raw/raw_data.csv',
                    processed_data_file_path: str = 'processed/processed_data.csv') -> pd.DataFrame:
    """
    Reads raw data from the specified path, performs standard preprocessing steps
    on the DataFrame for machine learning, and saves the processed data.

    Steps include:
    1. Null value checking and imputation.
    2. Type processing (e.g., converting categorical to numerical).
    3. Normalization of numerical features.

    Args:
        raw_data_file_path (str): The path to the raw data CSV file, relative to the project's data directory.
                                  Defaults to 'raw/raw_data.csv'.
        processed_data_file_path (str): The path to save the processed data CSV file, relative to the project's data directory.
                                      Defaults to 'processed/processed_data.csv'.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    print("Starting data preprocessing...")

    # Construct the full path relative to the project root (assuming this script is in src/data_pipeline/)
    base_data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

    full_raw_data_path = os.path.join(base_data_path, raw_data_file_path)
    
    if not os.path.exists(full_raw_data_path):
        print(f"Error: Raw data file not found at {full_raw_data_path}")
        return pd.DataFrame()

    df = pd.read_csv(full_raw_data_path)
    print(f"Successfully loaded {len(df)} rows from {full_raw_data_path}.")

    # Make a copy to avoid modifying the original DataFrame
    processed_df = df.copy()

    # 1. Null Value Checking and Imputation
    numerical_cols = processed_df.select_dtypes(include=['number']).columns
    categorical_cols = processed_df.select_dtypes(include=['object', 'category']).columns

    if not numerical_cols.empty:
        print(f"Imputing missing numerical values in columns: {list(numerical_cols)}")
        for col in numerical_cols:
            if processed_df[col].isnull().any():
                median_val = processed_df[col].median()
                processed_df[col].fillna(median_val, inplace=True)

    if not categorical_cols.empty:
        print(f"Imputing missing categorical values in columns: {list(categorical_cols)}")
        for col in categorical_cols:
            if processed_df[col].isnull().any():
                mode_val = processed_df[col].mode()[0]
                processed_df[col].fillna(mode_val, inplace=True)

    print("Null values handled.")

    # 2. Type Processing (e.g., One-Hot Encoding for categorical features)
    if not categorical_cols.empty:
        print(f"Applying one-hot encoding to categorical columns: {list(categorical_cols)}")
        processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
    print("Type processing completed.")

    # 3. Normalization of numerical features
    numerical_cols_after_encoding = processed_df.select_dtypes(include=['number']).columns

    if not numerical_cols_after_encoding.empty:
        print(f"Normalizing numerical features: {list(numerical_cols_after_encoding)}")
        scaler = StandardScaler()
        processed_df[numerical_cols_after_encoding] = scaler.fit_transform(processed_df[numerical_cols_after_encoding])
    print("Normalization completed.")

    print("Data preprocessing finished.")

    # Save processed data
    full_processed_data_path = os.path.join(base_data_path, processed_data_file_path)
    output_dir = os.path.dirname(full_processed_data_path)
    os.makedirs(output_dir, exist_ok=True)
    processed_df.to_csv(full_processed_data_path, index=False)
    print(f"Processed data saved to {full_processed_data_path}.")

    return processed_df