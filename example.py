import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def create_synthetic_data_with_categorical(n_loans=100, n_entries_per_loan=10, n_features=20):
    np.random.seed(42)
    
    # Total number of samples
    n_samples = n_loans * n_entries_per_loan
    
    # Generate loan numbers and dates
    loan_numbers = np.repeat(np.arange(1, n_loans + 1), n_entries_per_loan)
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Generate feature data
    X = np.random.randn(n_samples, n_features)
    
    # Generate categorical features
    categorical_features = {
        'cat_feature_1': np.random.choice(['A', 'B', 'C'], size=n_samples),
        'cat_feature_2': np.random.choice(['X', 'Y', 'Z'], size=n_samples)
    }
    
    # Generate target variable
    y = np.random.randint(0, 2, size=n_samples)
    
    # Create DataFrame
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    X_df['loan_number'] = loan_numbers
    X_df['date'] = dates
    X_df['target'] = y
    
    # Add categorical features
    for cat_name, cat_data in categorical_features.items():
        X_df[cat_name] = cat_data
    
    return X_df

def split_data_by_loan(X_df, test_size=0.2, val_size=0.25, random_state=42):
    # Extract unique loan numbers
    unique_loans = X_df['loan_number'].unique()
    
    # Split unique loan numbers into train and temp (which will be further split into val and test)
    train_loans, temp_loans = train_test_split(unique_loans, test_size=test_size, random_state=random_state)
    
    # Split temp loans into validation and test sets
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size relative to temp
    val_loans, test_loans = train_test_split(temp_loans, test_size=val_size_adjusted, random_state=random_state)
    
    # Filter data based on these splits
    X_train = X_df[X_df['loan_number'].isin(train_loans)]
    X_val = X_df[X_df['loan_number'].isin(val_loans)]
    X_test = X_df[X_df['loan_number'].isin(test_loans)]
    
    return X_train, X_val, X_test


def process_features(train_df, val_df, test_df, categorical_features, numerical_features):
    """
    Process categorical and numerical features for train, validation, and test datasets.

    Parameters:
    - train_df (pd.DataFrame): Training data.
    - val_df (pd.DataFrame): Validation data.
    - test_df (pd.DataFrame): Test data.
    - categorical_features (list): List of column names for categorical features.
    - numerical_features (list): List of column names for numerical features.

    Returns:
    - pd.DataFrame: Processed training data.
    - pd.DataFrame: Processed validation data.
    - pd.DataFrame: Processed test data.
    """

    # Create transformers for numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # Create a column transformer to apply transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    # Fit and transform the training data
    train_processed = preprocessor.fit_transform(train_df)
    val_processed = preprocessor.transform(val_df)
    test_processed = preprocessor.transform(test_df)

    # Convert the transformed data back to DataFrame
    # Get feature names for the transformed data
    num_features = numerical_features
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_features = num_features + list(cat_features)
    
    train_df_processed = pd.DataFrame(train_processed, columns=all_features)
    val_df_processed = pd.DataFrame(val_processed, columns=all_features)
    test_df_processed = pd.DataFrame(test_processed, columns=all_features)

    return train_df_processed, val_df_processed, test_df_processed






