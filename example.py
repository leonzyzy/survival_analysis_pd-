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

# Example usage
df = create_synthetic_data_with_categorical(n_loans=100, n_entries_per_loan=10, n_features=20)
X_train, X_val, X_test = split_data_by_loan(df, test_size=0.2, val_size=0.25)

X_val
