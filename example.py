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


def process_features(train_df, val_df, test_df):
    """
    Automatically process categorical and numerical features for train, validation, and test datasets.

    Parameters:
    - train_df (pd.DataFrame): Training data.
    - val_df (pd.DataFrame): Validation data.
    - test_df (pd.DataFrame): Test data.

    Returns:
    - pd.DataFrame: Processed training data.
    - pd.DataFrame: Processed validation data.
    - pd.DataFrame: Processed test data.
    """
    
    # Identify numerical and categorical features
    numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

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

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

# Step 1: Data Preparation
def load_and_split_data(file_path, target_column, test_size=0.3, val_size=0.5, random_state=42):
    data = pd.read_csv(file_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Step 2: Feature Selection with XGBoost
def select_features_with_xgboost(X_train, y_train, params):
    xgb = XGBClassifier(**params)
    xgb.fit(X_train, y_train)
    importances = xgb.feature_importances_
    features = X_train.columns
    important_features = pd.Series(importances, index=features).nlargest(30).index
    return important_features

# Step 3: Logistic Regression with L1 Regularization
def train_logistic_regression_models(X_train, y_train, important_features, K, penalty='l1', solver='liblinear', max_iter=1000):
    quarterly_models = {}
    for quarter in range(1, K + 1):
        lr = LogisticRegression(penalty=penalty, solver=solver, max_iter=max_iter)
        X_train_quarter = X_train[important_features]
        lr.fit(X_train_quarter, y_train)
        quarterly_models[quarter] = lr
    return quarterly_models

# Step 4: Compute Cumulative Probability of Default
def compute_cumulative_probability(models, X, important_features):
    probabilities = []
    for quarter, model in models.items():
        prob = model.predict_proba(X[important_features])[:, 1]
        probabilities.append(prob)
    cum_prob = 1 - np.prod([(1 - p) for p in probabilities], axis=0)
    return cum_prob

# Step 5: Model Evaluation
def evaluate_model(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc

# Step 6: Linear Search for Hyperparameter Tuning
def linear_search_xgboost_lr(file_path, target_column, param_options, num_features=30, K=4):
    # Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(file_path, target_column)
    
    best_auc = 0
    best_params = None
    best_models = None
    best_features = None
    
    for params in param_options:
        # Feature selection with XGBoost
        important_features = select_features_with_xgboost(X_train, y_train, params)
        
        # Train Logistic Regression models
        quarterly_models = train_logistic_regression_models(X_train, y_train, important_features, K)
        
        # Compute cumulative probability on validation set
        y_val_pred = compute_cumulative_probability(quarterly_models, X_val, important_features)
        
        # Evaluate model
        auc = evaluate_model(y_val, y_val_pred)
        
        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_models = quarterly_models
            best_features = important_features
    
    print(f'Best AUC on Validation Set: {best_auc}')
    print(f'Best Parameters: {best_params}')
    
    return best_models, best_features, best_params, X_test, y_test

# Step 7: Final Evaluation on Test Set
def evaluate_on_test_set(X_test, y_test, best_models, best_features):
    y_test_pred = compute_cumulative_probability(best_models, X_test, best_features)
    auc = evaluate_model(y_test, y_test_pred)
    print(f'Test AUC: {auc}')
    return auc

# Main function to run all steps
def main(file_path, target_column, param_options, num_features=30, K=4):
    best_models, best_features, best_params, X_test, y_test = linear_search_xgboost_lr(file_path, target_column, param_options, num_features, K)
    evaluate_on_test_set(X_test, y_test, best_models, best_features)

# Example Usage
param_options = [
    {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.9, 'colsample_bytree': 0.9},
    {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.8, 'colsample_bytree': 0.8},
    {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.9, 'colsample_bytree': 0.9},
    # Add more parameter combinations here
]

main('loan_data.csv', 'default', param_options, num_features=30, K=4)


import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def process_features(train_df, val_df, test_df, numerical_features, categorical_features):
    # Initialize the scaler and encoder
    scaler = StandardScaler()
    encoder = OneHotEncoder(sparse=False, drop='first')
    
    # Fit on training data
    scaler.fit(train_df[numerical_features])
    encoder.fit(train_df[categorical_features])
    
    # Transform training data
    train_numerical = scaler.transform(train_df[numerical_features])
    train_categorical = encoder.transform(train_df[categorical_features])
    
    train_numerical = pd.DataFrame(train_numerical, columns=numerical_features, index=train_df.index)
    train_categorical = pd.DataFrame(train_categorical, columns=encoder.get_feature_names_out(categorical_features), index=train_df.index)
    
    train_processed = pd.concat([train_numerical, train_categorical], axis=1)
    
    # Transform validation data
    val_numerical = scaler.transform(val_df[numerical_features])
    val_categorical = encoder.transform(val_df[categorical_features])
    
    val_numerical = pd.DataFrame(val_numerical, columns=numerical_features, index=val_df.index)
    val_categorical = pd.DataFrame(val_categorical, columns=encoder.get_feature_names_out(categorical_features), index=val_df.index)
    
    val_processed = pd.concat([val_numerical, val_categorical], axis=1)
    
    # Transform test data
    test_numerical = scaler.transform(test_df[numerical_features])
    test_categorical = encoder.transform(test_df[categorical_features])
    
    test_numerical = pd.DataFrame(test_numerical, columns=numerical_features, index=test_df.index)
    test_categorical = pd.DataFrame(test_categorical, columns=encoder.get_feature_names_out(categorical_features), index=test_df.index)
    
    test_processed = pd.concat([test_numerical, test_categorical], axis=1)
    
    return train_processed, val_processed, test_processed

# Example usage
# train_df, val_df, and test_df are pandas DataFrames containing the training, validation, and test data respectively
# numerical_features is a list of names of numerical columns
# categorical_features is a list of names of categorical columns
# train_processed, val_processed, test_processed = process_features(train_df, val_df, test_df, numerical_features, categorical_features)


