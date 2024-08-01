import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from itertools import product

def split_data(df, target_column='target'):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def feature_selection_with_xgb(X_train, y_train, xgb_params):
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **xgb_params)
    xgb_model.fit(X_train, y_train)
    feature_importances = xgb_model.feature_importances_
    top_features = np.argsort(feature_importances)[-30:]  # Select top 30 features
    return xgb_model, top_features

def train_logistic_regression(X_train, y_train, top_features):
    X_train_reduced = X_train.iloc[:, top_features]
    lr_model = LogisticRegression(penalty='l1', solver='saga', max_iter=1000)
    lr_model.fit(X_train_reduced, y_train)
    return lr_model

def compute_cumulative_probability(pred_probs):
    # Compute cumulative probability using the given formula
    cum_prob = 1 - np.prod(1 - np.array(pred_probs), axis=0)
    return cum_prob

def evaluate_model(xgb_params, X_train, y_train, X_val, y_val, quarter_columns):
    xgb_model, top_features = feature_selection_with_xgb(X_train, y_train, xgb_params)
    lr_model = train_logistic_regression(X_train, y_train, top_features)
    
    # Reduce validation data to top features
    X_val_reduced = X_val.iloc[:, top_features]
    
    # Predict probabilities for each quarter
    quarter_probs = []
    for quarter_col in quarter_columns:
        quarter_probs.append(lr_model.predict_proba(X_val_reduced[quarter_col])[:, 1])
    
    # Compute cumulative probability
    cum_prob = compute_cumulative_probability(quarter_probs)
    
    # Evaluate AUC using cumulative probabilities
    auc_score = roc_auc_score(y_val, cum_prob)
    return auc_score

def linear_search_tuning(X_train, y_train, X_val, y_val, quarter_columns, param_values):
    best_auc = 0
    best_params = None
    
    # Generate all combinations of parameters
    param_names = ['learning_rate', 'max_depth', 'subsample', 'colsample_bytree']
    all_combinations = product(*param_values)
    
    for combination in all_combinations:
        xgb_params = dict(zip(param_names, combination))
        xgb_params['n_estimators'] = 100  # Fixed value or include it in the search if needed
        auc_score = evaluate_model(xgb_params, X_train, y_train, X_val, y_val, quarter_columns)
        if auc_score > best_auc:
            best_auc = auc_score
            best_params = xgb_params
    
    return best_params, best_auc

def main(df, target_column='target', quarter_columns=None, param_values=None):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_column)
    
    # Define quarter columns if not provided
    if quarter_columns is None:
        quarter_columns = [col for col in X_train.columns if 'quarter' in col]
    
    # Define parameter values for linear search
    if param_values is None:
        param_values = [
            [0.1, 0.05, 0.01],    # learning_rate
            [3, 6, 9],            # max_depth
            [0.8, 0.9, 1.0],      # subsample
            [0.8, 0.9, 1.0]       # colsample_bytree
        ]
    
    best_params, best_auc = linear_search_tuning(X_train, y_train, X_val, y_val, quarter_columns, param_values)
    print(f'Best AUC: {best_auc:.4f}')
    print(f'Best Parameters: {best_params}')
    
    # Train final model with best parameters
    xgb_best = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        **best_params
    )
    xgb_best.fit(X_train, y_train)
    top_features = feature_selection_with_xgb(X_train, y_train, best_params)[1]
    lr_model = train_logistic_regression(X_train, y_train, top_features)
    
    # Evaluate on the test set
    X_test_reduced = X_test.iloc[:, top_features]
    y_test_prob = lr_model.predict_proba(X_test_reduced)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_prob)
    print(f'Test AUC: {test_auc:.4f}')

# Usage
# df = pd.read_csv('your_data.csv')  # Load your data
# main(df)
