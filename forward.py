from lifelines import CoxPHFitter

def forward_selection(data_df, duration_col, event_col, p_value_threshold=0.05):
    remaining_features = set(data_df.columns) - {duration_col, event_col}
    selected_features = []
    current_score = 0.0
    best_new_score = 0.0

    while remaining_features and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining_features:
            cox_model = CoxPHFitter()
            try:
                cox_model.fit(data_df[selected_features + [candidate] + [duration_col, event_col]], duration_col=duration_col, event_col=event_col)
                score = cox_model.score_
                scores_with_candidates.append((score, candidate))
            except:
                continue

        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining_features.remove(best_candidate)
            selected_features.append(best_candidate)
            current_score = best_new_score

    return selected_features

selected_features = forward_selection(data_df, 'duration', 'event')
print("Selected features:", selected_features)

from lifelines import CoxPHFitter

def backward_selection(data_df, duration_col, event_col, p_value_threshold=0.05):
    remaining_features = set(data_df.columns) - {duration_col, event_col}
    cox_model = CoxPHFitter()
    cox_model.fit(data_df[list(remaining_features) + [duration_col, event_col]], duration_col=duration_col, event_col=event_col)

    while True:
        p_values = cox_model.summary['p']
        worst_p_value = p_values.max()
        
        if worst_p_value < p_value_threshold:
            break

        worst_feature = p_values.idxmax()
        remaining_features.remove(worst_feature)
        
        cox_model.fit(data_df[list(remaining_features) + [duration_col, event_col]], duration_col=duration_col, event_col=event_col)

    return list(remaining_features)

# Fit the Cox model and perform backward selection
selected_features = backward_selection(data_df, 'duration', 'event')
print("Selected features:", selected_features)

# Fit the final model with selected features

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import numpy as np

def tune_penalty_cox(train_df, validation_df, duration_col, event_col, penalty_values):
    best_penalty = None
    best_c_index = -1

    for penalty in penalty_values:
        # Fit the Cox model with the current penalty
        cox_model = CoxPHFitter(penalizer=penalty)
        cox_model.fit(train_df, duration_col=duration_col, event_col=event_col)

        # Predict risk scores on the validation set
        validation_durations = validation_df[duration_col]
        validation_events = validation_df[event_col]
        validation_risk_scores = cox_model.predict_partial_hazard(validation_df)

        # Calculate the C-index on the validation set
        c_index = concordance_index(validation_durations, -validation_risk_scores, validation_events)

        print(f"Penalty: {penalty}, C-index: {c_index}")

        # Update the best penalty if current C-index is higher
        if c_index > best_c_index:
            best_c_index = c_index
            best_penalty = penalty

    print(f"Best penalty: {best_penalty}, Best C-index: {best_c_index}")
    return best_penalty

# Example usage
train_df = ...  # Your training dataframe with 'duration' and 'event' columns
validation_df = ...  # Your validation dataframe with 'duration' and 'event' columns
duration_col = 'duration'
event_col = 'event'
penalty_values = np.logspace(-4, 4, 50)  # Example range of penalty values

best_penalty = tune_penalty_cox(train_df, validation_df, duration_col, event_col, penalty_values)















final_cox_model = CoxPHFitter()
final_cox_model.fit(data_df[selected_features + ['duration', 'event']], duration_col='duration', event_col='event')
print(final_cox_model.summary)
