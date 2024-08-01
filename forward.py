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
final_cox_model = CoxPHFitter()
final_cox_model.fit(data_df[selected_features + ['duration', 'event']], duration_col='duration', event_col='event')
print(final_cox_model.summary)
