from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_auc_score

def time_dependent_auc(data, time_point):
    # Kaplan-Meier survival function for the testing data
    kmf = KaplanMeierFitter()
    kmf.fit(durations=data['duration'], event_observed=data['event'])
    
    # Convert time_point to duration (assuming time_point is a datetime)
    time_point_duration = (pd.to_datetime(time_point) - pd.to_datetime(start_date)).days / 30  # Convert days to months
    
    # Survival probabilities at the specified time point
    survival_probs = kmf.survival_function_at_times(time_point_duration).values.flatten()
    
    # Create a DataFrame with predicted risk scores and event indicators
    df = pd.DataFrame({
        'predicted_risk': data['predicted_risk'],
        'event': data['event'],
        'survival_prob': survival_probs
    })
    
    # Sort by predicted risk scores
    df = df.sort_values(by='predicted_risk', ascending=False)
    
    # Compute AUC for the sorted DataFrame
    auc = roc_auc_score(df['event'], df['predicted_risk'])
    return auc

# Compute AUC for each time point
auc_scores = {date: time_dependent_auc(test_data, date) for date in time_points_list}

# Display results
for date, auc in auc_scores.items():
    print(f'Time Point: {date.strftime("%Y-%m-%d")}, AUC: {auc:.4f}')
