survival_function = pd.DataFrame(index=hazard_rates.index, columns=hazard_rates.columns)

# Compute the survival function for each individual
for column in hazard_rates.columns:
    cumulative_hazard = hazard_rates[column].cumsum()
    survival_function[column] = np.exp(-cumulative_hazard)

print("Survival Function:")
print(survival_function)

# Calculate the probability of default
probability_of_default = 1 - survival_function


# Initialize DataFrames to store survival probabilities and default probabilities
survival_probabilities = pd.DataFrame(index=hazard_rates.index, columns=hazard_rates.columns)
default_probabilities = pd.DataFrame(index=hazard_rates.index, columns=hazard_rates.columns)

# Compute the survival and default probabilities
for column in hazard_rates.columns:
    # Initialize survival probability for the first time point
    survival_prob = 1.0
    survival_probs = []
    default_probs = []
    
    for t in hazard_rates.index:
        hazard_rate = hazard_rates.loc[t, column]
        
        # Calculate survival probability at this time point
        survival_prob *= np.exp(-hazard_rate)
        survival_probs.append(survival_prob)
        
        # Calculate default probability
        default_probs.append(1 - survival_prob)
    
    survival_probabilities[column] = survival_probs
    default_probabilities[column] = default_probs

print("Survival Probabilities:")
print(survival_probabilities)



from lifelines import KaplanMeierFitter
from sklearn.metrics import roc_auc_score
import numpy as np

def time_dependent_auc(data, time_point):
    # Kaplan-Meier survival function for the observed data
    kmf = KaplanMeierFitter()
    kmf.fit(durations=data['duration'], event_observed=data['event'])
    
    # Survival probabilities at the specified time point
    survival_probs = kmf.survival_function_at_times(time_point).values.flatten()
    
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

# Define time points at which to compute the AUC
time_points = np.arange(0, 30, 5)

# Compute AUC for each time point
auc_scores = {time: time_dependent_auc(data, time) for time in time_points}

# Display results
for time, auc in auc_scores.items():
    print(f'Time: {time}, AUC: {auc:.4f}')





