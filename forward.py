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
