survival_function = pd.DataFrame(index=hazard_rates.index, columns=hazard_rates.columns)

# Compute the survival function for each individual
for column in hazard_rates.columns:
    cumulative_hazard = hazard_rates[column].cumsum()
    survival_function[column] = np.exp(-cumulative_hazard)

print("Survival Function:")
print(survival_function)

# Calculate the probability of default
probability_of_default = 1 - survival_function
