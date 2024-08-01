# Ensure the indices of the baseline hazard and hazard ratios align
baseline_hazard = baseline_hazard.reset_index()
baseline_hazard.columns = ['timeline', 'baseline_hazard']

# Create a DataFrame for the individual hazard rates
individual_hazard_rates = pd.DataFrame()

for i in range(len(new_data)):
    # Multiply the baseline hazard by the individual's hazard ratio
    individual_hazard_rate = baseline_hazard.copy()
    individual_hazard_rate['individual_hazard'] = individual_hazard_rate['baseline_hazard'] * hazard_ratios.iloc[i, 0]
    individual_hazard_rate['individual'] = i
    individual_hazard_rates = pd.concat([individual_hazard_rates, individual_hazard_rate])

print("Individual Hazard Rates at Each Time Point:")
print(individual_hazard_rates)
