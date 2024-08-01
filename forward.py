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

# Find the top 5 minimal hazard rates for each loan
top_5_minimal_hazard_rates = {}

for loan in individual_hazard_rates_df.columns:
    top_5 = individual_hazard_rates_df[loan].nsmallest(5)
    top_5_minimal_hazard_rates[loan] = top_5

# Convert the dictionary to a DataFrame for better visualization
top_5_minimal_hazard_rates_df = pd.DataFrame(top_5_minimal_hazard_rates)

print("\nTop 5 Time Points with Minimal Hazard Rates for Each Loan:")
print(top_5_minimal_hazard_rates_df)
