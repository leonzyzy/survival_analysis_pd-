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
    print(f'Time Point: {date.strftime("%Y-%m-%d")}, AUC: {auc:.import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification

# Set seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 9399
X, y = make_classification(n_samples=n_samples, n_features=20, n_classes=2, n_clusters_per_class=1, flip_y=0.3, random_state=42)

# Simulate predicted probabilities (for simplicity, use a logistic regression model here)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
y_scores = model.predict_proba(X)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y, y_scores)
roc_auc = auc(fpr, tpr)

# Adjust AUC if necessary
desired_auc = 0.70
if np.isclose(roc_auc, desired_auc, atol=0.01):
    print(f"AUC is approximately {roc_auc:.2f}, which is close to the desired AUC of {desired_auc:.2f}.")
else:
    print(f"AUC is {roc_auc:.2f}, which may not be exactly the desired AUC of {desired_auc:.2f}.")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


