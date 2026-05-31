import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

num_rows = 1200  # Generate slightly more than 1000 rows to ensure we meet the criteria comfortably

customer_ids = [f"{random.randint(1000, 9999)}-{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}" for _ in range(num_rows)]
genders = [random.choice(["Female", "Male"]) for _ in range(num_rows)]
senior_citizens = [np.random.choice([0, 1], p=[0.84, 0.16]) for _ in range(num_rows)]
partners = [random.choice(["Yes", "No"]) for _ in range(num_rows)]
dependents = []
for partner in partners:
    if partner == "Yes":
        dependents.append(np.random.choice(["Yes", "No"], p=[0.6, 0.4]))
    else:
        dependents.append(np.random.choice(["Yes", "No"], p=[0.1, 0.9]))

tenures = []
contract_types = []
for _ in range(num_rows):
    contract = np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.23, 0.22])
    contract_types.append(contract)
    if contract == "Month-to-month":
        tenures.append(int(np.random.exponential(scale=12) + 1))
    elif contract == "One year":
        tenures.append(int(np.random.normal(loc=30, scale=10)))
    else:
        tenures.append(int(np.random.normal(loc=55, scale=12)))

# Clip tenures to reasonable ranges (1 to 72 months)
tenures = [max(1, min(72, int(t))) for t in tenures]

phone_services = [np.random.choice(["Yes", "No"], p=[0.9, 0.1]) for _ in range(num_rows)]

internet_services = []
for phone in phone_services:
    if phone == "Yes":
        internet_services.append(np.random.choice(["Fiber optic", "DSL", "No"], p=[0.5, 0.3, 0.2]))
    else:
        internet_services.append("DSL")  # If no phone service, they must have DSL or No (typically DSL in these datasets)

monthly_charges = []
for phone, internet in zip(phone_services, internet_services):
    charge = 20.0  # Base charge
    if phone == "Yes":
        charge += 10.0
    if internet == "DSL":
        charge += 30.0 + random.uniform(-5, 10)
    elif internet == "Fiber optic":
        charge += 55.0 + random.uniform(-10, 15)
    else:
        charge += random.uniform(-2, 5)
    monthly_charges.append(round(charge, 2))

total_charges = []
for tenure, charge in zip(tenures, monthly_charges):
    # Total charges is approximately tenure * monthly_charges with some small random variance
    total = tenure * charge * random.uniform(0.95, 1.02)
    total_charges.append(round(total, 2))

# Calculate Churn probability based on features
churn_labels = []
for i in range(num_rows):
    prob = 0.05  # Base churn probability
    
    # Contract type effect
    if contract_types[i] == "Month-to-month":
        prob += 0.35
    elif contract_types[i] == "One year":
        prob += 0.05
        
    # Tenure effect
    if tenures[i] < 12:
        prob += 0.25
    elif tenures[i] < 24:
        prob += 0.15
    elif tenures[i] > 48:
        prob -= 0.15
        
    # Internet service effect
    if internet_services[i] == "Fiber optic":
        prob += 0.15  # Fiber optic typically has higher churn due to higher cost or satisfaction issues in these datasets
        
    # Senior citizen effect
    if senior_citizens[i] == 1:
        prob += 0.08
        
    # Monthly charges effect
    if monthly_charges[i] > 80:
        prob += 0.10
        
    # Standardize/Bound probability
    prob = max(0.01, min(0.95, prob))
    
    churn_labels.append(np.random.choice(["Yes", "No"], p=[prob, 1 - prob]))

# Create DataFrame
df = pd.DataFrame({
    "CustomerID": customer_ids,
    "Gender": genders,
    "SeniorCitizen": senior_citizens,
    "Partner": partners,
    "Dependents": dependents,
    "Tenure (months)": tenures,
    "PhoneService": phone_services,
    "InternetService": internet_services,
    "ContractType": contract_types,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn_labels
})

# Save to CSV
df.to_csv("customer_churn_data.csv", index=False)
print(f"Successfully generated {len(df)} rows of synthetic customer churn data and saved to 'customer_churn_data.csv'.")
