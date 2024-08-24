import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import statsmodels.api as sm

# Load the data
data = pd.read_csv('data.csv')

# Filter the data
filtered_data = data[(data['Reporting Area'] == 'TOTAL') & 
                     (data['Current MMWR Year'].between(2022, 2024))]

# Sort data by year and week
filtered_data = filtered_data.sort_values(by=['Current MMWR Year', 'MMWR WEEK'])

# Get the top 8 diseases in 2022
top_diseases_2022 = filtered_data[filtered_data['Current MMWR Year'] == 2022]
top_diseases_2022 = top_diseases_2022.groupby('Label')['Cumulative YTD Current MMWR Year'].max().nlargest(8).index

print(top_diseases_2022)
print(len(data['Label'].unique()))
# Filter data for top 8 diseases
top_diseases_data = filtered_data[filtered_data['Label'].isin(top_diseases_2022)]

# Create a new column for year-week
top_diseases_data['Year-Week'] = top_diseases_data['Current MMWR Year'].astype(str) + '-' + top_diseases_data['MMWR WEEK'].astype(str)

# Plotting
plt.figure(figsize=(14, 8))
linestyles = ['-', '--', '-.', ':']
widths = [1, 2]

for i, disease in enumerate(top_diseases_2022):
    disease_data = top_diseases_data[top_diseases_data['Label'] == disease]
    plt.plot(disease_data['Year-Week'], disease_data['Cumulative YTD Current MMWR Year'], 
             label=disease, color='black', linestyle=linestyles[i % 4], linewidth=widths[i // 4])

# Rotate xticks and set xticks
plt.xticks(rotation=90)
plt.xticks(ticks=range(0, len(top_diseases_data['Year-Week'].unique()), len(top_diseases_data['Year-Week'].unique()) // 15))

# Add legend
plt.legend()

# Add labels and title
plt.xlabel('Year-Week')
plt.ylabel('Cumulative YTD Current MMWR Year')
plt.title('Top 8 Diseases from 2022 to 2024')

# Use tight layout
plt.tight_layout()
plt.savefig('result.png',dpi=300)
# Show plot
plt.show()

# Linear regression for top 2 diseases for each year
top_2_diseases = top_diseases_2022[:2]

for disease in top_2_diseases:
    for year in [2022, 2023, 2024]:
        disease_data = top_diseases_data[(top_diseases_data['Label'] == disease) & (top_diseases_data['Current MMWR Year'] == year)]
        X = np.array(disease_data['MMWR WEEK']).reshape(-1, 1)
        y = disease_data['Cumulative YTD Current MMWR Year']
        
        # Fit the model using sklearn
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]
        
        # Fit the model using statsmodels for p-value and R-squared
        X_sm = sm.add_constant(X)  # Add a constant term for the intercept
        model_sm = sm.OLS(y, X_sm).fit()
        p_value = model_sm.pvalues[1]  # p-value for the slope coefficient
        r_squared = model_sm.rsquared
        
        print(f"{disease} slope coefficient for {year}: {slope:.5f}")
        print(f"{disease} p-value for {year}: {p_value:.5f}")
        print(f"{disease} R-squared for {year}: {r_squared:.5f}")

