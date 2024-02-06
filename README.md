# finalproject
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Sample data 
data = {
    'Crop': ['Wheat', 'Rice', 'Maize', 'Wheat', 'Rice'],
    'Variety': ['Variety1', 'Variety2', 'Variety3', 'Variety1', 'Variety2'],
    'State': ['Punjab', 'Uttar Pradesh', 'Madhya Pradesh', 'Punjab', 'Uttar Pradesh'],
    'Quantity': [100, 150, 80, 120, 200],
    'Production': [2020, 2020, 2021, 2019, 2020],
    'Season': ['medium', 'long', 'medium', 'long', 'medium'],
    'Unit': ['Tons', 'Tons', 'Tons', 'Tons', 'Tons'],
    'Cost': [50000, 75000, 40000, 60000, 80000],
    'Recommended Zone': ['Zone1', 'Zone2', 'Zone3', 'Zone1', 'Zone2']
}

df = pd.DataFrame(data)

# Preprocess data
label_encoder = LabelEncoder()
df['Season'] = label_encoder.fit_transform(df['Season'])
df['Recommended Zone'] = label_encoder.fit_transform(df['Recommended Zone'])

# Convert 'Production' to years
current_year = datetime.now().year
df['YearsSinceProduction'] = current_year - df['Production']

# Select features and target variable
X = df[['Quantity', 'Season', 'Cost', 'Recommended Zone', 'YearsSinceProduction']]
y = df['Production']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Regressor model
model = DecisionTreeRegressor()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Example: Make a prediction for a new data point
new_data_point = [[130, label_encoder.transform(['medium'])[0], 55000, label_encoder.transform(['Zone1'])[0], current_year - 2]]
predicted_production = model.predict(new_data_point)
print(f'Predicted Production for the new data point: {predicted_production[0]} Tons')
