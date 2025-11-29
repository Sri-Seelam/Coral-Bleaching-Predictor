import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
import joblib

# Path to your CSV file
file_path = ''

# Load the CSV data into a pandas DataFrame
data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(data.head())

# Convert necessary columns to numerical values
data['Distance_to_Shore'] = pd.to_numeric(data['Distance_to_Shore'], errors='coerce')
data['Depth_m'] = pd.to_numeric(data['Depth_m'], errors='coerce')
data['Percent_Bleaching'] = pd.to_numeric(data['Percent_Bleaching'], errors='coerce')
data['Latitude_Degrees'] = pd.to_numeric(data['Latitude_Degrees'], errors='coerce')
data['Longitude_Degrees'] = pd.to_numeric(data['Longitude_Degrees'], errors='coerce')
data['Temperature_Mean'] = pd.to_numeric(data['Temperature_Mean'], errors='coerce')
data['SSTA'] = pd.to_numeric(data['SSTA'], errors='coerce')

# Remove rows with missing values
data = data.dropna(subset=['Longitude_Degrees', 'Latitude_Degrees', 'Temperature_Mean', 'Percent_Bleaching'])

X = data[['Longitude_Degrees', 'Latitude_Degrees', 'Temperature_Mean']]
y = data['Percent_Bleaching']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for later use
joblib.dump(scaler, 'scaler_bleaching.pkl')

model = ExtraTreesRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred_test = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred_test)
print(mae)

# Save the trained model
joblib.dump(model, 'extratrees_bleaching_model.pkl')
print("Model saved as 'extratrees_bleaching_model.pkl'")
