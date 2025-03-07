import netCDF4 as nc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load the NetCDF File
file_path = ''
ds = nc.Dataset(file_path)

# Extract SST Data, Time, Lat, Lon
sst = ds.variables['sst'][:]
time = ds.variables['time'][:]
lat = ds.variables['lat'][:]
lon = ds.variables['lon'][:]

# Convert Time to a Readable Format
time_units = ds.variables['time'].units
dates = nc.num2date(time, units=time_units)

# Fill Masked Values with NaN
sst = np.ma.filled(sst, np.nan)

# List of Coral Reef Locations (Rounded to Nearest Degree)
coral_reefs = [(-18, 148), (17, -88), (26, 36), (25, -78), (-21, 166),
               (25, -81), (20, -87), (9, 120), (-22, 113), (13, 120),
               (-9, 46), (-6, 72), (3, 73), (11, 72), (29, 34),
               (29, 35), (17, -63), (18, -64), (25, -83), (-17, 179)]


# Function to Find the Closest Indices for a Given Latitude and Longitude
def find_indices(lat_array, lon_array, lat_val, lon_val):
    lat_idx = np.abs(lat_array - lat_val).argmin()
    lon_idx = np.abs(lon_array - lon_val).argmin()
    return lat_idx, lon_idx


# Create a DataFrame to Hold Time Series Data for Each Coral Reef
data_dict = {'date': dates}
valid_reefs = []

for lat_val, lon_val in coral_reefs:
    lat_idx, lon_idx = find_indices(lat, lon, lat_val, lon_val)
    sst_location = sst[:, lat_idx, lon_idx]

    # Handle Masked Values
    if np.ma.is_masked(sst_location):
        sst_location = sst_location.filled(np.nan)

    # Add Data Only if it Doesn't Contain NaNs
    if not np.isnan(sst_location).any():
        data_dict[f'reef_{len(valid_reefs) + 1}'] = sst_location
        valid_reefs.append((lat_val, lon_val))

data = pd.DataFrame(data_dict)
data.set_index('date', inplace=True)

# Initialize Scaler
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)
sst_data = normalized_data

# Save the Scaler for Later Use
joblib.dump(scaler, 'scaler_fr_fr.pkl')


# Create LSTM Training Sequences
def create_sequences(data, seq_length, n_steps_ahead):
    sequences, targets = [], []
    for i in range(len(data) - seq_length - n_steps_ahead + 1):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length:i + seq_length + n_steps_ahead])
    return np.array(sequences), np.array(targets)


num_taken, num_predicted = 6, 12
X, y = create_sequences(sst_data, num_taken, num_predicted)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Convert to Tensors
X_train, X_test = map(torch.tensor, (X_train, X_test))
y_train, y_test = map(torch.tensor, (y_train, y_test))

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_predicted):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size * num_predicted)

    def forward(self, x):
        h0, c0 = (torch.zeros(num_layers, x.size(0), hidden_size).to(x.device),
                  torch.zeros(num_layers, x.size(0), hidden_size).to(x.device))
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out.view(x.size(0), num_predicted, input_size)


# Initialize Model and Training Parameters
input_size, hidden_size, num_layers, batch_size, learning_rate, num_epochs = X_train.shape[2], 8, 4, 4, 0.001, 120
model = LSTMModel(input_size, hidden_size, num_layers, num_predicted)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for sequences, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for sequences, targets in test_loader:
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    test_loss /= len(test_loader)

    print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Testing Loss: {test_loss:.4f}')

# Save the Model
torch.save(model.state_dict(), 'lstm_model_new_fr.pth')
print("Model saved as 'lstm_model_new_fr.pth'")
