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

# Normalize Data
normalized_data = scaler.fit_transform(data)
sst_data = normalized_data

# Create LSTM Training Sequences
def create_sequences(data, seq_length, n_steps_ahead):
    sequences = []
    targets = []
    for i in range(0, len(data) - seq_length - n_steps_ahead + 1):
        seq = data[i:i + seq_length]
        target = data[i + seq_length:i + seq_length + n_steps_ahead]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Num Taken - Number of Input Months
num_taken = 6

# Num Predicted - Number of Output Months
num_predicted = 12

# Create Dataset
X, y = create_sequences(sst_data, num_taken, num_predicted)

# Split Data Into Training and Testing
X_train, X_val, y_train, y_val =  train_test_split(X, y, test_size=0.2, random_state=42)

X_val, X_test, y_val, y_test =  train_test_split(X_val, y_val, test_size=0.5, random_state=42)

# Convert to Tensors
X_train = torch.tensor(X_train)
X_val = torch.tensor(X_val)
X_test = torch.tensor(X_val)
y_train = torch.tensor(y_train)
y_val = torch.tensor(y_val)
y_test = torch.tensor(y_val)

# Set Input Size, Output Size
input_size = X_train.shape[2]
output_size = num_predicted

# Set Hyperparameters
hidden_size = 50
num_layers = 1
batch_size = 1
learning_rate = 0.01
num_epochs = 50

# Class for the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_predicted):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size * num_predicted)

    # Forward Pass
    def forward(self, x):
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])

        out = out.view(x.size(0), num_predicted, input_size)

        return out

# Intitialize the Model
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_predicted=num_predicted)

# Set criterion
criterion = nn.MSELoss()
# Set optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create Training Dataset and Dataloader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create Validation Dataset and Dataloader
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Create Final Testing Dataset and Dataloader
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for sequences, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()

    val_losses = []
    val_predictions = []
    val_actuals = []

    # Validation
    with torch.no_grad():
        for sequences, targets in val_loader:
            val_outputs = model(sequences)
            val_loss = criterion(val_outputs, targets)
            val_losses.append(val_loss.item())
            val_predictions.append(val_outputs)
            val_actuals.append(targets)
    val_predictions = torch.cat(val_predictions)
    val_actuals = torch.cat(val_actuals)

    val_predictions = scaler.inverse_transform(val_predictions.numpy().reshape(-1, input_size))
    val_actuals = scaler.inverse_transform(val_actuals.numpy().reshape(-1, input_size))

    mae = mean_absolute_error(val_predictions, val_actuals)

    print(f'Epoch {epoch + 1}/{num_epochs}, Val MAE = {mae:.4f}')

# Final Testing
test_losses = []
test_predictions = []
test_actuals = []
with torch.no_grad():
    for sequences, targets in val_loader:
        test_outputs = model(sequences)
        test_loss = criterion(test_outputs, targets)
        test_losses.append(test_loss.item())
        test_predictions.append(test_outputs)
        test_actuals.append(targets)

test_predictions = torch.cat(test_predictions)
test_actuals = torch.cat(test_actuals)

test_predictions = scaler.inverse_transform(test_predictions.numpy().reshape(-1, input_size))
test_actuals = scaler.inverse_transform(test_actuals.numpy().reshape(-1, input_size))

mae = mean_absolute_error(test_predictions, test_actuals)

print(f'Testing MAE = {mae:.4f}')
