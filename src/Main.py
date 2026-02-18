import xarray as xr
import requests
import numpy as np
import torch
import torch.nn as nn
import joblib
import random
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.interpolate import griddata
import calendar
from datetime import datetime, timedelta

# Disable SSL verification warnings
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load trained scalers & models
scaler_sst = joblib.load("scaler_fr_fr.pkl")
scaler_bleaching = joblib.load("scaler_bleaching.pkl")
bleaching_model = joblib.load("extratrees_bleaching_model.pkl")

import calendar


def generate_month_year_labels(start_year=2024, start_month=3):
    month_year_labels = []
    current_year = start_year
    current_month = start_month

    for _ in range(12):
        month_year_labels.append(f"{calendar.month_name[current_month]} {current_year}")

        # Move to next month, update year if needed
        current_month += 1
        if current_month > 12:
            current_month = 1  # Reset to January
            current_year += 1  # Increment year

    return month_year_labels


# Load trained SST model
def load_sst_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    expected_input_size = checkpoint["lstm.weight_ih_l0"].shape[1]

    class SSTLSTMModel(nn.Module):

        def __init__(self, input_size=expected_input_size, hidden_size=8, num_layers=4, num_predicted=12):
            super(SSTLSTMModel, self).__init__()
            self.input_size = input_size
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, input_size * num_predicted)

        def forward(self, x):
            h0 = torch.zeros(4, x.size(0), 8).to(x.device)
            c0 = torch.zeros(4, x.size(0), 8).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out.view(x.size(0), 12, self.input_size)

    model = SSTLSTMModel(input_size=expected_input_size)
    model.load_state_dict(checkpoint)
    model.eval()
    return model, expected_input_size


# Function: Generate random locations with 0.5° granularity
def generate_locations(num_pairs):
    lat_range = np.arange(-89.5, 89.5, 0.5)
    lon_range = np.arange(-180, 180.5, 0.5)
    locations = random.sample([(lat, lon) for lat in lat_range for lon in lon_range], num_pairs)
    return locations


# Function: Fetch NOAA SST Data in 30-Day Batches
def get_real_sst(locations, expected_features):
    BATCH_SIZE = 30
    TOTAL_MONTHS = 6
    start_date = datetime(2024, 3, 9)

    sst_data = {loc: [[] for _ in range(expected_features)] for loc in locations}

    for month_num in range(1, TOTAL_MONTHS + 1):
        print(f"\n Fetching SST for Month {month_num} (30-day batch)")
        batch_sst = {loc: [] for loc in locations}

        for day_offset in range(BATCH_SIZE):
            day = start_date + timedelta(days=(BATCH_SIZE * (month_num - 1) + day_offset))
            date_str = day.strftime('%Y%m%d')
            sst_url = f"https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/5km/v3.1_op/nc/v1.0/daily/sst/2024/coraltemp_v3.1_{date_str}.nc"

            try:
                response = requests.get(sst_url, verify=False, stream=True)
                if response.status_code == 200:
                    dataset = xr.open_dataset(f"sst_{date_str}.nc")
                    sst_data_var = dataset['analysed_sst']

                    if np.nanmax(sst_data_var.values) > 200:
                        sst_data_var = sst_data_var - 273.15

                    for lat, lon in locations:
                        lon_adj = lon - 360 if lon > 180 else lon
                        try:
                            sst = sst_data_var.sel(lat=lat, lon=lon_adj, method='nearest').isel(time=0).values
                            if np.isnan(sst):
                                sst = np.nanmean(sst_data_var.values)
                        except Exception:
                            sst = np.nanmean(sst_data_var.values)

                        batch_sst[(lat, lon)].append(sst)

            except Exception as e:
                print(f" Error fetching data for {date_str}: {e}")

        for loc in locations:
            for feature_idx in range(expected_features):
                sst_data[loc][feature_idx].append(np.mean(batch_sst[loc]) if batch_sst[loc] else np.nan)

    return {loc: np.vstack(values) for loc, values in sst_data.items()}


# Predict SST for Next 12 Months
def predict_next_12_months_sst(model, past_sst):
    past_sst = past_sst.T
    normalized_sst = scaler_sst.transform(past_sst)
    input_tensor = torch.tensor(normalized_sst, dtype=torch.float32).reshape(1, 6, -1)

    with torch.no_grad():
        predictions = model(input_tensor)

    predictions = predictions.numpy().reshape(12, -1)
    sst_predictions = scaler_sst.inverse_transform(predictions)
    sst_predictions_k = sst_predictions + 273.15
    return sst_predictions, sst_predictions_k


# Predict Bleaching Risk
def predict_bleaching_for_all_months(longitude, latitude, sst_k):
    bleaching_predictions = []
    for month_idx, sst_values in enumerate(sst_k):
        sst_value = sst_values[0]
        X_input = np.array([[longitude, latitude, sst_value]])
        X_scaled = scaler_bleaching.transform(X_input)
        bleaching_risk = bleaching_model.predict(X_scaled)[0]
        bleaching_predictions.append(bleaching_risk)
    return bleaching_predictions


def plot_global_maps(locations, bleaching_predictions, start_year=2024, start_month=3):
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    lats, lons = zip(*locations)
    bleaching_predictions = np.array(bleaching_predictions)
    grid_lat, grid_lon = np.mgrid[-90:90.5:0.5, -180:180.5:0.5]

    # Use fixed month-year labels
    month_year_labels = generate_month_year_labels(start_year, start_month)

    for month in range(12):
        print(f" Generating heatmap for {month_year_labels[month]}")
        risks = bleaching_predictions[:, month]

        grid_risk = griddata((lats, lons), risks, (grid_lat, grid_lon), method="cubic", fill_value=np.nanmean(risks))

        fig, ax = plt.subplots(figsize=(12, 6))
        world.plot(ax=ax, color="lightgray", edgecolor="black")
        im = ax.imshow(grid_risk, extent=(-180, 180, -90, 90), origin="lower", cmap="YlOrRd", alpha=0.75)
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.7)
        cbar.set_label("Predicted Coral Bleaching Probability (%)")
        ax.set_title(f"Interpolated Coral Bleaching Risk - {month_year_labels[month]}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

# MAIN EXECUTION
if __name__ == "__main__":
    model, correct_input_size = load_sst_model("lstm_model_new_fr.pth")
    locations = generate_locations(600)
    real_sst_data = get_real_sst(locations, correct_input_size)
    bleaching_results = [
        predict_bleaching_for_all_months(lon, lat, predict_next_12_months_sst(model, real_sst_data[(lat, lon)])[1]) for
        lat, lon in locations]
    plot_global_maps(locations, np.array(bleaching_results))
