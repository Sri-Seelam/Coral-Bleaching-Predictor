import ssl
import os
from urllib.request import urlopen

# Create a context
context = ssl._create_unverified_context()

# Get URL, File Name for SST Data
url = "https://downloads.psl.noaa.gov/Datasets/COBE2/sst.mon.mean.nc"
filename = "SST_Time_Series_Data.nc"

# Write file
with urlopen(url, context=context) as response, open(filename, 'wb') as out_file:
    out_file.write(response.read())

# Get File Path
file_path = os.path.abspath(filename)

print(f"Downloaded SST Data to: {file_path}")

# Repeat for Bleaching Dataset

# Get URL, File Name for Bleaching Dataset Data
url = "https://datadocs.bco-dmo.org/file/B11vA82u7y2Owp/global_bleaching_environmental.csv"
filename = "Bleaching_Dataset.csv"

# Write file
with urlopen(url, context=context) as response, open(filename, 'wb') as out_file:
    out_file.write(response.read())

# Get File Path
file_path = os.path.abspath(filename)

print(f"Downloaded Bleaching Dataset to: {file_path}")
