# Coral Bleaching Predictor

To use this repository, begin with cloning all code files. Preliminarily run the file "Dataset Download.py", which will download the datasets used in the SST Predictor and Bleaching Predictor. The code will also print two lines, the first being the URL for the SST Data, and the second being the URL for the Bleaching Data. 

Place the URL for the SST Data in the labeled line within the "LSTM SST Predictor.py" file, and place the URL for the Bleaching Data in the labeled line within the "Simple Global Bleaching Predictor.py" file. Now, both code files can be run. 

The "LSTM SST Predictor.py" utilizes time-series sea surface temperature data with a resultion of 1 degree from NOAA's COBE-SST 2 Dataset, and is trained to predict 12 months of Sea Surface Temperature values based on 6 months of input (The input length and output length can be adjusted in the LSTM code). 

The "Simple Global Bleaching Predictor.py" utilizes a BMO-DCO Global Bleaching Dataset to predict the percent of bleaching that will occur in a reef based on the Depth of the reef (meters), Sea-Surface Temperature Climate (Can be derived from a year of Sea Surface Temperature data), Latitude (degrees), and Longitude (degrees) in the reef.
