# COMPAI_HW1 - Predict the electricity generating loading of power plants by LSTM
NE6081080 COMPAI 1st homework - electricity load prediction

This is a program that use power supply dataset(https://data.gov.tw/dataset/19995) provided by Taiwan Power Company to predict the loading of Taiwan's power plants.
Considering the loading of Taiwan's power plants is decided by the ability of generate electricity and the electricity usage of the country.
We use ARIMA with only reserve data from Taiwan Power Company to forcast the future trend.

![image](https://user-images.githubusercontent.com/12568316/160665248-df4acdc9-2602-4956-9bd6-19ee2ae352ac.png)

To execute the program:

1. Clone program from github(https://github.com/Welly0902/COMPAI_HW1)

2. Set up the environment:
* pipenv install pandas

3. Run the program:
* `pipenv run python app.py --training <training_dataset_name>.csv --output submission.csv --test <testing_dataset_name>.csv`
or simply run `pipenv run python app.py`
5. Get the prediction result of Jan and Feb from submission.csv 
