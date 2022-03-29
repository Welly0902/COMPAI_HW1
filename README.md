# COMPAI_HW1 - Predict the electricity generating loading of power plants by LSTM
NE6081080 COMPAI 1st homework - electricity load prediction

This is a program that use power supply dataset(https://data.gov.tw/dataset/19995) provided by Taiwan Power Company to predict the loading of Taiwan's power plants.
Considering the loading of Taiwan's power plants is decided by the ability of generate electricity and the electricity usage of the country.
Except for the ability of power plants information, I also take the season, which will have significant influence to the temperature(https://www.cwb.gov.tw/V8/C/C/Statistics/monthlymean.html), into account. 
I use LSTM as model structure and can finally get the result in the submission.csv as follow:

date,operating_reserve(MW)
20210323,2557
20210324,1899
20210325,1891
20210326,1811
20210327,1903
20210328,2333
20210329,1800

To execute the program:

1. Clone program from github(https://github.com/Welly0902/COMPAI_HW1)

2. Set up the environment:
* python 3.7.7
* pip install -r requirements.txt

3. Download dataset from https://data.gov.tw/dataset/19995.
* Select the data as training dataset(<training_dataset_name>.csv)
* Select the date duration you want as testing dataset(<testing_dataset_name>.csv)

4. Run the program:
* python app.py --training <training_dataset_name>.csv --output submission.csv --test <testing_dataset_name>.csv
