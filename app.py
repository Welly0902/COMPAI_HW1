import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

## read training data
def readData(path):
  data = pd.read_csv(path)
  return data

## preprocess csv data to dataframe for training
def augFeatures(df):
  df_train=df
  # Make some adjustment to the original data to the column we want
  cols = ['timestamp','peakprov','peakload','left','%left','industryuse','peopleuse','nuclear','coal','cogen','ippcoal','lng','ipplng','hydro','wind','solar','others']
  df_train[cols[0]]=df['日期']
  df_train[cols[1]]=df['淨尖峰供電能力(MW)']
  df_train[cols[2]]=df['尖峰負載(MW)']
  df_train[cols[3]]=df['備轉容量(MW)']
  df_train[cols[4]]=df['備轉容量率(%)']
  df_train[cols[5]]=df['工業用電(百萬度)']
  df_train[cols[6]]=df['民生用電(百萬度)']
  df_train[cols[7]]=df['核一#1(萬瓩)']+df['核一#2(萬瓩)']+df['核二#1(萬瓩)']+df['核二#2(萬瓩)']+df['核三#1']+df['核三#2']
  df_train[cols[8]]=df['林口#1']+df['林口#2']+df['林口#3']+df['台中#1']+df['台中#2']+df['台中#3']+df['台中#4']+df['台中#5']+df['台中#6']+df['台中#7']+df['台中#8']+df['台中#9']+df['台中#10']+df['興達#1']+df['興達#2']+df['興達#3']+df['興達#4']+df['大林#1']+df['大林#2']
  df_train[cols[9]]=df['汽電共生']
  df_train[cols[10]]=df['和平#1']+df['和平#2']+df['麥寮#1']+df['麥寮#2']+df['麥寮#3']
  df_train[cols[11]]=df['大潭 (#1-#6)']+df['通霄 (#1-#6)']+df['興達 (#1-#5)']+df['南部 (#1-#4)']+df['大林(#5-#6)']
  df_train[cols[12]]=df['海湖 (#1-#2)']+df['國光 #1']+df['新桃#1']+df['星元#1']+df['嘉惠#1']+df['星彰#1']+df['豐德(#1-#2)']
  df_train[cols[13]]=df['德基']+df['青山']+df['谷關']+df['天輪']+df['馬鞍']+df['萬大']+df['大觀']+df['鉅工']+df['大觀二']+df['明潭']+df['碧海']+df['立霧']+df['龍澗']+df['卓蘭']+df['水里']+df['其他小水力']
  df_train[cols[14]]=df['風力發電']
  df_train[cols[15]]=df['太陽能發電']
  df_train[cols[16]]=df['協和 (#1-#4)']+df['氣渦輪']+df['離島']

  # Add season columns from timestamp 
  df_train['timestamp2'] = df_train['timestamp'].apply(str)
  df_train['timestamp2'] = df_train['timestamp2'].str[4:6]

  conditions = [
    (df_train['timestamp2'] == ('01' or '02' or '03')),
    (df_train['timestamp2'] == ('04' or '05' or '06')),
    (df_train['timestamp2'] == ('07' or '08' or '09')),
    (df_train['timestamp2'] == ('10' or '11' or '12'))
    ]
  # 0 as spring, 1 as summer, 2 as fall, 6 as winter
  values = [0, 1, 2, 3]
  df_train['season'] = np.select(conditions, values)
  
  # Select and sort the columns we want
  cols2=['timestamp','peakprov','peakload','industryuse','peopleuse','nuclear','coal','cogen','ippcoal','lng','ipplng','hydro','wind','solar','others','season','left']
  df_train=df_train[cols2]
  return df_train

## transform dataframe and prepare to generate time series data
def normalize(data):
  data_t = data
  data_t = data_t.drop(['timestamp'], axis=1)
  data_t = data_t.drop(['season'], axis=1)
  data_t = data_t.drop(['left'], axis=1)
  # data_norm = data_t.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
  data_norm = data_t
  data_norm['season'] = data['season']
  data_norm['left'] = data['left']
  cols=['peakprov','peakload','industryuse','peopleuse','nuclear','coal','cogen','ippcoal','lng','ipplng','hydro','wind','solar','others','season','left']
  data_norm=data_norm[cols]
  return data_norm

## form time sires data
def buildTrain(train, pastDay=7, futureDay=1):
  X_train, Y_train = [], []
  print(train.shape[0]-futureDay-pastDay)
  for i in range(train.shape[0]-futureDay-pastDay):
    X_train.append(np.array(train.iloc[i:i+pastDay,:-1]))    
    Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]['left']))
  return np.array(X_train), np.array(Y_train)

## shuffle training data
def shuffle(X,Y):
  np.random.seed(15)
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  return X[randomList], Y[randomList]

## split validation data
def splitData(X,Y,rate):
  X_train = X[int(X.shape[0]*rate):]
  Y_train = Y[int(Y.shape[0]*rate):]
  X_val = X[:int(X.shape[0]*rate)]
  Y_val = Y[:int(Y.shape[0]*rate)]
  return X_train, Y_train, X_val, Y_val

## build lstm model
def buildLstm(x_train):
  # model = Sequential()
  # model.add(LSTM(30, input_length=shape[1], input_dim=shape[2]))
  # # output shape: (1, 1)
  # model.add(Dense(1))
  # model.compile(loss="mse", optimizer="adam")
  # model.summary()
  n_features = x_train.shape[1]


  # define model
  model = Sequential()
  model.add(LSTM(50, activation='relu', input_shape=(1, n_features)))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  # model.fit(x_train, y_train, epochs=1000, verbose=1)
  model.summary()
  return model

## generate dataframe for submission.csv
def genResultdf(df):
  dft = augFeatures(df)
  # dft = dft.apply(lambda x: model.predict(x_input, verbose=0))
  
  dft_norm = normalize(dft)
  print(dft_norm)
  print(dft_norm.shape)
  xp, yp = buildTrain(dft_norm,7,1)
  print(xp)
  print(len(xp))
  predictions=[]
  for i in xp:
    s=np.array(i)
    s=s.reshape((1, 7, 15))
    predictions.append(model.predict(s, verbose=0)[0][0])
    df_res = pd.DataFrame(predictions, columns=['operating_reserve(MW)'])
  #   print(model.predict(i, verbose=0))
    df_res['date'] = dft['timestamp']
    df_res=df_res[['date','operating_reserve(MW)']]
  return df_res


# You can write code above the if-main block.
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    # add one more parameter for input testing data
    parser.add_argument('--test',
                        default='test.csv',
                        help='testing data')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    # import pandas as pd
    # df_training = pd.read_csv(args.training)

    ##############
    #### main ####
    ##############

    # set up windows of timeseries data
    n_steps=7
    # read file for training
    data = readData(args.training)
    # data preprocessing
    df = augFeatures(data)
    df = normalize(df)
    # print(df.shape)
    # print(df.sample(frac=1))
    df = df.sample(frac=1)
    # x_train, y_train = buildTrain(df,7,1)
    x_train=df.drop('left',1)
    # print(x_train.columns)
    y_train=df['left']

    # x_train, y_train = shuffle(x_train, y_train)
    # x_train, y_train, x_val, y_val = splitData(x_train, y_train, 0.1)

    # train lstm model
    # print(x_train.shape)
    # print(x_train)
    model = buildLstm(x_train)

    x_train = np.reshape(x_train.to_numpy(), (x_train.shape[0], 1, x_train.shape[1]))
    # print(type(x_train))
    # print(x_train)
    # print(x_train.to_numpy())
    # y_train = np.reshape(y_train.to_numpy(), (y_train.shape[0], 1, y_train.shape[1]))
    y_train = y_train.to_numpy()
    model.fit(x_train, y_train, epochs=300, verbose=1)

    # read testing data and generate result
    df_test=pd.read_csv(args.test,encoding = 'big5')
    # df_res=pd.read_csv(args.test)
    df_res = augFeatures(df_test)
    df_res = normalize(df_res)
    test = df_res.drop('left',1)
    test = np.reshape(test.to_numpy(), (test.shape[0], 1, test.shape[1]))
    # print(test)

    predictions = model.predict(test)
    # print(predictions)
    df_result = pd.DataFrame()
    df_result['date'] = df_test['日期']
    df_result['operating_reserve(MW)'] = pd.DataFrame(predictions)
    # print(df_result)
    # df_result = genResultdf(df_res)
    df_result.to_csv(args.output, index=0)
    # print(df_res)
    print("submission.csv generated!")