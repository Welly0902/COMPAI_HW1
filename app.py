import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def readData(path):
  df = pd.read_csv(path)
  # df = df[['日期','備轉容量(MW)']]
  # df.columns = ['time','left']
  # df['time'] = pd.to_datetime(df['time'],format='%Y%M%d')
  # df['time'] = df['time'].dt.date
  df = df['備轉容量(MW)']
  df.columns=['left']
  print(df)
  return df

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
    # parser.add_argument('--test',
    #                     default='test.csv',
    #                     help='testing data')
    args = parser.parse_args()
    
    # The following part is an example.
    # You can modify it at will.
    # import pandas as pd
    # df_training = pd.read_csv(args.training)

    ##############
    #### main ####
    ##############

    series = readData(args.training)
    # series=series['left']
    # print(series.shape)
    # print(series.dtypes)
    # series.plot(x='time',figsize=(15, 10))
    # plt.show()

    from pandas.plotting import autocorrelation_plot
 
    # def parser(x):
	  # return datetime.strptime('190'+x, '%Y-%m')
 
    # autocorrelation_plot(series)
    # plt.show()

    # model = ARIMA(series, order=(5,1,0))#20-10.565
    model = ARIMA(series, order=(50,1,0))
    
    model_fit = model.fit()
    output = model_fit.forecast(15)

    print(type(output))

    result = pd.DataFrame(output)
    result['date'] = pd.date_range(start='30/3/2022', periods=15)
    result['date'] = result['date'].dt.strftime('%Y%m%d')
    result.columns=['operating_reserve(MW)','date']
    result = result[['date','operating_reserve(MW)']]
    result['operating_reserve(MW)'] = round(result['operating_reserve(MW)']).astype(int)
    # result= result['date',]
    # result['date'] = result['date'].replace("-","")
    # print(result)

    result.to_csv(args.output, index=0)