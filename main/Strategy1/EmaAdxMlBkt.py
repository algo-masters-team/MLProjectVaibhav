import pandas as pd
import numpy as np
import yfinance as yf
import talib
from backtesting import Backtest,Strategy
from backtesting.test import EURUSD,GOOG
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR
import os

def Fetchdata():
    cwd = os.getcwd()
    file_path = os.path.join(cwd, 'data', 'ETH-USDT', 'ETH-USDT_1d.csv')
    data1 = pd.read_csv(file_path)
    var = int((len(data1))/2)
    data = data1[var:]
    data.columns = ['Timestamp','Open', 'High', 'Low', 'Close', 'Volume']
    data['Date'] = pd.to_datetime(data['Timestamp'])
    data.set_index('Date', inplace=True)
    return data

class GoldenDeathCross(Strategy):
    stlo=98
    tkpr=102
    Volume_var=14
    mom_var=14
    adx_var=14
    ema1_var=20
    ema2_var=50
    rsi_var=7

    def init(self):
        self.ema1 = self.I(talib.EMA,self.data.Close,timeperiod=self.ema1_var)
        self.ema2 = self.I(talib.EMA,self.data.Close,timeperiod=self.ema2_var)
        self.volume_avg =self.I(talib.SMA,self.data.Volume.astype(float),timeperiod=self.Volume_var) # used in ml
        self.adx = self.I(talib.ADX,self.data.High,self.data.Low,self.data.Close,timeperiod=self.adx_var) # used in ml
        self.mom = self.I(talib.MOM,self.data.Close,timeperiod=self.mom_var) # used in ml
        self.rsi = self.I(talib.RSI,self.data.Close,timeperiod=self.rsi_var) # used in ml
        self.Train()

    def Train(self):
        close_series = pd.Series(self.data.Close, index=self.data.index)
        self.indicator_df = pd.DataFrame(index=self.data.index)
        self.indicator_df['ADX'] = self.adx
        self.indicator_df['MOM'] = self.mom
        self.indicator_df['VolumeAvg'] = self.volume_avg
        self.indicator_df['Rsi'] = self.rsi
        self.indicator_df['returns'] = close_series.pct_change(1)
        
        self.indicator_df.dropna(inplace=True)

        split = int(len(self.indicator_df) * 0.8)
    
        x_train = self.indicator_df[['ADX', 'MOM','VolumeAvg','Rsi']].iloc[:split]
        y_train = self.indicator_df['returns'].iloc[:split]

        reg = Pipeline([
            ('std_scaler', StandardScaler()),
            ('pca',PCA(n_components=4)),
            ('reg', VotingRegressor(estimators=[
                ('rf', RandomForestRegressor(n_estimators=500,max_depth=10,min_samples_split=5,min_samples_leaf=2)),
                ('knn', KNeighborsRegressor(n_neighbors=2,weights='distance')),
                ('mlp', MLPRegressor(hidden_layer_sizes=(150, 100, 50),activation='relu',solver='adam',alpha=0.001)),
                ('svr', SVR(C=1,kernel='rbf')),
                ('lin', LinearRegression())
                ]))
            ])

        reg.fit(x_train, y_train)
        
        self.indicator_df['predict'] = reg.predict(self.indicator_df[['ADX', 'MOM', 'VolumeAvg','Rsi']])
        self.indicator_df["position_reg"] = np.sign(self.indicator_df["predict"])

    
    def next(self):
        if ((self.ema1[-2]<self.ema2[-2] and self.ema1[-1]>self.ema2[-1]) and (self.adx[-1]>self.adx[-2] or self.mom[-1]>0)):
            self.position.close()
            self.buy(sl=((self.data.Close*self.stlo)/100),tp=((self.data.Close*self.tkpr)/100))

        elif ((self.ema1[-2]>self.ema2[-2] and self.ema1[-1]<self.ema2[-1]) and (self.adx[-1]<self.adx[-2] or self.mom[-1]>0)):
            self.position.close()
            self.sell(sl=((self.data.Close*self.tkpr)/100),tp=((self.data.Close*self.stlo)/100))

def main():
    data=Fetchdata()
    bt=Backtest(data,GoldenDeathCross,cash=10000000)
    bt.run()
    stats1=bt.optimize(
        stlo=range(98,99,1),
        tkpr=range(103,104,3),
        adx_var=range(10,20,5),
        mom_var=range(10,20,5),
        rsi_var=range(5,10,5),
        ema1_var=range(10,20,5),
        ema2_var=range(35,55,10),
        maximize='Sharpe Ratio'
    )
    print(stats1)
    bt.plot()

main()

# file:///C:/Users/Vaibhav/OneDrive/Documents/FolderPython/AlgorithmTrading/Official_project_01/GoldenDeathCross_stlo-98,tkpr-102,ema1_var-10,ema2_var-35_.html
# file:///C:/Users/Vaibhav/OneDrive/Documents/FolderPython/AlgorithmTrading/Official_project_01/GoldenDeathCross_stlo-96,tkpr-103,ema1_var-10,ema2_var-45_.html
# file:///C:/Users/Vaibhav/OneDrive/Documents/FolderPython/AlgorithmTrading/Official_project_01/GoldenDeathCross_stlo-98,tkpr-103,ema1_var-20,ema2_var-55_.html