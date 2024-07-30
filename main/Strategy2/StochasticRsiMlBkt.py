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
    # data1 = pd.read_csv(r"C:\Users\Vaibhav\OneDrive\Documents\FolderPython\AlgorithmTrading\Official_project_01\data\BTC-USDT\BTC-USDT_1d.csv")
    # data1 = pd.read_csv(r"C:\Users\Vaibhav\OneDrive\Documents\FolderPython\AlgorithmTrading\Official_project_01\data\ETH-USDT\ETH-USDT_1d.csv")
    data1 = pd.read_csv(r"C:\Users\Vaibhav\OneDrive\Documents\FolderPython\AlgorithmTrading\Official_project_01\data\SOL-USDT\SOL-USDT_1d.csv")
    var = int((len(data1))/2)
    data = data1[var:]
    data.columns = ['Timestamp','Open', 'High', 'Low', 'Close', 'Volume']
    data['Date'] = pd.to_datetime(data['Timestamp'])
    data.set_index('Date', inplace=True)
    return data

class StochasticEma(Strategy):
    stlo=98
    tkpr=102
    Volume_var=14
    mom_var=14
    adx_var=14
    rsi_var=7
    stochrsi_min=25
    stochrsi_max=65
    stochrsi_tp_var=14

    def init(self):
        self.volume_avg =self.I(talib.SMA,self.data.Volume.astype(float),timeperiod=self.Volume_var) # used in ml
        self.adx = self.I(talib.ADX,self.data.High,self.data.Low,self.data.Close,timeperiod=self.adx_var) # used in ml
        self.mom = self.I(talib.MOM,self.data.Close,timeperiod=self.mom_var) # used in ml
        self.rsi = self.I(talib.RSI,self.data.Close,timeperiod=self.rsi_var) # used in ml
        self.ema200 = self.I(talib.EMA,self.data.Close,timeperiod=200)
        self.ema20 = self.I(talib.EMA,self.data.Close,timeperiod=20)
        self.ema50 = self.I(talib.EMA,self.data.Close,timeperiod=50)
        self.STOCHK, self.STOCHD = self.I(talib.STOCHRSI, self.data.Close, timeperiod=self.stochrsi_tp_var, fastk_period=3, fastd_period=3, fastd_matype=0)
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
        if ((self.indicator_df['predict'].iloc[-1]>0) and (self.STOCHK[-1]<self.stochrsi_min) and (self.ema20[-1]-self.ema50[-1]>self.ema20[-2]-self.ema50[-2] and self.ema20[-1]<self.ema50[-1])):
            self.position.close()
            self.buy(sl=((self.data.Close*self.stlo)/100),tp=((self.data.Close*self.tkpr)/100))
            
        elif ((self.indicator_df['predict'].iloc[-1]<0) and (self.STOCHK[-1]>self.stochrsi_max) and (self.ema20[-1]-self.ema50[-1]<self.ema20[-2]-self.ema50[-2] and self.ema20[-1]>self.ema50[-1])):
            self.position.close()
            self.sell(sl=((self.data.Close*self.tkpr)/100),tp=((self.data.Close*self.stlo)/100))

def main():
    data=Fetchdata()
    bt=Backtest(data,StochasticEma,cash=10000000)
    bt.run()
    stats4=bt.optimize(
        stlo=range(96,97,1),
        tkpr=range(101,102,1),
        mom_var=range(5,21,4),
        stochrsi_tp_var=range(7,21,7),
        stochrsi_min=range(20,30,5),
        stochrsi_max=range(60,80,10),
        maximize='Sharpe Ratio'
    )
    print(stats4)
    bt.plot()

main()

# file:///C:/Users/Vaibhav/OneDrive/Documents/FolderPython/AlgorithmTrading/Official_project_01/StochasticEma_stlo-96,tkpr-101,stochrsi_tp_var-7,stochrsi_min-20,stochrsi_max-60_.html
# file:///C:/Users/Vaibhav/OneDrive/Documents/FolderPython/AlgorithmTrading/Official_project_01/StochasticEma_stlo-96,tkpr-101,stochrsi_tp_var-7,stochrsi_min-20,stochrsi_max-60_.html
# file:///C:/Users/Vaibhav/OneDrive/Documents/FolderPython/AlgorithmTrading/Official_project_01/StochasticEma_stlo-96,tkpr-101,stochrsi_tp_var-7,stochrsi_min-20,stochrsi_max-60_.html