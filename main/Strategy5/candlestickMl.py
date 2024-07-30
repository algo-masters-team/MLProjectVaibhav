# here i will put the code for candlestick pattern identifier
# here i will write the code for supertrend and bbands
# here there will be the research paper strategy
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

class CandlestickReversal(Strategy):
    stlo=98
    tkpr=102
    volume_var=14
    adx_var=14
    mom_var=14
    rsi_var=7
    ema_var=60

    def init(self):
        self.EMA = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_var)
        self.Hammer = self.I(talib.CDLHAMMER, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.MorningStar = self.I(talib.CDLMORNINGSTAR, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.EveningStar = self.I(talib.CDLEVENINGSTAR, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.ShootingStar = self.I(talib.CDLSHOOTINGSTAR, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.marubozu = self.I(talib.CDLMARUBOZU,self.data.Open,self.data.High,self.data.Low,self.data.Close)
        self.InvertedHammer = self.I(talib.CDLMARUBOZU,self.data.Open,self.data.High,self.data.Low,self.data.Close)  
        self.White = self.I(talib.CDL3WHITESOLDIERS,self.data.Open,self.data.High,self.data.Low,self.data.Close)
        self.Black = self.I(talib.CDL3BLACKCROWS,self.data.Open,self.data.High,self.data.Low,self.data.Close)

        self.MACD, self.MACDSignal, _ = self.I(talib.MACD, self.data.Close, fastperiod=12, slowperiod=26)
        self.volume_avg =self.I(talib.SMA,self.data.Volume.astype(float),timeperiod=self.volume_var) # used in ml
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
        if ((self.indicator_df['predict'].iloc[-1]>0) and (self.EMA[-1]>self.data.Close[-1] and self.MACD[-1]>self.MACDSignal[-1]) and (self.Hammer[-2] != 0 or self.MorningStar[-2] != 0 or self.marubozu[-2]==100 or self.White[-2]!=0)):
            self.position.close()
            self.buy(sl=((self.data.Close*self.stlo)/100),tp=((self.data.Close*self.tkpr)/100))

        elif ((self.indicator_df['predict'].iloc[-1]<0) and (self.EMA[-1]<self.data.Close[-1] and self.MACD[-1]<self.MACDSignal[-1]) and (self.InvertedHammer[-2] != 0 or self.EveningStar[-2] != 0 or self.marubozu[-2]==-100 or self.Black[-2]!=0)):
            self.position.close()
            self.sell(sl=((self.data.Close*self.tkpr)/100),tp=((self.data.Close*self.stlo)/100))

def main():
    data=Fetchdata()
    bt=Backtest(data,CandlestickReversal,cash=10000000)
    bt.run()
    stats7=bt.optimize(
        stlo=range(96,97,1),
        tkpr=range(103,104,1),
        ema_var=range(40,120,20),
        mom_var=range(5,23,4),
        adx_var=range(5,23,4),
        maximize='Equity Final [$]'
    )
    print(stats7)
    bt.plot()

main()

# file:///C:/Users/Vaibhav/OneDrive/Documents/FolderPython/AlgorithmTrading/Official_project_01/CandlestickReversal_stlo-96,tkpr-103,ema_var-80,RsiMomVolumeAdx_var-5_.html
# file:///C:/Users/Vaibhav/OneDrive/Documents/FolderPython/AlgorithmTrading/Official_project_01/CandlestickReversal_stlo-96,tkpr-103,ema_var-40,RsiMomVolumeAdx_var-5_.html