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
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
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

class AdxMomentum(Strategy):
    stlo=98
    tkpr=102
    volume_var=14
    adx_var=14
    rsi_var=7
    mom_var=14
    sma_var=7

    def init(self):
        self.sma7 = self.I(talib.SMA,self.data.Close,timeperiod=self.sma_var)
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
        if ((self.indicator_df['predict'].iloc[-1]>0 and (self.adx[-1]-self.adx[-11]>0 and self.mom[-1]>0) and (self.sma7[-1]-self.sma7[-11]>0 and self.mom[-1]>0) and (self.adx[-1]-self.adx[-11]>0 and self.sma7[-1]-self.sma7[-11]>0))):
            self.position.close()
            self.buy(sl=((self.data.Close*self.stlo)/100),tp=((self.data.Close*self.tkpr)/100))

        elif ((self.indicator_df['predict'].iloc[-1]<0 and (self.adx[-1]-self.adx[-11]<0 and self.mom[-1]<0) and (self.sma7[-1]-self.sma7[-11]<0 and self.mom[-1]<0) and (self.adx[-1]-self.adx[-11]<0 and self.sma7[-1]-self.sma7[-11]<0))):
            self.position.close()
            self.sell(sl=((self.data.Close*self.tkpr)/100),tp=((self.data.Close*self.stlo)/100))

def main():
    data=Fetchdata()
    bt=Backtest(data,AdxMomentum,cash=10000000)
    bt.run()
    stats6=bt.optimize(
        stlo=range(96,99,1),
        tkpr=range(101,104,1),
        rsi_var=range(5,21,4),
        mom_var=range(5,21,4),
        adx_var=range(5,21,4),
        sma_var=range(5,21,4),
        maximize='Sharpe Ratio'
    )
    print(stats6)
    bt.plot()

main()

# file:///C:/Users/Vaibhav/OneDrive/Documents/FolderPython/AlgorithmTrading/Official_project_01/AdxMomentum_stlo-96,tkpr-101,RsiMomVolumeAdx_var-5_.html