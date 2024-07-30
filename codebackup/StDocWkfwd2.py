import pandas as pd
import numpy as np
import yfinance as yf
import talib
from backtesting import Backtest,Strategy
from backtesting.test import EURUSD,GOOG
import matplotlib.pyplot as plt
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import os


def Fetchdata():
    data = pd.read_csv(r"C:\Users\Vaibhav\OneDrive\Documents\FolderPython\AlgorithmTrading\Official_project_01\data\BTC-USDT\BTC-USDT_1d.csv")
    data.columns = ['Timestamp','Open', 'High', 'Low', 'Close', 'Volume']
    data['Date'] = pd.to_datetime(data['Timestamp'])
    data.set_index('Date', inplace=True)
    return data


class FearGreed(Strategy):
    stlo=98
    tkpr=102
    dema_var=50
    RsiMomVolume_var=14
    atr_var=14
    mom_var=14
    ema1_var=10
    ema2_var=20
    volume_var=14
    adx_var=14
    bb_var=14
    reg_var=1

    def init(self):
        self.dema = self.I(talib.DEMA,self.data.Close,timeperiod=self.dema_var)
        self.bblow,self.bbmid,self.bbhigh = self.I(talib.BBANDS,self.data.Close,timeperiod=self.bb_var)
        self.rsi = self.I(talib.RSI,self.data.Close,timeperiod=self.RsiMomVolume_var)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_var)
        self.ema1 = self.I(talib.EMA,self.data.Close,timeperiod=self.ema1_var)
        self.ema2 = self.I(talib.EMA,self.data.Close,timeperiod=self.ema2_var)
        self.volume_avg =self.I(talib.SMA,self.data.Volume.astype(float),timeperiod=self.RsiMomVolume_var)
        self.adx = self.I(talib.ADX,self.data.High,self.data.Low,self.data.Close,timeperiod=self.adx_var)
        self.mom = self.I(talib.MOM,self.data.Close,timeperiod=self.RsiMomVolume_var)
        self.Train()

    def Train(self):
        close_series = pd.Series(self.data.Close, index=self.data.index)
        self.indicator_df = pd.DataFrame(index=self.data.index)
        self.indicator_df['ADX'] = self.adx
        self.indicator_df['MOM'] = self.mom
        self.indicator_df['EMA1'] = self.ema1
        self.indicator_df['EMA2'] = self.ema2
        self.indicator_df['VolumeAvg'] = self.volume_avg
        self.indicator_df['Rsi'] = self.rsi
        self.indicator_df['returns'] = close_series.pct_change(1)
        
        self.indicator_df.dropna(inplace=True)

        split = int(len(self.indicator_df) * 0.8)
    
        x_train = self.indicator_df[['ADX', 'MOM','VolumeAvg','Rsi','EMA1','EMA2']].iloc[:split]
        y_train = self.indicator_df['returns'].iloc[:split]
        
        if(self.reg_var==1):
            reg = Pipeline([
                ('std_scaler',StandardScaler()),
                ('rad_clf',RandomForestRegressor(n_estimators=500))
            ])
        elif(self.reg_var==2):
            reg = Pipeline([
                ('std_scaler',StandardScaler()),
                ('k_means',KMeans(n_clusters=2)),
                ('k_clf',KNeighborsRegressor(n_neighbors=2))
            ])
        elif(self.reg_var==3):
            reg = Pipeline([
                ('std_scaler',StandardScaler()),
                ('xgb_clf',XGBRegressor())
            ])
        elif(self.reg_var==4):
            reg = Pipeline([
            ('std_scaler', StandardScaler()),
            ('neural_clf', MLPRegressor(
                hidden_layer_sizes=(100, 50, 25)             
            ))
        ])

        reg.fit(x_train, y_train)
        
        self.indicator_df['predict'] = reg.predict(self.indicator_df[['ADX', 'MOM', 'VolumeAvg','Rsi','EMA1','EMA2']])
        self.indicator_df["position_reg"] = np.sign(self.indicator_df["predict"])

    
    def next(self):
        if (self.indicator_df['predict'].iloc[-1]>0) and (self.adx[-1]>self.adx[-2] and self.mom[-1]>0) and (self.data.Close[-2]<self.bblow[-3]) and (self.data.Close[-1]>self.ema1[-2] and self.data.Close[-1]>self.ema2[-2]):
            self.position.close()
            self.buy(sl=((self.data.Close*self.stlo)/100),tp=((self.data.Close*self.tkpr)/100))

        elif (self.indicator_df['predict'].iloc[-1]<0) and (self.adx[-1]<self.adx[-2] and self.mom[-1]<0) and (self.data.Close[-2]>self.bbhigh[-3]) and (self.data.Close[-1]<self.ema1[-2] and self.data.Close[-1]<self.ema2[-2]):
            self.position.close()
            self.sell(sl=((self.data.Close*self.tkpr)/100),tp=((self.data.Close*self.stlo)/100))


def walk_forward(strategy, data_full, warmup_bars, lookback_bars, validation_bars, cash=10000000, commission=0):
    stats_master = []
    for i in range(lookback_bars + warmup_bars, len(data_full) - validation_bars, validation_bars):
        training_data = data_full.iloc[i - lookback_bars:i]
        validation_data = data_full.iloc[i:i + validation_bars]
        bt_training = Backtest(training_data, strategy, cash=cash, commission=commission)
        stats_training = bt_training.optimize(
            stlo=range(96,98,2),
            tkpr=range(102,104,2),
            atr_var=range(8, 16, 4),
            dema_var=range(40,60,20),
            RsiMomVolume_var=range(7,19,4),
            ema1_var=range(7,21,7),
            ema2_var=range(30,60,15),
            bb_var = range(12,16,2),
            reg_var = range(1,4,1),
            maximize='Sharpe Ratio'
        )
        opt_stlo = stats_training._strategy.stlo
        opt_tkpr = stats_training._strategy.tkpr
        opt_atr = stats_training._strategy.atr_var
        opt_dema = stats_training._strategy.dema_var
        opt_RsiMomVolume = stats_training._strategy.RsiMomVolume_var
        opt_ema1 = stats_training._strategy.ema1_var
        opt_ema2 = stats_training._strategy.ema2_var
        opt_bbvar = stats_training._strategy.bb_var
        opt_reg = stats_training._strategy.reg_var
        

        bt_validation = Backtest(validation_data, strategy, cash=cash, commission=commission)
        stats_validation = bt_validation.run(
            stlo=opt_stlo, tkpr=opt_tkpr, atr_var=opt_atr,dema_var=opt_dema,RsiMomVolume_var=opt_RsiMomVolume,ema1_var=opt_ema1,ema2_var=opt_ema2,bb_var=opt_bbvar,reg_var=opt_reg
        )
        equity_curve = stats_validation['_equity_curve']['Equity']
        a=validation_data['Close']
        b=validation_data['Close'].iloc[0]
        c=a/b
        plt.figure(figsize=(14, 7))
        plt.plot(equity_curve, label='Strategy Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Equity')
        plt.title('Equity Curve')
        plt.legend()
        graph_path = os.path.join('graph',f'equity_curve{i}_2.png')
        plt.savefig(graph_path)

        plt.figure(figsize=(14,7))
        plt.plot(c,label='Increase Curve')
        plt.xlabel('Time')
        plt.ylabel('Equity')
        plt.title('Equity Curve')
        plt.legend()

        graph_path2 = os.path.join('graph',f'equity_normal_curve{i}_2.png')
        plt.savefig(graph_path2)
        plt.close()

        stats_master.append(stats_validation)
        var2=pd.DataFrame(stats_master)
        csv_path=os.path.join('csv','file2.csv')
        var2.to_csv(csv_path)

    return stats_master

lookback_bars = 1600
validation_bars = 400 
warmup_bars = 60


data = Fetchdata()
stats = walk_forward(FearGreed, data, warmup_bars, lookback_bars, validation_bars)
for stat in stats:
    print(stat)
