import os
import sys
import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import optuna
import joblib

# Get current directory and append data path
current_dir = Path.cwd()
symbol='ETH-USDT'
data_folder_path = current_dir / 'data' / symbol

# Add data folder path to sys.path
sys.path.append(str(data_folder_path))

# Function to load and process data
def load_data(file_name):
    file_path = data_folder_path / file_name
    data = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    return data

# Function to calculate features
def calculate_features(data):
    data['EMA_100'] = ta.trend.ema_indicator(data['close'], window=100)
    data['EMA_200'] = ta.trend.ema_indicator(data['close'], window=200)
    data['EMA_10'] = ta.trend.ema_indicator(data['close'], window=10)
    data['EMA_20'] = ta.trend.ema_indicator(data['close'], window=20)
    data['RSI'] = ta.momentum.rsi(data['close'])
    data['MACD'] = ta.trend.macd_diff(data['close'])
    data['Pivot_High'] = ta.volatility.bollinger_hband_indicator(data['close'])
    data['Pivot_Low'] = ta.volatility.bollinger_lband_indicator(data['close'])
    data['Ichimoku'] = ta.trend.ichimoku_a(data['high'], data['low'])
    data['VWAP'] = ta.volume.volume_weighted_average_price(data['high'], data['low'], data['close'], data['volume'])
    data['Super_Trend'] = ta.trend.stc(data['close'])
    data['ATR'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'])
    data['ADX'] = ta.trend.adx(data['high'], data['low'], data['close'])
    data['Return'] = data['close'].pct_change()
    data['Log_Return'] = np.log(data['close']).diff()
    data['Price_Roll_Mean'] = data['close'].rolling(window=10).mean()
    data['Price_Roll_Std'] = data['close'].rolling(window=10).std()
    data['Volume_Roll_Mean'] = data['volume'].rolling(window=10).mean()
    data['Volume_Roll_Std'] = data['volume'].rolling(window=10).std()
    data.dropna(inplace=True)
    return data

# Backtesting function
def backtest(predictions, ohlc, initial_balance=1000):
    balance = initial_balance
    holdings = 0
    equity_curve = []
    trades = 0
    wins = 0
    losses = 0
    returns = []
    daily_returns = []

    for i in range(len(predictions)):
        if predictions[i] == 1:  # Buy signal
            if holdings == 0:  # If not holding any position, buy
                holdings = balance / ohlc['close'].iloc[i]
                balance = 0
        elif predictions[i] == 0 and holdings > 0:  # Sell signal
            balance = holdings * ohlc['close'].iloc[i]
            holdings = 0
            trades += 1
            if balance > initial_balance:
                wins += 1
            else:
                losses += 1
        equity_curve.append(balance + holdings * ohlc['close'].iloc[i])

        # Calculate daily returns
        if i == 0:
            daily_returns.append(0)
        else:
            daily_returns.append((equity_curve[-1] / equity_curve[-2]) - 1)
    
    equity_curve = np.array(equity_curve)
    returns = np.array(daily_returns)

    return balance, equity_curve, trades, wins, losses, returns, daily_returns

# Function to calculate additional metrics
def calculate_metrics(returns):
    returns = np.array(returns)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    sharpe_ratio = mean_return / std_return * np.sqrt(252)
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns)
    sortino_ratio = mean_return / downside_std * np.sqrt(252)
    return sharpe_ratio, sortino_ratio

# Load data
time = '4h'
data = load_data(f'{symbol}_{time}.csv')

# Calculate features
data = calculate_features(data)

# Feature and target separation
X = data[['EMA_100', 'EMA_200', 'EMA_10', 'EMA_20', 'RSI', 'MACD', 'Pivot_High', 'Pivot_Low', 'Ichimoku', 'VWAP', 'Super_Trend', 'ATR', 'ADX', 'Return', 'Log_Return', 'Price_Roll_Mean', 'Price_Roll_Std', 'Volume_Roll_Mean', 'Volume_Roll_Std']]
# X = data[['EMA_100', 'EMA_10', 'EMA_20', 'RSI', 'VWAP','ADX','Pivot_High','Pivot_Low','Volume_Roll_Mean','Volume_Roll_Std','Return']]
y_classification = np.where(data['close'].shift(-1) > data['close'], 1, 0)  # For classification
y_regression = data['close'].shift(-1)  # For regression

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Manual train-test split (keep data continuous)
split_index = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train_classification, y_test_classification = y_classification[:split_index], y_classification[split_index:]
y_train_regression, y_test_regression = y_regression[:split_index], y_regression[split_index:]
ohlc_test = data.iloc[split_index:split_index + len(X_test)]

# Function to optimize hyperparameters
def objective(trial):
    classifier_name = trial.suggest_categorical('classifier', ['RandomForest', 'GradientBoosting', 'XGBoost','KNeighbours'])
    if classifier_name == 'RandomForest':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 2, 32)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    elif classifier_name == 'GradientBoosting':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
        max_depth = trial.suggest_int('max_depth', 2, 32)
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    elif classifier_name == 'SVM':
        C = trial.suggest_float('C', 0.1, 10.0)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        if kernel == 'poly' or kernel == 'rbf' or kernel == 'sigmoid':
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            clf = SVC(C=C, kernel=kernel, gamma=gamma)
    elif classifier_name == 'KNeighbours':
        n_neighbors = trial.suggest_int('n_neighbors', 3, 20)
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
        max_depth = trial.suggest_int('max_depth', 2, 32)
        clf = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)

    clf.fit(X_train, y_train_classification)
    y_pred = clf.predict(X_test)
    return (y_pred == y_test_classification).mean()

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
best_params = study.best_params

# Train the best model
if best_params['classifier'] == 'RandomForest':
    best_model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'])
elif best_params['classifier'] == 'GradientBoosting':
    best_model = GradientBoostingClassifier(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'], max_depth=best_params['max_depth'])
elif best_params['classifier'] == 'SVM':
    if best_params['kernel'] == 'poly' or best_params['kernel'] == 'rbf' or best_params['kernel'] == 'sigmoid':
        best_model = SVC(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'])
    else:
        best_model = SVC(C=best_params['C'], kernel=best_params['kernel'])
elif best_params['classifier'] == 'KNeighbours':
    best_model = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'])
else:
    best_model = XGBClassifier(n_estimators=best_params['n_estimators'], learning_rate=best_params['learning_rate'], max_depth=best_params['max_depth'])

best_model.fit(X_train, y_train_classification)
y_pred = best_model.predict(X_test)

# Backtest and evaluate the best model
final_balance, equity_curve, trades, wins, losses, returns, daily_returns = backtest(y_pred, ohlc_test)
sharpe_ratio, sortino_ratio = calculate_metrics(daily_returns)
winrate = wins / trades if trades > 0 else 0
cumulative_returns = np.cumprod(1 + np.array(daily_returns)) - 1
asset_returns = ohlc_test['close'].pct_change().fillna(0)
cumulative_asset_returns = np.cumprod(1 + asset_returns.values) - 1

# Display results
print(f"Best Model: {best_params['classifier']}")
print(f"Classification Report:\n")
print(classification_report(y_test_classification, y_pred))
print(f"Hitrate: {(y_pred == y_test_classification).mean():.2f}")
print(f"Winrate: {winrate:.2f}")
print(f"Number of Trades: {trades}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Sortino Ratio: {sortino_ratio:.2f}")
print(f"Final Balance: ${final_balance:.2f}")

# Plot equity curves with asset price
plt.figure(figsize=(14, 7))
plt.plot(cumulative_asset_returns, label='Buy and Hold', color='black', linestyle='--')
plt.plot(cumulative_returns, label='Equity Curve', color='blue')
plt.title('Equity Curve with Asset Price')
plt.xlabel('Time')
plt.ylabel('Cumulative Returns')
plt.legend()
graph_path=os.path.join('graph',f'equity_curve_{symbol}_{time}.png')
plt.savefig(graph_path)
plt.close()
# plt.show()

# Export results
results = {
    'Best Model': best_params['classifier'],
    'Hitrate': (y_pred == y_test_classification).mean(),
    'Winrate': winrate,
    'Number of Trades': trades,
    'Sharpe Ratio': sharpe_ratio,
    'Sortino Ratio': sortino_ratio,
    'Final Balance': final_balance
}
results_df = pd.DataFrame([results])
results_df.to_csv(current_dir /'csv'/ f'backtest_results_{symbol}_{time}.csv', index=False)
print("Results exported to 'backtest_results.csv'")

# save the model and equity curve and run on all timeframe of 3 stocks and take an entry and take position if all three timeframe is good.
path = os.path.join('Joblib',f'model_{symbol}_{time}.joblib')
joblib.dump(best_model,path)
print("model saved successfully in the joblib file")



# Now you can use the loaded_model to make predictions
# loaded_model = joblib.load('model_name_here_from_joblib_folder.joblib')
# predictions = loaded_model.predict(X_test)
# print(predictions)

