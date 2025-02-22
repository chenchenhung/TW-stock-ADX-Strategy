# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 00:56:41 2025

@author: RonaldChen_moneymaker
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ----------------------------
# 1. 直接從網路下載台股指數的歷史資料 (2000/01/01 - 2025/01/01)
# ----------------------------
df = pd.read_csv("taiwan_stock_data.csv")

# 2. ADX 指標計算函數 (採用14日參數)
# ----------------------------
def compute_adx(data, n=14):
    df = data.copy()
    # 計算 True Range
    df['H-L']   = df['High'] - df['Low']
    df['H-PC']  = abs(df['High'] - df['Close'].shift(1))
    df['L-PC']  = abs(df['Low'] - df['Close'].shift(1))
    df['TR']    = df[['H-L','H-PC','L-PC']].max(axis=1)
    
    # 計算 Directional Movements
    df['UpMove']   = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    
    # 平滑計算 (用簡單累加法)
    df['TR_s']   = df['TR'].rolling(window=n).sum()
    df['+DM_s']  = df['+DM'].rolling(window=n).sum()
    df['-DM_s']  = df['-DM'].rolling(window=n).sum()
    
    # 計算 +DI 與 -DI
    df['+DI'] = 100 * (df['+DM_s'] / df['TR_s'])
    df['-DI'] = 100 * (df['-DM_s'] / df['TR_s'])
    
    # 計算 DX 與 ADX
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].rolling(window=n).mean()
    
    # 刪除中間計算欄位
    df.drop(['H-L','H-PC','L-PC','TR','UpMove','DownMove','+DM','-DM','TR_s','+DM_s','-DM_s','+DI','-DI','DX'], axis=1, inplace=True)
    return df

df = compute_adx(df, n=14)

# ----------------------------
# 3. 趨勢市場策略：均線交叉 (適用於 ADX > 25)
# ----------------------------
df['MA_short'] = df['Close'].rolling(window=20).mean()  # 短期均線
df['MA_long']  = df['Close'].rolling(window=50).mean()   # 長期均線
# 趨勢策略訊號：當短期均線 > 長期均線則視為多頭 (+1)，反之空頭 (-1)
df['trend_signal'] = np.where(df['MA_short'] > df['MA_long'], 1, -1)

# ----------------------------
# 4. 震盪市場策略：布林通道均值回歸 (適用於 ADX < 20)
# ----------------------------
window_bb = 20
df['BB_middle'] = df['Close'].rolling(window=window_bb).mean()
df['BB_std']    = df['Close'].rolling(window=window_bb).std()
df['BB_upper']  = df['BB_middle'] + 2 * df['BB_std']
df['BB_lower']  = df['BB_middle'] - 2 * df['BB_std']

# 初步布林通道訊號：若收盤價 <= 下軌，則買多 (+1)；若收盤價 >= 上軌，則放空 (-1)
df['bollinger_signal'] = 0
df.loc[df['Close'] <= df['BB_lower'], 'bollinger_signal'] = 1
df.loc[df['Close'] >= df['BB_upper'], 'bollinger_signal'] = -1

# 利用布林中軌作為退出點
bollinger_position = []
position = 0  # 初始無持倉
for idx, row in df.iterrows():
    signal = row['bollinger_signal']
    if position == 0:
        if signal != 0:
            position = signal
    else:
        # 若持多頭，當價格回升至中軌或以上則平倉
        if position == 1 and row['Close'] >= row['BB_middle']:
            position = 0
        # 若持空頭，當價格回落至中軌或以下則平倉
        elif position == -1 and row['Close'] <= row['BB_middle']:
            position = 0
        # 若有新信號且方向相反，則轉換部位
        elif signal != 0 and signal != position:
            position = signal
    bollinger_position.append(position)
df['bollinger_position'] = bollinger_position

# ----------------------------
# 5. 根據 ADX 決定採用哪套策略：
#    ADX > 25：採用 trend_signal
#    ADX < 20：採用 bollinger_position
#    ADX 介於 20 與 25：採用前一日持倉 (ffill)
# ----------------------------
df['final_signal'] = np.nan
df.loc[df['ADX'] > 25, 'final_signal'] = df.loc[df['ADX'] > 25, 'trend_signal']
df.loc[df['ADX'] < 20, 'final_signal'] = df.loc[df['ADX'] < 20, 'bollinger_position']

df['final_signal'].ffill(inplace=True)
df['final_signal'].fillna(0, inplace=True)
df['final_position'] = df['final_signal']  # 設定最終部位 (可正可負)

# ----------------------------
# 6. 策略績效模擬 (多空皆計算)
# ----------------------------
df['daily_return'] = df['Close'].pct_change()
# 使用 Series.mul() 並填入缺失值0，確保對齊運算
df['strategy_return'] = df['final_position'].shift(1).mul(df['daily_return'], fill_value=0)
df['cumulative_return'] = (1 + df['strategy_return']).cumprod() - 1

# ----------------------------
# 輸出部分策略訊號與績效資料
# ----------------------------
print(df[['Close', 'ADX', 'MA_short', 'MA_long', 'trend_signal', 
          'BB_upper', 'BB_middle', 'BB_lower', 'bollinger_position', 
          'final_position', 'strategy_return', 'cumulative_return']].tail(20))

# ----------------------------
# 7. 繪圖顯示
# ----------------------------
plt.figure(figsize=(14,16))

# 上半部：價格走勢與策略部位
ax1 = plt.subplot(4,1,1)
plt.title('TW stock ADX methods')
plt.plot(df.index, df['Close'], label='Close Price', color='black', alpha=0.7)
plt.plot(df.index, df['MA_short'], label='MA Short (20)', color='blue', alpha=0.6)
plt.plot(df.index, df['MA_long'], label='MA Long (50)', color='red', alpha=0.6)
plt.plot(df.index, df['BB_upper'], label='BB Upper', linestyle='--', color='magenta', alpha=0.6)
plt.plot(df.index, df['BB_middle'], label='BB Middle', linestyle='--', color='grey', alpha=0.6)
plt.plot(df.index, df['BB_lower'], label='BB Lower', linestyle='--', color='green', alpha=0.6)
plt.ylabel('Price')
plt.legend()
plt.grid(True)
# 標示多空訊號
ax1 = plt.subplot(4,1,2)
buy_signals = df[df['final_position'] == 1]
sell_signals = df[df['final_position'] == -1]
plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy', s=80)
plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell', s=80)
plt.ylabel('Price')
plt.legend()
plt.grid(True)


# 下半部：累積策略報酬率
ax2 = plt.subplot(4,1,3)
plt.plot(df.index, df['cumulative_return']*100, label='Cumulative Return', color='blue')
plt.title('策略累積報酬率(%)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
