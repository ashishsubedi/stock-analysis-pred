# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from IPython.display import display


# %%
plt.style.use('fivethirtyeight')


# %%



# %%

datasets = ['nic.csv']
data = datasets[0]


# %%
df = pd.read_csv(data,index_col='Date')
df.head()


# %%
df.describe()


# %%
def EMA(df, period = 12,column='Close Price'):
    return df[column].ewm(span=period,adjust=False).mean()


# %%
def MACD(df, low_period=12,high_period=26,signal_period=9,column='Close Price'):
    EMA_low = EMA(df,period=low_period,column=column)
    EMA_high = EMA(df,period=high_period,column=column)
    MACD_line= EMA_low - EMA_high
    signal_line = MACD_line.ewm(span=signal_period,adjust=False).mean()
    df['MACD'] = MACD_line
    df['Signal'] = signal_line
    return MACD_line,signal_line





# %%



# %%
def plot_MACD_signal(macd,signal,df=df,use_plotly=True,period_text='(12,26,9)'):
    if use_plotly:
        fig = go.Figure()
        fig = fig.add_trace(go.Scatter(x=df.index,y=macd,name='MACD',line=dict(color="green"), opacity=0.7))
        fig = fig.add_trace(go.Scatter(x=df.index,y=signal,name='Signal Line',line=dict(color="red"), opacity=0.7))
        fig.update_layout(
            title={
                'text': "MACD and Signal of ->"+ data+ period_text},
            legend_title="Legend",
            xaxis_title='Date',
            yaxis_title='MACD Value',
           
        )
        

        fig.show()
        
    else:
        plt.figure(figsize=(10,5))
        plt.plot(df.index,macd,label='MACD',color='green',alpha=0.7)
        plt.plot(df.index,signal,color='red',label='Signal Line',alpha=0.7)
        plt.xticks(rotation=90,fontsize=6)
        plt.title("MACD and Signal of "+data)
        plt.legend(loc=0)
        plt.show()


# %%
def buy_sell(df,close='Close Price'):
    buy = []
    sell = []
    flag = -1

    for i in range(len(df)):
        if df['MACD'][i]>df['Signal'][i] and flag != 1:
            #MACD crosses signal line from bottom to top
            flag = 1
            buy.append(df[close][i])
            sell.append(np.nan)
        elif df['MACD'][i] < df['Signal'][i] and flag!= 0:
            #MACD crosses from top to below
            flag=0
            buy.append(np.nan)
            sell.append(df[close][i])
        else:
            buy.append(np.nan)
            sell.append(np.nan)
            
    df['Buy'] = buy
    df['Sell'] = sell

    return (buy,sell)


# %%
def plot_close_price(df,column='Close Price',plot_EMA=True,show_buy_sell=True,use_plotly=True,period_text='(12,26,9)'):
    if plot_EMA:
        ema10= EMA(df,period=10,column=column)
        ema100= EMA(df,period=100,column=column)
    if show_buy_sell:
        buy,sell = buy_sell(df)

    if use_plotly:

        fig = go.Figure()
        fig = fig.add_trace(go.Scatter(x=df.index,y=df[column],name='Close Price',opacity=0.7))
        if plot_EMA:
            fig = fig.add_trace(go.Scatter(x=df.index,y=ema10,name='EMA10',opacity=0.7))
            fig = fig.add_trace(go.Scatter(x=df.index,y=ema100,name='EMA100',opacity=0.7))
        if show_buy_sell:
            fig = fig.add_trace(go.Scatter(mode='markers',x=df.index,y=df['Buy'],name='Buy',marker=dict(
                color='green',
                size=10,
                symbol='triangle-up'
                
            ),
            opacity=0.7))

            fig = fig.add_trace(go.Scatter(mode='markers',x=df.index,y=df['Sell'],name='Sell',marker=dict(
                color='red',
                size=10,
                symbol='triangle-down'
                
            ),
             opacity=0.7))
        fig.update_layout(
            title={
                'text': "Close Price History ->"+ data+period_text},
            xaxis_title='Date',
            yaxis_title='Close Price'
        )
        fig.show()
    else:
        mpl_fig = plt.figure(figsize=(10,5))
        plt.plot(df.index,df[column],label='Close Price',color='blue',alpha=0.7)
        if plot_EMA:
            plt.plot(df.index,ema10,label='EMA10',color='green',alpha=0.7)
            plt.plot(df.index,ema100,label='EMA100',color='cyan',alpha=0.7)
        if show_buy_sell:
            plt.plot(df.index,df['Buy'],label='Buy',marker='^',alpha=1)
            plt.plot(df.index,df['Sell'],label='Sell',marker='^',alpha=1)
        plt.xticks(rotation=80,fontsize=5)
        plt.title("Close Price History "+ data)
        plt.legend(loc=0)
        plt.show()


# %%
macd,signal = MACD(df)


# %%
plot_close_price(df,use_plotly=True,plot_EMA=True)
plot_MACD_signal(macd,signal,df)


# %%

period = (5,35,5)

macd2,signal2 = MACD(df,*period)
plot_close_price(df,use_plotly=True,plot_EMA=True,period_text=str(period))
plot_MACD_signal(macd2,signal2,df,period_text=str(period))


# %%



