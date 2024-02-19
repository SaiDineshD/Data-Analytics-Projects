#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# List of technology sector companies
tech_companies = ['AAPL', 'MSFT', 'GOOGL']

# Fetch data
tech_data = yf.download(tech_companies, start="2022-01-01", end="2024-01-01")['Adj Close']


# In[3]:


#Caluclate daily reurns
daily_returns=tech_data.pct_change()
#Calculate volatility
volatility=daily_returns.std()


# In[4]:


#Caluclate cumulative returns
cumulative_returns=(1+daily_returns).cumprod()


# In[5]:


# Plot cumulative returns
cumulative_returns.plot(figsize=(10, 6))
plt.title('Cumulative Returns of Technology Sector Stocks')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid(True)
plt.legend()
plt.show()

# Plot volatility
volatility.plot(kind='bar', figsize=(10, 6))
plt.title('Volatility of Technology Sector Stocks')
plt.xlabel('Stock')
plt.ylabel('Volatility')
plt.grid(axis='y')
plt.show()


# In[6]:


# Download historical data for the benchmark index (S&P 500)
benchmark_data = yf.download('^GSPC', start="2022-01-01", end="2024-01-01")['Adj Close']

# Calculate cumulative returns for sector and benchmark index
cumulative_returns_sector = (1 + daily_returns.mean(axis=1)).cumprod()
cumulative_returns_benchmark = (1 + benchmark_data.pct_change()).cumprod()

# Plot cumulative returns comparison
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns_sector, label='Technology Sector')
plt.plot(cumulative_returns_benchmark, label='S&P 500')
plt.title('Cumulative Returns Comparison')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.grid(True)
plt.show()


# In[7]:


# Calculate Sharpe ratio
risk_free_rate = 0.02  # Assume a risk-free rate of 2%
sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std()

# Calculate Sortino ratio
downside_returns = daily_returns[daily_returns < 0]
downside_volatility = downside_returns.std()
sortino_ratio = (daily_returns.mean() - risk_free_rate) / downside_volatility

print("Sharpe Ratio:")
print(sharpe_ratio)
print("\nSortino Ratio:")
print(sortino_ratio)


# In[8]:


import seaborn as sns
correlation_matrix = daily_returns.corr()
# Plot correlation matrix heatmap
plt.figure(figsize=(10, 6))
plt.title('Correlation Matrix of Technology Sector Stocks')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()


# In[9]:


import plotly.graph_objects as go

# Interactive plot using Plotly
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=daily_returns.index, y=daily_returns['AAPL'], mode='lines', name='AAPL'))
fig.add_trace(go.Scatter(x=daily_returns.index, y=daily_returns['MSFT'], mode='lines', name='MSFT'))
fig.add_trace(go.Scatter(x=daily_returns.index, y=daily_returns['GOOGL'], mode='lines', name='GOOGL'))

# Add layout options
fig.update_layout(title='Daily Returns of Technology Sector Stocks',
                  xaxis_title='Date',
                  yaxis_title='Daily Returns',
                  xaxis_rangeslider_visible=True)

# Show plot
fig.show()

