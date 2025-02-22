Trading Strategy Using ADX, Moving Average Crossover, and Bollinger Bands Mean Reversion

Overview
This repository contains a Python implementation of a trading strategy that dynamically selects a trading approach based on market conditions as indicated by the Average Directional Index (ADX). The strategy employs:
Trend-Following (Moving Average Crossover): When the ADX is greater than 25, indicating a strong trend, a moving average crossover strategy is used. A 20-day short-term moving average is compared with a 50-day long-term moving average to generate trading signals.
Mean Reversion (Bollinger Bands): When the ADX is less than 20, suggesting a sideways or ranging market, a Bollinger Bands mean reversion strategy is applied. The strategy initiates a long position when the price nears the lower Bollinger Band and a short position when it nears the upper band. The middle band acts as the exit point.
The strategy supports both long and short positions, calculates daily returns, and computes cumulative returns over a 25-year period.

Features
Data : TWstock index from Yahoo Finance using yfinance.
Technical Indicator Calculations: Computes ADX, moving averages, and Bollinger Bands.
Dynamic Strategy Selection: Switches between trend-following and mean reversion strategies based on ADX values.
Performance Simulation: Calculates daily strategy returns and cumulative returns.
Visualization: Generates plots showing price movements, technical indicators, trade signals, and cumulative returns.

Requirements:
Python 3.x
pandas
numpy
matplotlib
yfinance
