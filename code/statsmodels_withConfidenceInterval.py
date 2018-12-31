#!/usr/bin/env python
# coding: utf-8

import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

df = pd.read_csv("../RStanBook/chap04/input/data-salary.txt")


sns.scatterplot(data=df, x=df.X, y=df.Y)
result = sm.OLS(df.Y, sm.add_constant(df.X)).fit()
_, u, b = wls_prediction_std(result)
_, u50, b50 = wls_prediction_std(result, alpha=0.5)


result.summary()


sns.scatterplot(data=df, x=df.X, y=df.Y)
plt.plot(df.X, result.predict(), 'r')
plt.plot(df.X, u, 'r--')
plt.plot(df.X, b, 'r--')
plt.plot(df.X, u50, 'r-.')
plt.plot(df.X, b50, 'r-.')
