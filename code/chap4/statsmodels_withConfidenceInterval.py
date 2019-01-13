#!/usr/bin/env python
# coding: utf-8

import statsmodels.api as sm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

df = pd.read_csv("../../RStanBook/chap04/input/data-salary.txt")


sns.scatterplot(data=df, x=df.X, y=df.Y)

result = sm.OLS(df.Y, sm.add_constant(df.X)).fit()

result95 = result.get_prediction().summary_frame(alpha=0.05)
result50 = result.get_prediction().summary_frame(alpha=0.5)


sns.scatterplot(data=df, x=df.X, y=df.Y)
plt.plot(df.X, result95.loc[:,"mean"], 'r')
plt.fill_between(df.X, result95.loc[:,"mean_ci_upper"], result95.loc[:,"mean_ci_lower"], facecolor='r',alpha=0.2, label="95%")
plt.fill_between(df.X, result50.loc[:,"mean_ci_upper"], result50.loc[:,"mean_ci_lower"], facecolor='r',alpha=0.5, label="50%")
plt.legend()
plt.savefig("../../output/chap4/CI.png")
plt.show()

sns.scatterplot(data=df, x=df.X, y=df.Y)
plt.plot(df.X, result95.loc[:,"mean"], 'r')
plt.fill_between(df.X, result95.loc[:,"obs_ci_upper"], result95.loc[:,"obs_ci_lower"], facecolor='r',alpha=0.2, label="95%")
plt.fill_between(df.X, result50.loc[:,"obs_ci_upper"], result50.loc[:,"obs_ci_lower"], facecolor='r',alpha=0.5, label="50%")
plt.legend()
plt.savefig("../../output/chap4/PI.png")
plt.show()
