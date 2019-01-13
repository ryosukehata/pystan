#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import pystan
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv("../../RStanBook/chap04/input/data-salary.txt"")


# In[ ]:


stanmodel='''
data {
    int N;
    real X[N];
    real Y[N];
}

parameters {
    real a;
    real b;
    real<lower=0> sigma;
} 

model {
    for (n in 1:N){
    Y[n] ~ normal(a + b*X[n], sigma);
    }
}
'''


# In[ ]:


stan_input = { 'N':df.X.size, 'X':df.X, 'Y':df.Y}


# In[ ]:


sm = pystan.StanModel(model_code=stanmodel)


# In[ ]:


fit = sm.sampling(data=stan_input , iter=1000, chains=50)


# In[ ]:


print(fit.stansummary())


# In[ ]:


a_coef = fit.extract("a")["a"]
b_coef = fit.extract("b")["b"]
sigma_coef = fit.extract("sigma")["sigma"]


# In[ ]:


x=np.linspace(df.X.min(),df.X.max()+1)


# In[ ]:


df_stan = fit.to_dataframe()


# In[ ]:


df_stan=df_stan[["a","b","sigma"]]


# In[ ]:


sns.jointplot(data=df_stan, x="a", y="b")


# In[ ]:


def BayesConfidenceInterval(data, alpha=0.05):
    MeanValue = np.mean(data)
    alpha = alpha*100
    BottomValue, UpperValue = np.percentile(data, [alpha, 100-alpha])
    return MeanValue, BottomValue, UpperValue

def BayesPredictionInterval(data, sigma, alpha=0.05):
    DataSize = data.size
    newProbabilityDistribution=np.random.normal(loc=data, scale=sigma, size=DataSize)
    MeanValue, BottomValue, UpperValue = BayesConfidenceInterval(newProbabilityDistribution, alpha=alpha)
    return MeanValue, BottomValue, UpperValue


# In[ ]:


mean_y=[]
upper_CI_y=[]
bottom_CI_y=[]
upper_CI50_y=[]
bottom_CI50_y=[]
upper_PI_y=[]
bottom_PI_y=[]
upper_PI50_y=[]
bottom_PI50_y=[]
for i in x:
    df_stan.loc[:,"CI"] = df_stan.a + df_stan.b * i
    mean, bottomCI, upperCI = BayesConfidenceInterval(df_stan.CI)
    mean_y.append(mean)
    bottom_CI_y.append(bottomCI)
    upper_CI_y.append(upperCI)

    mean, bottomCI, upperCI = BayesConfidenceInterval(df_stan.CI, alpha=0.25)
    bottom_CI50_y.append(bottomCI)
    upper_CI50_y.append(upperCI)
    
    _, bottomPI, upperPI = BayesPredictionInterval(df_stan.CI, df_stan.sigma)
    
    bottom_PI_y.append(bottomPI)
    upper_PI_y.append(upperPI)
    
    _, bottomPI, upperPI = BayesPredictionInterval(df_stan.CI, df_stan.sigma, alpha=0.25)
    
    bottom_PI50_y.append(bottomPI)
    upper_PI50_y.append(upperPI)
    


# In[ ]:


sns.scatterplot(data=df, x="X", y="Y")
plt.plot(x, mean_y, "g")
plt.fill_between(x, upper_CI_y, bottom_CI_y, facecolor='r',alpha=0.2, label="95%")
plt.fill_between(x, upper_CI50_y, bottom_CI50_y, facecolor='r',alpha=0.4, label="50%")
plt.legend(loc="best")
plt.savefig("BayesConfidenceInterval.png")


# In[ ]:


sns.scatterplot(data=df, x="X", y="Y")
plt.plot(x, mean_y, "g")
plt.fill_between(x, upper_PI_y, bottom_PI_y, facecolor='r',alpha=0.2, label="95%")
plt.fill_between(x, upper_PI50_y, bottom_PI50_y, facecolor='r',alpha=0.4, label="50%")
plt.legend(loc="best")
plt.savefig("BayesPredictionInterval.png")


# In[ ]:




