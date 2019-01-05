import numpy as np


def BayesConfidenceInterval(data, alpha=0.05):
    '''
    Input
        data  : List of values
        alpha : bottom percentage of interval.
    Return
        give three values
        1. mean of data
        2. bottom value of interval.
           If aplha=0.05, function returens 5% from bottom.
        3. upper value of interval.
           If aplha=0.05, function returens 95% from bottom.
    '''
    MeanValue = np.mean(data)
    alpha = alpha*100
    BottomValue, UpperValue = np.percentile(data, [alpha, 100-alpha])
    return MeanValue, BottomValue, UpperValue
