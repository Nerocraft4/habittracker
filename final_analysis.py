import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pmdarima import arima

N=332 #number of samples
STEST = 2/np.sqrt(N)
TTEST = 1.96

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def significance_pearson(val):
    if np.abs(val)<STEST:
        return True
    return False

def significance_spearman(val):
    if val==1:
        return True
    t = val * np.sqrt((N-2)/(1-val*val))    
    if np.abs(t)<1.96:
        return True
    return False

def combine_masks(df,triangle):
    N = df.shape[0]
    M = df.shape[1]
    mask_comb = np.zeros(df.shape, dtype=bool)
    for i in range(N):
        for j in range(M):
            mask_comb[i][j] = df.iat[i, j] or triangle[i][j]
    return mask_comb

raw = pd.read_csv("final_stats.csv", sep=";")
numerics = raw.select_dtypes('number')
normzd = normalize(numerics)

#pearson correlation matrix
# corr = normzd.corr(method='pearson')
# mask = corr.copy().applymap(significance_pearson)
# mask2 = np.triu(np.ones_like(corr, dtype=bool))
# mask_comb = combine_masks(mask, mask2)
# c = sns.heatmap(corr, annot=True)
# sns.heatmap(corr, annot=True).set_xticklabels(c.get_xticklabels(), rotation=-45)
# plt.show()
# sns.heatmap(corr, annot=True, mask=mask_comb).set_xticklabels(c.get_xticklabels(), rotation=-45)
# plt.show()

# #spearman's rank correlation matrix
# corr = normzd.corr(method='spearman')
# mask = corr.copy().applymap(significance_spearman)
# mask2 = np.triu(np.ones_like(corr, dtype=bool))
# mask_comb = combine_masks(mask, mask2)
# c = sns.heatmap(corr, annot=True)
# sns.heatmap(corr, annot=True).set_xticklabels(c.get_xticklabels(), rotation=-45)
# plt.show()
# sns.heatmap(corr, annot=True, mask=mask_comb).set_xticklabels(c.get_xticklabels(), rotation=-45)
# plt.show()

#autoarima
# arima.auto_arima(numerics['Sleep'], trace=True)
# arima.auto_arima(numerics['Studying'], trace=True)
# arima.auto_arima(numerics['Socializing'], trace=True)
# arima.auto_arima(numerics['Mood'], trace=True)

#FFT
# for v in ['Sleep','Studying','Socializing','Mood']:
#     t = np.arange(0,N,1)
#     x = numerics[v]

#     X = np.fft.fft(x)
#     N = len(X)
#     n = np.arange(0,N,1)
#     T = N
#     freq = n/T 

#     plt.figure(figsize = (8, 4))

#     plt.subplot(121)
#     plt.plot(t, x, 'r')
#     plt.xlabel('Time (days)')
#     plt.ylabel(v)

#     plt.subplot(122)
#     plt.stem(n, np.abs(X), 'b', \
#             markerfmt=" ", basefmt="-b")
#     plt.xlabel('Freq (1/days)')
#     plt.ylabel('FFT |X(freq)|')
#     plt.xlim(0, 30)
#     plt.ylim(0, 500)

#     plt.tight_layout()
#     plt.show()

#FFT-MA(k)
k = 5
for v in ['Sleep','Studying','Socializing','Mood']:
    t = np.arange(0,N-k+1,1)
    x = moving_average(numerics[v], k)

    X = np.fft.fft(x)
    n = np.arange(0,len(X),1)
    T = N-k+1
    freq = n/T

    plt.figure(figsize = (8, 4))

    plt.subplot(121)
    plt.plot(t, x, 'r')
    plt.xlabel('Time (days)')
    plt.ylabel(v)

    plt.subplot(122)
    print(n)
    print(x)
    plt.stem(n, np.abs(X), 'b', \
            markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (1/days)')
    plt.ylabel('FFT |X(freq)|')
    plt.xlim(0, 30)
    plt.ylim(0, 500)

    plt.tight_layout()
    plt.show()