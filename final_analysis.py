import pandas as pd               #1.4.4
import numpy as np                #1.22.4
import seaborn as sns             #0.12.0
import matplotlib.pyplot as plt   #3.5.2
from pmdarima import arima        #2.0.4

N=332 #number of samples
STEST = 2/np.sqrt(N)
TTEST = 1.96

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

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

raw = pd.read_csv("final_stats.csv", sep=";")
numerics = raw.select_dtypes('number')

#pearson correlation matrix
corr = numerics.corr(method='pearson')
mask = corr.copy().applymap(significance_pearson)
mask2 = np.triu(np.ones_like(corr, dtype=bool))
mask_comb = np.logical_or(mask, mask2)
c = sns.heatmap(corr, annot=True)
c.set_xticklabels(c.get_xticklabels(), rotation=-45)
plt.show()
c = sns.heatmap(corr, annot=True, mask=mask_comb)
c.set_xticklabels(c.get_xticklabels(), rotation=-45)
plt.show()

#spearman's rank correlation matrix
corr = numerics.corr(method='spearman')
mask = corr.copy().applymap(significance_spearman)
mask2 = np.triu(np.ones_like(corr, dtype=bool))
mask_comb = np.logical_or(mask, mask2)
c = sns.heatmap(corr, annot=True)
c.set_xticklabels(c.get_xticklabels(), rotation=-45)
plt.show()
c = sns.heatmap(corr, annot=True, mask=mask_comb)
c.set_xticklabels(c.get_xticklabels(), rotation=-45)
plt.show()

#autoarima
for v in ['Sleep','Studying','Socializing','Mood']:
    arima.auto_arima(numerics[v], trace=True)

# FFT
for v in ['Sleep','Studying','Socializing','Mood']:
    t = np.arange(0,N,1)
    x = numerics[v]
    X = np.fft.fft(x)
    n = np.arange(0,len(X),1)
    T = N
    freq = n/T 

    plt.figure(figsize = (8, 4))

    plt.subplot(121)
    plt.plot(t, x, 'r')
    plt.xlabel('Time (days)')
    plt.ylabel(v)

    plt.subplot(122)
    plt.stem(n, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (1/days)')
    plt.ylabel('FFT |X(freq)|')
    plt.xlim(0, 30)
    plt.ylim(0, 500)

    plt.tight_layout()
    plt.show()

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
    plt.stem(n, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (1/days)')
    plt.ylabel('FFT |X(freq)|')
    plt.xlim(0, 30)
    plt.ylim(0, 500)

    plt.tight_layout()
    plt.show()