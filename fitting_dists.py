#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 18:28:35 2022

@author: mahdi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy import stats
from scipy.stats import (
    norm, genextreme, logistic, beta,
    gengamma, gennorm, dweibull, dgamma, powernorm,
    rayleigh, weibull_max, weibull_min,
    laplace, alpha, genexpon, genpareto, hypsecant,
    halfnorm, invgamma, ncf, nct, nakagami, semicircular,
    rice, foldnorm,  cosine, wald, truncexpon, truncnorm, t, rdist
    )

distributions = [
    norm, genextreme, logistic,
    gengamma, gennorm, dweibull, dgamma, powernorm,
    rayleigh, weibull_max, weibull_min,
    laplace, alpha, genexpon, genpareto, hypsecant,
    halfnorm, invgamma, ncf, nct, nakagami, semicircular,
    rice, foldnorm,  cosine, wald, truncexpon, truncnorm, t, rdist
    ]


def kstest(data, distname, paramtup):
    ks = stats.kstest(data, distname, paramtup, ksN)[1]   # return p-value
    return ks             # return p-value


def fitdist(data, dist):
    fitted = dist.fit(data, floc=0.0)
    ks = kstest(data, dist.name, fitted)
    res = (dist.name, ks, *fitted)
    return res


def plot_fitted_pdf(df, data):

    N = len(df)
    chrows = math.ceil(N/3)      # how many rows of charts if 3 in a row
    fig, ax = plt.subplots(chrows, 3, figsize=(20, 5*chrows))
    ax = ax.ravel()
    dfRV = pd.DataFrame()

    for i in df.index:

        # D_row = df.iloc[i,:-1]
        D_name = df.iloc[i, 0]
        D = df.iloc[i, 7]
        KSp = df.iloc[i, 1]
        params = df.iloc[i, 2:7]
        params = [p for p in params if ~np.isnan(p)]

        # calibrate x-axis by finding the 1% and 99% quantiles in percent point function
        x = np.linspace(
                    D.ppf(0.01, *params),
                    D.ppf(0.99, *params), 100)

        # plot histogram of actual observations
        ax[i].hist(data, density=True, histtype='stepfilled', alpha=0.5, bins=30)
        # plot fitted distribution
        rv = D(*params)
        title = f'pdf {D_name}, with p(KS): {KSp:.2f}'
        ax[i].plot(x, rv.pdf(x), 'r-', lw=2, label=title)
        ax[i].legend(loc="upper right", frameon=False, fontsize=font)


stdratio = 2.0
font = 20
A = np.random.uniform
NF = 1000000                  # number of frequency in spectrum!
ksN = 100           # Kolmogorov-Smirnov KS test for goodness of fit: samples
ALPHA = 0.05        # significance level for hypothesis test
np.random.seed(20)
dt = 0.1
T = np.arange(0, 32.0, dt)
y = np.zeros((T.shape), dtype=np.float)
for i in range(NF):
    y += A(-1, 1) * np.sin(2*np.pi*np.random.uniform(0, 0.5*(dt**-1))*T)
# y /= NF
noise = (y.std()/stdratio) * np.random.randn(T.size)
yy = y + noise
s = np.abs(2*(np.fft.rfft(y, T.size))/T.size)
ss = np.abs(2*(np.fft.rfft(yy, T.size))/T.size)
df = 1./(T.size*dt)
F = np.arange(T.size//2+1)*df
spec_res = ss - s
# s = (0:N-1)*Fs/N
result_noise = stats.describe(noise, bias=False)
result_res = stats.describe(spec_res, bias=False)

# ks = stats.stats.kstest(spec_res, 'norm')
# # print(ks)
fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(T, yy, 'b', alpha=0.5, lw=0.5)
ax[0, 0].plot(T, y, 'k--')
ax[0, 0].set_title('noisy/noise-free data', fontsize=font)

ax[0, 1].hist(noise, bins=50, density=True, alpha=0.5)
ax[0, 1].set_title('Hist of noise', fontsize=font)
ax[0, 1].legend(["N: {describe.nobs}\nMinMax: \
({describe.minmax[0]:2.3f}, {describe.minmax[1]:2.3f})\nMean:\
{describe.mean:E}\nVar:{describe.variance:E}\nSkewness:\
{describe.skewness:2.3f}\nKurtosis:{describe.kurtosis:2.3f}".
                 format(describe=result_noise)], fontsize=font)
ax[1, 0].plot(F, ss, 'b', alpha=0.5, lw=0.5)
ax[1, 0].plot(F, s, 'k--')
ax[1, 0].set_title('Spectrum of data', fontsize=font)

ax[1, 1].hist(spec_res, bins=50, density=True, alpha=0.5)
ax[1, 1].set_title('Hist of spectrum residual', fontsize=font)
ax[1, 1].legend(["N: {describe.nobs}\nMinMax: ({describe.minmax[0]:2.3f},\
                 {describe.minmax[1]:2.3f})\nMean:{describe.mean:E}\nVar:\
                 {describe.variance:E}\nSkewness:{describe.skewness:2.3f}\n\
                 Kurtosis:{describe.kurtosis:2.3f}".format(describe=result_res)], fontsize=font)


res = [fitdist(spec_res, D) for D in distributions]

# convert the fitted list of tuples to dataframe
pd.options.display.float_format = '{:,.3f}'.format
df = pd.DataFrame(res, columns=["distribution", "KS p-value", "param1", "param2", "param3", "param4", "param5"])
df["distobj"] = distributions
df.sort_values(by=["KS p-value"], inplace=True, ascending=False)
df.reset_index(inplace=True)
df.drop("index", axis=1, inplace=True)


df_ks = df.loc[df["KS p-value"] > ALPHA]
print(df_ks.shape)
print("Fitted Distributions with KS p-values > ALPHA:")
if df_ks.shape[0] != 0:
    plot_fitted_pdf(df_ks, spec_res)
