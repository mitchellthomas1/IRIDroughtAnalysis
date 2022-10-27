#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:57:20 2022

@author: Mitchell
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from builddata import return_data

arr = return_data()['imagearray']

vals = np.random.choice(arr.flatten(), 100000)
a,l,s = stats.gamma.fit(vals)
cdf_vals = stats.gamma.cdf(vals, a, loc = l, scale = s)
plt.hist(cdf_vals, bins = 'auto')
plt.show()

norm = stats.norm.ppf(cdf_vals)
plt.hist(norm, bins = 'auto')
plt.show()


cdf_vals = stats.gamma.cdf(vals, a, loc = l, scale = s)
norm = stats.norm.ppf(cdf_vals)


# np.random.seed(42)
# data = sorted(stats.lognorm.rvs(s=0.5, loc=1, scale=1000, size=1000))

# # fit lognormal distribution
# shape, loc, scale = stats.lognorm.fit(data, loc=0)
# pdf_lognorm = stats.lognorm.pdf(data, shape, loc, scale)
# # visualize

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# ax1.hist(data, bins='auto', density=True)
# ax1.plot(data, pdf_lognorm)
# ax1.set_ylabel('probability')
# ax1.set_title('Linear Scale')


