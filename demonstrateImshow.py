# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Make the plots have a bigger font size
plt.rcParams.update({
     'font.size' : 18, 
     'axes.titlesize' : 24})

# %%
x = np.random.rand(1000) # values between 0 and 1
y = 2+np.random.rand(1000)+3*np.sqrt(x) # values between 2 and 6

plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %%
# (not doing any clever statistic, because this is a basic example)
res = stats.binned_statistic_2d(x, y, None, 'count', bins=[np.linspace(0, 1, 20), np.linspace(2, 6, 20)])


plt.imshow(res.statistic.T, # Note the transpose - this is because of how imshow works
           aspect='auto', # In some cases you need to specify the aspect ratio
           origin='lower', # This is important to get the right orientation (though I believe it is the default)
           extent=[0, 1, 2, 6] # This is the range of the x and y axis. If not set you will just get the bin numbers on these axes
           )
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Number of points')
plt.show()
