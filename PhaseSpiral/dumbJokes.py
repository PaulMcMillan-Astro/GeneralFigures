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

# %%
x = np.linspace(0, 1, 10)
y = 2 + 3. * x + np.random.rand(10)
plt.plot(x,y,'kx', ms=10)
plt.plot(x, 2.5 + 3. * x, 'k-')
plt.xlabel('twitter')
plt.ylabel('y')
plt.xkcd()
#plt.show()
