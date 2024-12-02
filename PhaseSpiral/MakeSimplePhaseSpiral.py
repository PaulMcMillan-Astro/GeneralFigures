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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import matplotlib as mpl
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1 import AxesGrid

import matplotlib.animation as animation
from IPython.display import HTML


# %% [markdown]
# # Really really basic model
#
# $Vz = r \cos (\phi-\Omega t)$  
# $z = r \sin (\phi-\Omega t)$
#

# %%

plt.rcParams.update({
     "axes.facecolor": "black",     
     "figure.facecolor": "black",
     "figure.edgecolor": "black", 
     "savefig.facecolor": "black",
     "savefig.edgecolor": "black",
     "text.color": "white",
     "axes.labelcolor": "white",
     "axes.edgecolor": "white",
     "xtick.color": "white",
     "ytick.color": "white",
     'xtick.minor.visible' : True, 
     'xtick.top' : True,
     'ytick.minor.visible' : True, 
     'ytick.right' : True,
     'xtick.direction' : 'in', 
     'ytick.direction' :'in',
     'font.size' : 14, 
     'axes.titlesize' : 24})

# %%
N = 1000000
vz = np.random.randn(N)
z = np.random.randn(N)-0.3
r = (vz**2 + z**2)**0.5
phi = np.arctan2(vz,z)
Omega = 1/(1 + 0.5 * r)


# %%
t = 30
plt.hist2d(r*np.cos(phi-Omega*t),30*r*np.sin(phi-Omega*t), [np.linspace(-2.5,2.5,50),np.linspace(-75,75,50)])
plt.xlabel(r'$Z$')
plt.ylabel(r'$V_Z$')
plt.show()

# %%
fig, ax = plt.subplots(figsize=(5,5),facecolor='k')
for side in ['top','bottom','left','right']:
    ax.spines[side].set_color('white')
ax.tick_params(which='both', labelcolor='white')

ax.hist2d(r*np.cos(phi-Omega*t),30*r*np.sin(phi-Omega*t), [np.linspace(-2.5,2.5,50),np.linspace(-75,75,50)], cmap='bone')
ax.set_xlabel(r'$Z$', color='w')
ax.set_ylabel(r'$V_Z$', color='w')
plt.show()


# %%
N = 1000000
vz = np.random.randn(N)
z = np.random.randn(N)
r = (vz**2 + z**2)**0.5
phi = np.arctan2(vz,z)
Omega = 1/(1 + 0.5 * r)

zscale = 0.5
vzscale = 20

zbins, vzbins = np.linspace(-1,1,100),np.linspace(-60,60,100)

fig, ax = plt.subplots(figsize=(5,5),facecolor='k', constrained_layout=True)
for side in ['top','bottom','left','right']:
    ax.spines[side].set_color('white')
ax.tick_params(which='both', labelcolor='white')
im_ani = []


for i in range(40):
    t = 0.2 * i
    z = r*np.cos(phi-Omega*t)
    vz = r*np.sin(phi-Omega*t)
    im = ax.hist2d(zscale * z, vzscale * vz, [zbins, vzbins], cmap='bone')[-1]
    im_ani.append([im])

z = z-0.3
r = (vz**2 + z**2)**0.5
phi = np.arctan2(vz,z)
Omega = 1/(1 + 0.5 * r)
    
for i in range(200):
    t = 0.2 * i
    im = ax.hist2d(zscale * r*np.cos(phi-Omega*t),vzscale * r*np.sin(phi-Omega*t),
                    [zbins, vzbins], cmap='bone')[-1]
    im_ani.append([im])

ax.set_xlabel(r'$Z$', color='w')
ax.set_ylabel(r'$V_Z$', color='w')


ani = animation.ArtistAnimation(fig, im_ani, interval=50, blit=True,
                                    repeat_delay=5000, repeat=True)
HTML(ani.to_html5_video())

# %%
ani.save('Movies/SimplePhaseSpiral.mp4')

