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
plt.rcParams['font.size'] = 18
plt.rcParams.update({
     'xtick.minor.visible' : True, 
     'xtick.top' : True,
     'ytick.minor.visible' : True, 
     'ytick.right' : True,
     'xtick.direction' : 'in', 
     'ytick.direction' :'in',
     'font.size' : 14, 
     'axes.titlesize' : 24})

# %%

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"

plt.rcParams["text.latex.preamble"] = r"\usepackage{txfonts}"

# %%
plt.figure(figsize=(5,1.5))

x=np.linspace (0, 10, 1000)
t=0

k = 2*np.pi/0.5
omega = 2*np.pi/1
Deltak = 0.178*k
Deltaomega = 0.2*omega
y = 2 * np.sin((k+Deltak / 2) * x - (omega+Deltaomega / 2) * t) * np.cos (Deltak * x / 2 - Deltaomega * t / 2)
plt.yticks([-2,-1,0,1,2], [])
plt.xticks([0,2,4,6,8,10], [])
plt.xlabel(r'$x$')
plt.plot(x,y, 'k')
plt.savefig('wavepacket.pdf', bbox_inches='tight')
plt.show()
