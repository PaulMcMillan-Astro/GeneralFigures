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
# Some helpful imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as mtransforms
from IPython.display import display, HTML
import sys

# %% [markdown]
# # Code used to produce animations of the Sun's orbit on a background picture
#
# Sun's orbit is found with GalPot  
# Picture is from NASA/JPL-Caltech/ESO/R. Hurt  
# Galaxy rotates as a solid body at roughly 2.3 deg/Myr (i.e. 6 deg per frame) with initial assumptions. This places corotation inside the Sun's radius.

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
SaveAnim = False


# %%
SaveAnim = False
fig, ax = plt.subplots(1,1, figsize=(10,5),facecolor='k')
ims = []

#im = ax.imshow(img, extent=[-17, 17, -17, 17])

def solution(t, y0, dy0):
    a = y0
    b= (dy0+a-1)*(2.*np.sqrt(2))
    y = np.exp(-t) * (a*np.cos(2*np.sqrt(2)*t) + b*np.sin(2*np.sqrt(2)*t)) + np.sin(3*t)/3.
    return y

t = np.linspace(0, 10, 1000)


plt.plot(t, solution(t, 0, 0))
plt.show()

# %% [markdown]
# # Animate orbit on a black background

# %%
SaveAnim = False
fig, ax = plt.subplots(1,1, figsize=(10,5),facecolor='k')
ims = []

#im = ax.imshow(img, extent=[-17, 17, -17, 17])

def solution(t, y0, dy0):
    a = y0
    b= (dy0+a-1)(2*np.sqrt(2))
    y = np.exp(-t) * (a*np.cos(2*np.sqrt(2)*t) + b*np.sin(2*np.sqrt(2)*t)) + np.sin(3*t)/3.
    return y

t = np.linspace(0, 10, 1000)





for i in range(501):
    lines, = ax[0].plot((R*np.cos(phi))[0:i*10],(R*np.sin(phi))[0:i*10],c='w', lw=0.5)
    lines_z, = ax[1].plot((R*np.cos(phi))[0:i*10],z[0:i*10],c='w', lw=0.5)
    if i==0: 
        sca_xy = ax[0].scatter((R*np.cos(phi))[0],(R*np.sin(phi))[0],s=100,c='gold', marker = '*')
        sca_z  = ax[1].scatter((R*np.cos(phi))[0],z[0],s=100,c='gold', marker = '*')
    else: 
        sca_xy = ax[0].scatter((R*np.cos(phi))[i*10-1],(R*np.sin(phi))[i*10-1],s=100,c='gold', marker = '*')
        sca_z  = ax[1].scatter((R*np.cos(phi))[i*10-1],z[i*10-1],s=100,c='gold', marker = '*')
    #lines.append(sca_xy)
    #lines_z.append(sca_z)
    ims.append([lines, lines_z, sca_xy, sca_z])

for a in ax:
    a.set_aspect('equal')
    #a.axis('off')
    a.set_xlim(-17,17)
    a.set_ylim(-17,17)
    a.set_facecolor('k')
    for side in ['left', 'bottom']:
        a.spines[side].set_color('w')
    a.xaxis.label.set_color('w')
    a.yaxis.label.set_color('w')

    a.set_xlabel('x', color='w', weight='bold', fontsize=20, labelpad=-5)
ax[0].set_ylabel('y', color='w', weight='bold', fontsize=20, labelpad=-20)
ax[1].set_ylabel('z', color='w', weight='bold', fontsize=20, labelpad=-20)

ani = animation.ArtistAnimation(fig, ims, interval=40, blit=True,
                                repeat_delay=5000, repeat=True)

if SaveAnim: 
    ani.save('OrbitAnimations/ResonantXYXZOrbit.mp4')
else:
    plt.draw()
    plt.close(fig)
    display(HTML(ani.to_html5_video()))

# %%

# %%

vR_res, vz_res = 40, 77

xv_ini = np.array([8.21, 0.014, 0,
                   vR_res/Phi.kpc_Myr_to_km_s,
                   vz_res/Phi.kpc_Myr_to_km_s,
                   (-233.1-12.24)/Phi.kpc_Myr_to_km_s])

t_eval=np.linspace(0,13000,5000)

OrbitPath, _ = OI.getOrbitPathandStats(xv_ini,t_eval)

R    = OrbitPath[:,0]
z    = OrbitPath[:,1]
phi  = OrbitPath[:,2]-np.pi/2 # shift to put at x,y = 0,-8.21 at the start
vR   = OrbitPath[:,3]
vz   = OrbitPath[:,4]
vphi = OrbitPath[:,5]

SaveAnim = True
fig, ax = plt.subplots(figsize=(5,5),facecolor='k')
ims = []

#im = ax.imshow(img, extent=[-17, 17, -17, 17])

for i in range(501):
    lines, = ax.plot((R[0:i*10]),z[0:i*10],c='w', lw=0.5)
    if i==0:
        sca = ax.scatter((R[0]),z[0],s=100,c='gold', marker = '*')
    else:
        sca = ax.scatter((R[i*10-1]),z[i*10-1],s=100,c='gold', marker = '*')
    ims.append([lines, sca])

ax.set_aspect('equal')
#ax.axis('off')
ax.set_xlim(R.min() - 0.1 * (R.max()-R.min()),R.max() + 0.1 * (R.max()-R.min()))
ax.set_ylim(-1.1 * z.max(), 1.1 * z.max())
ax.set_facecolor('k')
for side in ['left', 'bottom']:
    ax.spines[side].set_color('w')
ax.xaxis.label.set_color('w')
ax.yaxis.label.set_color('w')

ax.set_xlabel('R', color='w', weight='bold', fontsize=20, labelpad=-5)
ax.set_ylabel('z', color='w', weight='bold', fontsize=20)

ani = animation.ArtistAnimation(fig, ims, interval=40, blit=True,
                                repeat_delay=5000, repeat=True)

if SaveAnim:
    ani.save('OrbitAnimations/ResonantRZOrbit.mp4')
else:
    plt.draw()
    plt.close(fig)
    display(HTML(ani.to_html5_video()))

