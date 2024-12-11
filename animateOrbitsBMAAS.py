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
# Some helpful imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.transforms as mtransforms
from IPython.display import display, HTML
import sys

# %% [markdown]
# # Code used to produce animations of orbits
#
# This uses Galpot to give a gravitational potential. 
#
# To use it properly you need to install GalPot from https://github.com/PaulMcMillan-Astro/GalPot

# %%
# Change this to the name of the directory where you have the GalPot code
GalPotDir = '../GalPot/'
SaveAnim = False


# %%

sys.path.append(GalPotDir)
from GalPot import GalaxyPotential, OrbitIntegrator

# %%
# Plot on black background

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

Phi= GalaxyPotential(GalPotDir + "pot/PJM17_best.Tpot")
OI = OrbitIntegrator(Phi)

# Initial conditions for the orbit
xv_ini = np.array([8.21, 0.014, 0,
                   -51.1/Phi.kpc_Myr_to_km_s,
                   47.25/Phi.kpc_Myr_to_km_s,
                   (-233.1-12.24)/Phi.kpc_Myr_to_km_s])

t_eval=np.linspace(0,1300,5000)

OrbitPath, _ = OI.getOrbitPathandStats(xv_ini,t_eval)

R    = OrbitPath[:,0]
z    = OrbitPath[:,1]
phi  = OrbitPath[:,2]-np.pi/2 # shift to put at x,y = 0,-8.21 at the start
vR   = OrbitPath[:,3]
vz   = OrbitPath[:,4]
vphi = OrbitPath[:,5]


# %% [markdown]
# # Animate star's orbit on a black background

# %%
SaveAnim = False
fig, ax = plt.subplots(1,2, figsize=(10,5),facecolor='k')
ims = []

#im = ax.imshow(img, extent=[-17, 17, -17, 17])

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
    ani.save('OrbitAnimations/SingleXYXZOrbit.mp4')
else:
    plt.draw()
    plt.close(fig)
    display(HTML(ani.to_html5_video()))

# %% [markdown]
# ## Now I care about the R-z plane

# %%
# repeat the above except only one panel which shows the orbit in the R-z plane
SaveAnim = False
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
ax.set_ylabel('z', color='w', weight='bold', fontsize=20, labelpad=-20)

ani = animation.ArtistAnimation(fig, ims, interval=40, blit=True,
                                repeat_delay=5000, repeat=True)

if SaveAnim:
    ani.save('OrbitAnimations/SingleRZOrbit.gif')
else:
    plt.draw()
    plt.close(fig)
    display(HTML(ani.to_html5_video()))


# %%
# Animate three orbits with same starting point, energy and Lz

KE = 60**2
Lz = (-233.1-12.24) * 8.21

vR0 = np.array([15, 30, 50])

t_eval=np.linspace(0,1300,5000)

R = np.empty([len(vR0), len(t_eval)])
z = np.empty([len(vR0), len(t_eval)])

for i in range(len(vR0)):
    xv_ini = np.array([8.21, 0.014, 0,
                   -vR0[i]/Phi.kpc_Myr_to_km_s,
                   np.sqrt(KE - vR0[i]**2)/Phi.kpc_Myr_to_km_s,
                   (-233.1-12.24)/Phi.kpc_Myr_to_km_s])

    t_eval=np.linspace(0,1300,5000)

    OrbitPath, _ = OI.getOrbitPathandStats(xv_ini,t_eval)

    R[i]    = OrbitPath[:,0]
    z[i]    = OrbitPath[:,1]


SaveAnim = False
fig, ax = plt.subplots(figsize=(5,5),facecolor='k')
ims = []

#im = ax.imshow(img, extent=[-17, 17, -17, 17])

for i in range(501):
    lines0, = ax.plot((R[0][0:i*10]),z[0][0:i*10],c='w', lw=0.9)
    lines1, = ax.plot((R[1][0:i*10]),z[1][0:i*10],c='cyan', lw=0.9)
    lines2, = ax.plot((R[2][0:i*10]),z[2][0:i*10],c='r', lw=0.9)
    ims.append([lines0, lines1, lines2])

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
ax.set_ylabel('z', color='w', weight='bold', fontsize=20, labelpad=-20)

ani = animation.ArtistAnimation(fig, ims, interval=40, blit=True,
                                repeat_delay=5000, repeat=True)

if SaveAnim:
    ani.save('OrbitAnimations/ThreeRZOrbit.gif')
else:
    plt.draw()
    plt.close(fig)
    display(HTML(ani.to_html5_video()))

# %%
# Animate as above, but now with an additional panel showing a surface-of-section plot
SaveAnim = False
fig, ax = plt.subplots(1,2,figsize=(10,5),facecolor='k')
ims = []

#im = ax.imshow(img, extent=[-17, 17, -17, 17])

vR0 = np.array([15, 30, 50])

t_eval=np.linspace(0,1300,10000)

R = np.empty([len(vR0), len(t_eval)])
z = np.empty([len(vR0), len(t_eval)])
vR = np.empty([len(vR0), len(t_eval)])

for i in range(len(vR0)):
    xv_ini = np.array([8.21, 0.014, 0,
                   -vR0[i]/Phi.kpc_Myr_to_km_s,
                   np.sqrt(KE - vR0[i]**2)/Phi.kpc_Myr_to_km_s,
                   (-233.1-12.24)/Phi.kpc_Myr_to_km_s])

    OrbitPath, _ = OI.getOrbitPathandStats(xv_ini,t_eval)

    R[i]    = OrbitPath[:,0]
    z[i]    = OrbitPath[:,1]
    vR[i]   = OrbitPath[:,3]


for i in range(501):
    if i<50:
        lines0, = ax[0].plot((R[0][0:i*20]),z[0][0:i*20],c='w', lw=0.9)
        lines1, = ax[0].plot((R[1][0:i*20]),z[1][0:i*20],c='cyan', lw=0.9)
        lines2, = ax[0].plot((R[2][0:i*20]),z[2][0:i*20],c='r', lw=0.9)
    else:
        lines0, = ax[0].plot((R[0][i*20-1000:i*20]),z[0][i*20-1000:i*20],c='w', lw=0.9)
        lines1, = ax[0].plot((R[1][i*20-1000:i*20]),z[1][i*20-1000:i*20],c='cyan', lw=0.9)
        lines2, = ax[0].plot((R[2][i*20-1000:i*20]),z[2][i*20-1000:i*20],c='r', lw=0.9)
    mask0 = np.abs(z[0][0:i*20]) < 0.01
    mask1 = np.abs(z[1][0:i*20]) < 0.01
    mask2 = np.abs(z[2][0:i*20]) < 0.01
    sos0 = ax[1].scatter(R[0][0:i*20][mask0],vR[0][0:i*20][mask0],c='w', s=2)
    sos1 = ax[1].scatter(R[1][0:i*20][mask1],vR[1][0:i*20][mask1],c='cyan', s=2)
    sos2 = ax[1].scatter(R[2][0:i*20][mask2],vR[2][0:i*20][mask2],c='r', s=2)
    ims.append([lines0, lines1, lines2,sos0, sos1, sos2])

ax[0].set_aspect('equal')
#ax.axis('off')
ax[0].set_xlim(R.min() - 0.1 * (R.max()-R.min()),R.max() + 0.1 * (R.max()-R.min()))
ax[0].set_ylim(-1.1 * z.max(), 1.1 * z.max())

ax[1].set_xlim(R.min() - 0.1 * (R.max()-R.min()),R.max() + 0.1 * (R.max()-R.min()))
ax[1].set_ylim(1.1 * vR.min(), 1.1 * vR.max())
for a in ax:
    a.set_facecolor('k')
    for side in ['left', 'bottom']:
        a.spines[side].set_color('w')
    a.xaxis.label.set_color('w')
    a.yaxis.label.set_color('w')

    a.set_xlabel('R', color='w', weight='bold', fontsize=20, labelpad=-5)
ax[0].set_ylabel('z', color='w', weight='bold', fontsize=20, labelpad=-20)
ax[1].set_ylabel('vR', color='w', weight='bold', fontsize=20, labelpad=-20)

ani = animation.ArtistAnimation(fig, ims, interval=40, blit=True,
                                repeat_delay=5000, repeat=True)

if SaveAnim:
    ani.save('OrbitAnimations/ThreeRzSoSOrbit.gif')
else:
    plt.draw()
    plt.close(fig)
    display(HTML(ani.to_html5_video()))


# %%
R

# %%
R1, R2 = 6, 12
Lz1, Lz2 = Phi.LfromRc(R1), Phi.LfromRc(R2)
xv_ini_1 = np.array([R1,0, 0,
                   120/Phi.kpc_Myr_to_km_s,
                   80/Phi.kpc_Myr_to_km_s,
                   Lz1/R1])
xv_ini_2 = np.array([R2,0, 0,
                     130/Phi.kpc_Myr_to_km_s,
                     60/Phi.kpc_Myr_to_km_s,
                     Lz2/R2])

t_eval=np.linspace(0,1300,5000)

OrbitPath_1, _ = OI.getOrbitPathandStats(xv_ini_1,t_eval)
OrbitPath_2, _ = OI.getOrbitPathandStats(xv_ini_2,t_eval)

fig, ax = plt.subplots(figsize=(5,3),facecolor='k')

ax.set_facecolor('k')
for side in ['left', 'bottom']:
    ax.spines[side].set_color('w')
ax.xaxis.label.set_color('w')
ax.yaxis.label.set_color('w')


ax.set_xlabel('R', color='w', weight='bold', fontsize=20, labelpad=-5)
ax.set_ylabel('z', color='w', weight='bold', fontsize=20, labelpad=0)

plt.plot(OrbitPath_1[:,0],OrbitPath_1[:,1],c='b')
plt.plot(OrbitPath_2[:,0],OrbitPath_2[:,1],c='r')
plt.vlines([R1,R2], -5, 5, ls=':', color='w')
ax.set_aspect('equal')
plt.show()


# %%
R1, R2 = 6, 12
Lz1, Lz2 = Phi.LfromRc(R1), Phi.LfromRc(R2)
xv_ini_1 = np.array([R1,0, 0,
                   120/Phi.kpc_Myr_to_km_s,
                   80/Phi.kpc_Myr_to_km_s,
                   Lz1/R1])
xv_ini_2 = np.array([R2,0, 0,
                     130/Phi.kpc_Myr_to_km_s,
                     60/Phi.kpc_Myr_to_km_s,
                     Lz2/R2])

t_eval=np.linspace(0,1300,5000)

OrbitPath_1, _ = OI.getOrbitPathandStats(xv_ini_1,t_eval)
OrbitPath_2, _ = OI.getOrbitPathandStats(xv_ini_2,t_eval)

fig, ax = plt.subplots(figsize=(5,3),facecolor='k')

ax.set_facecolor('k')
for side in ['left', 'bottom']:
    ax.spines[side].set_color('w')
ax.xaxis.label.set_color('w')
ax.yaxis.label.set_color('w')


ax.set_xlabel('R', color='w', weight='bold', fontsize=20, labelpad=-5)
ax.set_ylabel('z', color='w', weight='bold', fontsize=20, labelpad=0)
plt.vlines([R1,R2], -5, 5, ls=':', color='w')

plt.scatter(OrbitPath_1[:,0],OrbitPath_1[:,1],c=Phi.kpc_Myr_to_km_s*Lz1/OrbitPath_1[:,0], s=1, vmin=150, vmax=300)
#plt.colorbar()
plt.scatter(OrbitPath_2[:,0],OrbitPath_2[:,1],c=Phi.kpc_Myr_to_km_s*Lz2/OrbitPath_2[:,0], s=1, vmin=150, vmax=300)
plt.colorbar(label = r'$v_\phi$')
ax.set_aspect('equal')
plt.show()

# %%
