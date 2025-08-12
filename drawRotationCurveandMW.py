# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
# Some helpful imports
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#import matplotlib.transforms as mtransforms
#from IPython.display import display, HTML
import sys

# %% [markdown]
# # Code used to produce animations of the Sun's orbit on a background picture
#
# Sun's orbit is found with GalPot  
# Picture is from NASA/JPL-Caltech/ESO/R. Hurt  
# Galaxy rotates as a solid body at roughly 2.3 deg/Myr (i.e. 6 deg per frame) with initial assumptions. This places corotation inside the Sun's radius.

# %%
GalPotDir = '../GalPot/'
SaveFig = False
rotationDegreesPerStep = 6

# %%

sys.path.append(GalPotDir)
from GalPot import GalaxyPotential

# %%

Phi= GalaxyPotential(GalPotDir + "pot/PJM17_best.Tpot")


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
     'font.size' : 14})

# %%
Rlist=np.linspace(0.01,30,300)
vc = np.empty_like(Rlist)
fig, ax = plt.subplots(figsize=(5,3))
ax = plt.gca()

for i in range(len(Rlist)):
    vc[i] = Phi.Vcirc(Rlist[i]) * Phi.kpc_Myr_to_km_s
plt.plot(Rlist, vc, 'w')
ax.spines[['bottom', 'right', 'top']].set_visible(False)
ax.set_xticks([])
ax.set_yticks([0,100,200, 300])
ax.set_ylabel(r'rotation velocity [km/s]')
plt.tight_layout()
plt.savefig('OrbitAnimations/RotationCurve.png', transparent=True)
plt.show()

# %%
Rlist=np.linspace(0.01,30,300)
vc = np.empty_like(Rlist)
Rlist2 = np.linspace(12,30,300)
fig, ax = plt.subplots(figsize=(5,3))
ax = plt.gca()

for i in range(len(Rlist)):
    vc[i] = Phi.Vcirc(Rlist[i]) * Phi.kpc_Myr_to_km_s
plt.plot(Rlist, vc, 'w')
plt.plot(Rlist2, Phi.Vcirc(Rlist2[0]) * Phi.kpc_Myr_to_km_s * (Rlist2[0]/Rlist2)**0.5, 'w--')
ax.spines[['bottom', 'right', 'top']].set_visible(False)
ax.set_xticks([])
ax.set_yticks([0,100,200, 300])
ax.set_ylabel(r'rotation velocity [km/s]')
plt.tight_layout()
plt.savefig('OrbitAnimations/RotationCurve.png', transparent=True)
plt.show()

# %% [markdown]
# # Draw Sun's orbit on background

# %%
img = plt.imread("MilkyWay background.jpg")
imgwide = plt.imread('MilkyWay background_largecanvas.jpg')
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(img, extent=[-17, 17, -17, 17])
ax.plot(R*np.cos(phi),R*np.sin(phi),c='r')
ax.set_aspect('equal')
ax.axis('off')
plt.show()

# %% [markdown]
# # Animate Sun's orbit on a black background

# %%

fig, ax = plt.subplots(figsize=(10,10),facecolor='k')
ims = []

#im = ax.imshow(img, extent=[-17, 17, -17, 17])

for i in range(501):
    lines = ax.plot((R*np.cos(phi))[0:i*10],(R*np.sin(phi))[0:i*10],c='r')
    if i==0: sca = ax.scatter((R*np.cos(phi))[0],(R*np.sin(phi))[0],s=100,c='gold')
    else: sca = ax.scatter((R*np.cos(phi))[i*10-1],(R*np.sin(phi))[i*10-1],s=100,c='gold')
    lines.append(sca)
    ims.append(lines)
ax.set_aspect('equal')
ax.axis('off')

ax.set_xlim(-17,17)
ax.set_ylim(-17,17)

ani = animation.ArtistAnimation(fig, ims, interval=40, blit=True,
                                    repeat_delay=5000, repeat=True)

if SaveAnim: 
    ani.save('SunOrbit.mp4')
else:
    plt.draw()
    plt.close(fig)
    display(HTML(ani.to_html5_video()))

# %% [markdown]
# # Animate the Sun's orbit on a static background MW
#
# ### N.B. from here onwards the render timegets longer:~20 minutes for the first one. Longer when you want the MW to rotate

# %%

fig, ax = plt.subplots(figsize=(10,10),facecolor='k')
ims = []

im = ax.imshow(img, extent=[-17, 17, -17, 17])

for i in range(501):
    lines = ax.plot((R*np.cos(phi))[0:i*10],(R*np.sin(phi))[0:i*10],c='r')
    if i==0: sca = ax.scatter((R*np.cos(phi))[0],(R*np.sin(phi))[0],s=100,c='gold')
    else: sca = ax.scatter((R*np.cos(phi))[i*10-1],(R*np.sin(phi))[i*10-1],s=100,c='gold')
    lines.append(sca)
    ims.append(lines)
ax.set_aspect('equal')
ax.axis('off')


ani = animation.ArtistAnimation(fig, ims, interval=40, blit=True,
                                    repeat_delay=5000, repeat=True)

if SaveAnim:
    ani.save('SunOrbitStaticMW2.mp4')
else: 
    plt.draw()
    plt.close(fig)
    display(HTML(ani.to_html5_video())) 


# %% [markdown]
# # Sun orbiting on rotating Milky Way
#
# The default value: 6 deg/step means that the Sun is rotating slower than the spiral arms by a bit.

# %%

fig, ax = plt.subplots(figsize=(10,10),facecolor='k')
ims = []

for i in range(501):
    im = ax.imshow(imgwide, extent=[-17*8/5.6, 17*8/5.6, -17*8/5.6, 17*8/5.6])
    trans_data = mtransforms.Affine2D().rotate_deg(-rotationDegreesPerStep*i) + ax.transData
    im.set_transform(trans_data)
    lines = ax.plot((R*np.cos(phi))[0:i*10],(R*np.sin(phi))[0:i*10],c='r')
    if i==0: sca = ax.scatter((R*np.cos(phi))[0],(R*np.sin(phi))[0],s=100,c='gold')
    else: sca = ax.scatter((R*np.cos(phi))[i*10-1],(R*np.sin(phi))[i*10-1],s=100,c='gold')
    lines.append(im)
    lines.append(sca)
    ims.append(lines)
ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim([-17,17])
ax.set_ylim([-17,17])


ani = animation.ArtistAnimation(fig, ims, interval=40, blit=True,
                                    repeat_delay=5000, repeat=True)

if SaveAnim:
    ani.save('SunOrbitRotatingMW.mp4')
else: 
    plt.draw()
    plt.close(fig)
    display(HTML(ani.to_html5_video()))

# %% [markdown]
# # Sun with evolving radius on rotating MW
#
# Initially the star is (artificially) given the same rotation speed as the spiral arms at a lower radius. It is then gradually moved outwards to match the Sun's current orbit.
#

# %%

fig, ax = plt.subplots(figsize=(10,10),facecolor='k')
ims = []


phiPlusOmegaT = phi+0.1*np.deg2rad(6)*np.arange(len(phi))

R0 = 8.21*np.ones_like(R)
# Initial radius 5.5 kpc
R0[:(len(R) // 3)] = 5.5
R0[(len(R) // 3):(2 * (len(R) // 3))] = 5.5 + (8.21 - 5.5) * np.linspace(0,1,len(R) // 3)

R_fake = R0 + R - 8.21

phi_fake = np.zeros_like(R_fake) - np.pi/2
dphi_basic = -0.1*np.deg2rad(rotationDegreesPerStep)

# Rotation rate = 6 deg/frame (i.e. same as spiral) initially, slowing to raotation rate of Sun
for i in range(1,len(R)):
    dphi_real = (phi[i]-phi[i-1])
    if dphi_real > np.pi: dphi_real -=  2*np.pi
    phi_fake[i] = phi_fake[i-1] + dphi_basic + (dphi_real-dphi_basic) * (R0[i]-5.5)/(8.21-5.5)
    
for i in range(501):
    im = ax.imshow(imgwide, extent=[-17*8/5.6, 17*8/5.6, -17*8/5.6, 17*8/5.6])
    trans_data = mtransforms.Affine2D().rotate_deg(-6*i) + ax.transData
    im.set_transform(trans_data)
    lines = ax.plot((R_fake*np.cos(phi_fake))[0:i*10],(R_fake*np.sin(phi_fake))[0:i*10],c='r')
    if i==0: sca = ax.scatter((R_fake*np.cos(phi_fake))[0],(R_fake*np.sin(phi_fake))[0],s=100,c='gold')
    else: sca = ax.scatter((R_fake*np.cos(phi_fake))[i*10-1],(R_fake*np.sin(phi_fake))[i*10-1],s=100,c='gold')
    lines.append(im)
    lines.append(sca)
    ims.append(lines)
ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim([-17,17])
ax.set_ylim([-17,17])


ani = animation.ArtistAnimation(fig, ims, interval=40, blit=True,
                                    repeat_delay=5000, repeat=True)

if SaveAnim:
    ani.save('SunHistoryOrbitRotatingMW.mp4')
else: 
    plt.draw()
    plt.close(fig)
    display(HTML(ani.to_html5_video()))


# %%
