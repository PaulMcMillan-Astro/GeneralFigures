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

# %% [markdown]
# # Version history
#
# ### As MakeMWVideos:
# v0 - Hack something together for Uppsala talk to match Simon's spiral  
# v1 - also hack something together to match phase spiral animation

# %% [markdown]
#

# %%
plotdir = 'PhaseSpiral/'
import os
#os.mkdir(plotdir)

# %%
import matplotlib as mpl
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import stats

#from plotparameters import *

import matplotlib.animation as animation
from IPython.display import HTML


# %% [markdown]
# ## Start with some basics
#
# Put the Hurt pic where I want it and show a sector

# %%
img = plt.imread("MilkyWay background.jpg")
R0 = 8


# %%
def drawLboxLines(lmin, lmax, ax, Rmin=10.5, Rmax =13.5):
    R0, R = 8.0, np.array([Rmin,Rmax])
    alphamin = np.pi - lmin - np.arcsin(R0*np.sin(lmin)/R)
    ymin,xmin = -R * np.cos(alphamin), -R * np.sin(alphamin)
    alphamax = np.pi - lmax - np.arcsin(R0*np.sin(lmax)/R)
    ymax,xmax = -R * np.cos(alphamax), -R * np.sin(alphamax)

    xlo = -R[0] * np.sin(np.linspace(alphamin[0],alphamax[0]))
    ylo = - R[0] * np.cos(np.linspace(alphamin[0],alphamax[0]))

    xhi = -R[1] * np.sin(np.linspace(alphamin[1],alphamax[1]))
    yhi = - R[1] * np.cos(np.linspace(alphamin[1],alphamax[1]))
    lines = []
    lines.append(ax.plot(xmin,ymin,'w')[0])
    lines.append(ax.plot(xmax,ymax,'w')[0])
    lines.append(ax.plot(xlo,ylo,'w')[0])
    lines.append(ax.plot(xhi,yhi,'w')[0])
    return lines

def drawLboxLabel(lmin, lmax, ax):
    return ax.text(0,-16,rf'${{{np.rad2deg(lmin):.0f}}}^\circ<\ell<{{{np.rad2deg(lmax):.0f}}}^\circ$', 
                   color='w', ha='center', va='bottom',size='xx-large',fontweight='bold')


# %%
def drawRboxLines(Rmin, Rmax, ax):
    R0, R = 8.0, np.array([Rmin,Rmax])
    lmin, lmax = np.deg2rad(170),np.deg2rad(190)
    alphamin = np.pi - lmin - np.arcsin(R0*np.sin(lmin)/R)
    ymin,xmin = -R * np.cos(alphamin), -R * np.sin(alphamin)
    alphamax = np.pi - lmax - np.arcsin(R0*np.sin(lmax)/R)
    ymax,xmax = -R * np.cos(alphamax), -R * np.sin(alphamax)

    xlo = -R[0] * np.sin(np.linspace(alphamin[0],alphamax[0]))
    ylo = - R[0] * np.cos(np.linspace(alphamin[0],alphamax[0]))

    xhi = -R[1] * np.sin(np.linspace(alphamin[1],alphamax[1]))
    yhi = - R[1] * np.cos(np.linspace(alphamin[1],alphamax[1]))
    lines = []
    lines.append(ax.plot(xmin,ymin,'r',lw=2)[0])
    lines.append(ax.plot(xmax,ymax,'r',lw=2)[0])
    lines.append(ax.plot(xlo,ylo,'r',lw=2)[0])
    lines.append(ax.plot(xhi,yhi,'r',lw=2)[0])
    return lines

def drawRboxLabel(Rmin, Rmax, ax):
    return ax.text(0,-15.5,rf'${{{Rmin:.1f}}}<R<{{{Rmax:.1f}}}$', 
                   color='w', ha='center', va='top',size='xx-large',fontweight='bold')


# %%
def drawPhiboxLines(phimin, phimax, ax, Rmin=8.5, Rmax =10.5):
    R0, R = 8.0, np.array([Rmin,Rmax])

    xlo = -R[0] * np.sin(np.linspace(phimin,phimax))
    ylo = R[0] * np.cos(np.linspace(phimin,phimax))

    xhi = -R[1] * np.sin(np.linspace(phimin,phimax))
    yhi = R[1] * np.cos(np.linspace(phimin,phimax))
    lines = []
    lines.append(ax.plot([xlo[0],xhi[0]],[ylo[0],yhi[0]],'w')[0])
    lines.append(ax.plot([xlo[-1],xhi[-1]],[ylo[-1],yhi[-1]],'w')[0])
    lines.append(ax.plot(xlo,ylo,'w')[0])
    lines.append(ax.plot(xhi,yhi,'w')[0])
    return lines

def drawPhiboxLabel(phimin, phimax, ax):
    return ax.text(0,-16,rf'${{{np.rad2deg(phimin):.0f}}}^\circ<\phi<{{{np.rad2deg(phimax):.0f}}}^\circ$', 
                   color='w', ha='center', va='bottom',size='xx-large',fontweight='bold')


# %%
fig, ax = plt.subplots(figsize=(5,5))
ax.imshow(img, extent=[-19, 19, -19, 19])
#ax.plot(R*np.cos(phi),R*np.sin(phi),c='r')
ax.set_aspect('equal')
ax.axis('off')
#ax.scatter(TMIN, PRCP, color="#ebb734")
ax.set_xlim(-8,8)
ax.set_ylim(-17,0)

plt.plot([0],[-R0], 'wo')


phimindeg = 150
lmin=np.deg2rad(phimindeg)
lmax=np.deg2rad(phimindeg+20)

drawPhiboxLines(lmin,lmax,ax)
drawPhiboxLabel(lmin,lmax,ax)
plt.show()

# %%
fig, ax = plt.subplots(figsize=(5,5))
ax.imshow(img, extent=[-19, 19, -19, 19])
#ax.plot(R*np.cos(phi),R*np.sin(phi),c='r')
ax.set_aspect('equal')
ax.axis('off')
#ax.scatter(TMIN, PRCP, color="#ebb734")
ax.set_xlim(-8,8)
ax.set_ylim(-17,0)

plt.plot([0],[-R0], 'wo')


phimindeg = 150
lmin=np.deg2rad(phimindeg)
lmax=np.deg2rad(phimindeg+60)

drawPhiboxLines(lmin,lmax,ax)
drawPhiboxLabel(lmin,lmax,ax)
plt.savefig(plotdir+'BasicVersion.png')
plt.show()

# %%
ims = []
fig, ax = plt.subplots(figsize=(5,5), facecolor='black')
ax.imshow(img, extent=[-19, 19, -19, 19])

n_frames = 60-1
phi_st = np.linspace(201.6,147.5,n_frames)
width = 10
for phimin in phi_st:
    lines = drawPhiboxLines(np.deg2rad(phimin),np.deg2rad(phimin+10),ax)
    #lines.append(drawLboxLabel(np.deg2rad(lmin),np.deg2rad(lmin+10),ax))
    ims.append(lines)

ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(-8,8)
ax.set_ylim(-17,0)
ax.plot([0],[-R0], 'wo')

fps = n_frames * 0.3

aniR = animation.ArtistAnimation(fig, ims, interval=50,
  blit=True,
  repeat=True,
  repeat_delay=50)
#plt.show()

aniR.save(plotdir+'testMW.gif')
plt.draw()
plt.close(fig)

# %%

# %%
ims = []
fig, ax = plt.subplots(figsize=(5,5), facecolor='black')
ax.imshow(img, extent=[-19, 19, -19, 19])

dl, width = 2,10
for lmin in np.arange(130,230+dl-width,dl):
    lines = drawLboxLines(np.deg2rad(lmin),np.deg2rad(lmin+10),ax)
    lines.append(drawLboxLabel(np.deg2rad(lmin),np.deg2rad(lmin+10),ax))
    ims.append(lines)

ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(-8,8)
ax.set_ylim(-17,0)
ax.plot([0],[-R0], 'wo')


aniR = animation.ArtistAnimation(fig, ims, interval=400, blit=True,
                                    repeat_delay=2500, repeat=True)
#plt.show()


aniR.save(plotdir+'lrange_Hurt.mp4')
plt.draw()
plt.close(fig)


# %% [markdown]
# ### Match phase spiral video (in chemical cartograpy directory)

# %%
Rcmin = 6
Rcmax = 11
Nframes = 46
Rcwidth = 0.5
dRc = (Rcmax - Rcwidth - Rcmin) / (Nframes - 1)

PHIlim = 0.10

ims = []
fig, ax = plt.subplots(figsize=(5,5), facecolor='black')
ax.imshow(img, extent=[-19, 19, -19, 19])


for Rmin in np.arange(Rcmin, Rcmax-Rcwidth+dRc, dRc):
    lines = drawPhiboxLines(np.pi - PHIlim, np.pi + PHIlim, ax, Rmin=Rmin, Rmax=Rmin+Rcwidth)
    lines.append(drawRboxLabel(Rmin,Rmin+Rcwidth,ax))
    ims.append(lines)

ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(-8,8)
ax.set_ylim(-17,0)
ax.plot([0],[-R0], 'wo')


aniR = animation.ArtistAnimation(fig, ims, interval=200, blit=True)

aniR.save(plotdir+'RrangeSpiral_Hurt.mp4')
plt.draw()
plt.close(fig)

# %%

# %%

#f=r'Movie_Rsweep_100.gif'
#Writer = animation.writers['ffmpeg']
#writergif = matplotlib.animation.ImageMagickFileWriter(fps=2)
#writervideo = Writer(fps=2,codec='mpeg4')
#writervideo = animation.FFMpegWriter(fps=fps) 
#aniR.save(f,writergif)
