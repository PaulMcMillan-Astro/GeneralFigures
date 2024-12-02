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
import matplotlib.animation as animation
from IPython.display import display, HTML



# %% [markdown]
# ## Introduction
#
# The Milky Way, like many disc galaxies, is warped
#
# <img src="images/ugc3697fullcolor.jpg" alt="UGC 3697 - a warped galaxy" width="450" height="400">
#
# [Credit: NRAO/AUI/NSF, Image copyright J. M. Uson (NRAO), observers L. D. Matthews (CfA), J. M. Uson (NRAO)](https://public.nrao.edu/gallery/warped-disk-of-galaxy-ugc-3697-2/)
#
# This is an extreme example seen in the stars (white) and neutral gas (blue) of UGC3697
#
# As ever when something is widely observed in astronomy, there are two possible reasons:
#
# 1. It is very easy to create a warp
# 2. Warps are long-lived structures, so if you get one it stays around
#
# Our starting assumption is that 2 is at least partly true
#
#

# %% [markdown]
# ## The basic model - tilted rings
#
# The simplest way to view this is making the approximation that stars are rotating in circles around the galactic centre, with a set of them spread around the galaxy that we can consider as separate rings around the galaxy.  
#
# However, in a warped galaxy these rings are tilted with respect to one another. The central region of the galaxy is a flat disc, but the rings start to tilt in the outer galaxy (beyond about 10 kpc for the Milky Way)
#
# <img src="images/Tilted rings.png" alt="Tilted rings" width="400" height="400">
#
#

# %% [markdown]
# ### The line of nodes
#
# The line along which the rings go through $z=0$ (i.e., the plane that includes the flat central part of the disc) is called the line of nodes, which we will give the symbol $\phi_\mathrm{LON}$.

# %% [markdown]
# ### A non-precessing warp
#
# If the warp is fixed and not moving, with stars orbiting around those rings then there is a very simple relationship between position and vertical velocity - stars have to go upwards as the head towards the top of the warp, and downwards towards the bottom of the warp.
#
# We can approximate the warp as a simple power-law in radius (outside a starting radius, $R_0$) and then given that we have tilted rings we can write the z position of any point on the warp as being
# $$
# z(R,\phi) = \begin{cases}
#     0 & \text{if } z < z0  \\ 
#     z_s (R-R_0)^\alpha \sin(\phi-\phi_\mathrm{LON})& \text{otherwise.} 
# \end{cases}
# $$
# with the power-law of the slope being $\alpha$ which is often taken to be $1$
#
# We then have to think about the star moving around this warp. If we assume that it is rotating on this ring with the $\phi$ component of velocity being $v_{\phi,c} = \mathrm{const}$, then we have a velocity $v_{z,*}$ for any star being
#
#  $$v_{z,*}(R,\phi) = \frac{\mathrm{d}z}{\mathrm{d}\phi} \frac{\mathrm{d}\phi}{\mathrm{d}t} =  \begin{cases}
#     0 & \text{if } z < z0  \\ z_s (R-R_0)^\alpha \cos(\phi-\phi_\mathrm{LON})\; v_{\phi,c}/R & \text{otherwise.} 
# \end{cases}
# $$
#
#

# %% [markdown]
# #### A video
#
# We can produce a simple video of how this looks
#
# <video width="400" height="400" 
#        src="Warp Animations/non-precessing warp.mp4"  
#        controls>
# </video>

# %% [markdown]
# ### A precessing warp
#
# What if the warp is precessing? For simplicity, we will look at the case where the precession is at a constant angular speed $\omega_w$, i.e., the warp moves like a solid body. Note that the speed with which the warp precesses is not related to the velocity of stars $v_{\phi,*}$.
#
# We therefore have
# $$
# z(R,\phi) = \begin{cases}
#     0 & \text{if } z < z0  \\ 
#     z_s (R-R_0)^\alpha \sin(\phi-\phi_\mathrm{LON,0}-\omega_w t)& \text{otherwise.} 
# \end{cases}
# $$
# where we now have $\phi_\mathrm{LON,0} = \phi_{LON}(t=0)$
#
# So now $z(R,\phi)$ is itself is a function of time, so our differentiation gets an extra term
#
#
#  $$v_{z,*}(R,\phi) = \frac{\partial z}{\partial\phi} \frac{\mathrm{d}\phi}{\mathrm{d}t} + \frac{\partial z}{\partial t} =  \begin{cases}
#     0 & \text{if } z < z0  \\ z_s (R-R_0)^\alpha \cos(\phi-\phi_\mathrm{LON})\; (v_{\phi,c}/R - \omega_w) & \text{otherwise.} 
# \end{cases}
# $$

# %% [markdown]
# #### A video
#
# Again, we can show what 
#
# <video width="400" height="400" 
#        src="Warp Animations/precessing warp.mp4"  
#        controls>
# </video>

# %%
alpha_warp = 1.1
# 1.5 with z0=0.4 



def warp_height(R, phi, R0=8., z0=0.9, phi0=np.pi):
    '''Gives height of simple power-law warp
    
    phi & phi0 are in radians
    phi0 is the line of nodes

    z = 0 [for z < R0]
    z = z0 * (R-R0) * sin(phi-phi0) [for z>= R0]
    '''
    z = np.zeros_like(R)
    z[R>R0] = z0*(R[R>R0]-R0)**alpha_warp * np.sin(phi[R>R0]-phi0)
    return z

def dphidt(R):
    '''dphi/dt for stars with flat rotation curve
        
       dphi/dt = -12 deg/unit at R=12'''
    return -12. / R * np.pi/180

def warp_velocity(R, phi, R0=8., z0=0.6, phi0=np.pi, vphi_scale=500):
    
    vz = np.zeros_like(R)
    vz[R>R0] = (z0*(R[R>R0]-R0)**alpha_warp * np.cos(phi[R>R0]-phi0) * dphidt(R[R>R0])) * vphi_scale
    return vz

def precessing_warp_height(R, phi, R0=8., z0=0.9, phi0=np.pi, omega=0, t=0):
    z = np.zeros_like(R)
    z[R>R0] = z0*(R[R>R0]-R0)**alpha_warp * np.sin(phi[R>R0]-phi0-omega*t)
    return z

def precessing_warp_velocity(R, phi, R0=8., z0=0.9, phi0=np.pi, vphi_scale=500, omega=0, t=0):
    vz = np.zeros_like(R)
    vz[R>R0] = z0*(R[R>R0]-R0)**alpha_warp * np.cos(phi[R>R0] - phi0 - omega*t) * (dphidt(R[R>R0]) - omega) * vphi_scale
    return vz


def accelerating_precessing_warp(R, phi, R0=8., z0=0.9, phi0=np.pi, vphi_scale=500, omega0=0, omega_max=10, t0=20, tstop=200, t=0):
    z = np.zeros_like(R)
    vz = np.zeros_like(R)
    if t<t0:
        omega = omega0
        lon = phi0 + omega0*t
    elif t<tstop:
        omega = omega0 + (omega_max-omega0) * (t-t0)/(tstop-t0)
        lon = phi0 + omega0 * t0 + (0.5 * t**2 - t0 * t)/(tstop-t0) * (omega_max-omega0)
    else:  
        omega = omega_max
        lon = phi0 + omega0 * t0 + (0.5 * tstop**2 - t0 * tstop)/(tstop-t0) * (omega_max-omega0) + omega_max * (t-tstop)


    z[R>R0] = z0*(R[R>R0]-R0)**alpha_warp * np.sin(phi[R>R0]-lon)
    vz[R>R0] = z0*(R[R>R0]-R0)**alpha_warp * np.cos(phi[R>R0] - phi0 - omega*t) * (dphidt(R[R>R0]) - omega) * vphi_scale
    z[R>R0] += alpha * t**2
    vz[R>R0] += 2*alpha*t
    return z, vz


omega = -np.pi/220.


R_ring = np.linspace(0, 15, 20)
R_star = np.linspace(0, 15, 10)[3:]
phi_ring = np.linspace(0, 2*np.pi, 180)
phi_star = np.zeros_like(R_star)
R_ring, phi_ring = np.meshgrid(R_ring, phi_ring, indexing='ij')
x_ring = R_ring * np.cos(phi_ring)
y_ring = R_ring * np.sin(phi_ring)
z_ring = np.zeros_like(x_ring)

# For R>8 set z = 0.3*(R-8)^alpha_warp * sin(phi)
z_ring = warp_height(R_ring, phi_ring)



# %%
ax = plt.axes(projection='3d')
for i in range(x_ring.shape[0]):
    ax.plot3D(x_ring[i], y_ring[i], z_ring[i], 'w-')

#ax.plot3D(x.flatten(), y.flatten(), z.flatten(), 'k.')
ax.set_aspect('equal')

ax.set_axis_off()
ax.view_init(elev=20, azim=0)

plt.show()

# %% [markdown]
# ### Basic side-on view

# %%
plt.scatter(y_ring, z_ring, c=R_ring, cmap='viridis')
plt.colorbar(label='z')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.show()

# %% [markdown]
# ### Planned viewing angle & z colour bar as surface

# %%
surf = plt.axes(projection='3d').plot_surface(x_ring, y_ring, z_ring, cmap='RdYlBu')
plt.gca().set_aspect('equal')
plt.gca().view_init(elev=20, azim=0)

plt.colorbar(mappable=surf, label='z')

plt.xlabel('x')
plt.ylabel('y')
plt.show()

# %% [markdown]
# ### Ring view

# %%
ax = plt.axes(projection='3d')
for i in range(x_ring.shape[0]):
    ax.plot3D(x_ring[i], y_ring[i], z_ring[i], 'k-')

R_lon = np.linspace(-15, 15, 10)
phi_lon = np.pi * np.ones_like(R_lon)

x_lon = R_lon * np.cos(phi_lon)
y_lon = R_lon * np.sin(phi_lon)
z_lon = np.zeros_like(R_lon)

ax.plot3D(x_lon, y_lon, z_lon, 'k:')

phi_star = np.zeros_like(R_star) - np.pi/4
ax.scatter(R_star*np.cos(phi_star), R_star*np.sin(phi_star), warp_height(R_star, phi_star, phi0=np.pi), marker='*', c='w', s=20)
#ax.plot3D(x.flatten(), y.flatten(), z.flatten(), 'k.')
ax.set_aspect('equal')

ax.view_init(elev=20, azim=0)


#for angle in range(0, 360):
#   ax.view_init(angle, 30)
#   plt.draw()
#   plt.pause(.001)


plt.show()

# %% [markdown]
# ### Vary viewing angle movie

# %%
ax = plt.axes(projection='3d')
for i in range(x_ring.shape[0]):
    ax.plot3D(x_ring[i], y_ring[i], z_ring[i], 'w-')

#ax.plot3D(x.flatten(), y.flatten(), z.flatten(), 'k.')
ax.set_aspect('equal')

ax.set_axis_off()

fig = ax.figure

def animate(i):
    ax.view_init(elev=20, azim=5*i)
    return fig,

#for angle in range(0, 360):
#   ax.view_init(angle, 30)
#   plt.draw()
#   plt.pause(.001)
ani = animation.FuncAnimation(ax.figure, animate, frames=72, interval=50, blit=True)

plt.draw()
plt.close(fig)
display(HTML(ani.to_html5_video()))

# %% [markdown]
# ## Animated stars non-precessing
#
# One star per ring

# %% [markdown]
# ### with colour according to z

# %%
ax = plt.axes(projection='3d')
for i in range(x_ring.shape[0]):
    ax.plot3D(x_ring[i], y_ring[i], z_ring[i], 'w-', lw=0.5)

sca = ax.scatter3D(R_star*np.cos(phi_star), R_star*np.sin(phi_star), warp_height(R_star, phi_star, phi0=np.pi), marker='*', c=warp_height(R_star, phi_star, phi0=np.pi), cmap='RdBu', vmin=-1, vmax=1, s=120)

#ax.plot3D(x.flatten(), y.flatten(), z.flatten(), 'k.')
ax.set_aspect('equal')

ax.set_axis_off()

fig = ax.figure

def animate(i):
    #sca.set_data(R_star*np.cos(phi_star + i*np.pi/36), R_star*np.sin(phi_star + i*np.pi/36))
    #sca.set_3d_properties(warp_height(R_star, phi_star + i*np.pi/36), 'z')
    dphi = (12. / R_star) * i*np.pi/36
    sca._offsets3d = (R_star*np.cos(phi_star + dphi), R_star*np.sin(phi_star + dphi), warp_height(R_star, phi_star + dphi))
    sca.set_array(warp_height(R_star, phi_star + dphi))
    ax.view_init(elev=30, azim=0)
    return fig,

#for angle in range(0, 360):
#   ax.view_init(angle, 30)
#   plt.draw()
#   plt.pause(.001)
ani = animation.FuncAnimation(ax.figure, animate, frames=720, interval=50, blit=True)

plt.draw()
plt.close(fig)
display(HTML(ani.to_html5_video()))

# %% [markdown]
# ### Many stars per ring

# %%
R_star = np.linspace(0, 15, 10)[3:]
phi_star = np.linspace(0, 2*np.pi, 8)
R_star, phi_star = np.meshgrid(R_star, phi_star, indexing='ij')
R_star = R_star.flatten()
phi_star = phi_star.flatten()




ax = plt.axes(projection='3d')
for i in range(x_ring.shape[0]):
    ax.plot3D(x_ring[i], y_ring[i], z_ring[i], 'w-', lw=0.5)

sca = ax.scatter3D(R_star*np.cos(phi_star), R_star*np.sin(phi_star), warp_height(R_star, phi_star), marker='*', c=warp_velocity(R_star, phi_star), cmap='bwr', s=120)

#ax.plot3D(x.flatten(), y.flatten(), z.flatten(), 'k.')
ax.set_aspect('equal')
plt.colorbar(sca, label='vz')
plt.show()

# %%

R_star = np.linspace(0, 15, 10)[3:]
phi_star = np.linspace(0, 2*np.pi, 16)
R_star, phi_star = np.meshgrid(R_star, phi_star, indexing='ij')
R_star = R_star.flatten()
phi_star = phi_star.flatten()
plt.figure(figsize=(10, 10))

ax = plt.axes(projection='3d')
for i in range(x_ring.shape[0]):
    ax.plot3D(x_ring[i], y_ring[i], z_ring[i], 'w-', lw=0.5)

ax.plot3D(x_lon, y_lon, z_lon, 'w:')

sca = ax.scatter3D(R_star*np.cos(phi_star), R_star*np.sin(phi_star), warp_height(R_star, phi_star, phi0=np.pi), marker='*', c=warp_velocity(R_star, phi_star, phi0=np.pi), cmap='bwr', vmin=-20, vmax=20, s=120)

#ax.plot3D(x.flatten(), y.flatten(), z.flatten(), 'k.')
ax.set_aspect('equal')

ax.set_axis_off()

fig = ax.figure

def animate(i):
    #sca.set_data(R_star*np.cos(phi_star + i*np.pi/36), R_star*np.sin(phi_star + i*np.pi/36))
    #sca.set_3d_properties(warp_height(R_star, phi_star + i*np.pi/36), 'z')
    dphi = dphidt(R_star) * i
    sca._offsets3d = (R_star*np.cos(phi_star + dphi), R_star*np.sin(phi_star + dphi), warp_height(R_star, phi_star + dphi, phi0=np.pi))
    sca.set_array(warp_velocity(R_star, phi_star + dphi, phi0=np.pi))
    ax.view_init(elev=20, azim=0)
    return fig,

#for angle in range(0, 360):
#   ax.view_init(angle, 30)
#   plt.draw()
#   plt.pause(.001)
ani = animation.FuncAnimation(ax.figure, animate, frames=720, interval=50, blit=True)

plt.draw()
plt.close(fig)
display(HTML(ani.to_html5_video()))

# %%

R_star = np.linspace(0, 15, 10)[3:]
phi_star = np.linspace(0, 2*np.pi, 16)
R_star, phi_star = np.meshgrid(R_star, phi_star, indexing='ij')
R_star = R_star.flatten()
phi_star = phi_star.flatten()



plt.figure(figsize=(10, 10))

ax = plt.axes(projection='3d')
for i in range(x_ring.shape[0]):
    ax.plot3D(x_ring[i], y_ring[i], z_ring[i], 'w-', lw=0.5)

sca = ax.scatter3D(R_star*np.cos(phi_star), R_star*np.sin(phi_star), warp_height(R_star, phi_star, phi0=np.pi), marker='*', c=warp_velocity(R_star, phi_star, phi0=np.pi), cmap='bwr', vmin=-1, vmax=1, s=120)

#ax.plot3D(x.flatten(), y.flatten(), z.flatten(), 'k.')
ax.set_aspect('equal')
ax.view_init(elev=90, azim=0)

ax.set_axis_off()

fig = ax.figure

def animate(i):
    #sca.set_data(R_star*np.cos(phi_star + i*np.pi/36), R_star*np.sin(phi_star + i*np.pi/36))
    #sca.set_3d_properties(warp_height(R_star, phi_star + i*np.pi/36), 'z')
    dphi = dphidt(R_star) * i
    sca._offsets3d = (R_star*np.cos(phi_star + dphi), R_star*np.sin(phi_star + dphi), warp_height(R_star, phi_star + dphi, phi0=np.pi))
    sca.set_array(warp_velocity(R_star, phi_star + dphi, phi0=np.pi))
    return fig,

#for angle in range(0, 360):
#   ax.view_init(angle, 30)
#   plt.draw()
#   plt.pause(.001)
ani = animation.FuncAnimation(ax.figure, animate, frames=720, interval=50, blit=True)

plt.draw()
plt.close(fig)
display(HTML(ani.to_html5_video()))

# %% [markdown]
# ### Plan views

# %%
surf = plt.axes(projection='3d').plot_surface(x_ring, y_ring, z_ring, cmap='PuOr')
plt.gca().set_aspect('equal')
plt.gca().view_init(elev=90, azim=0)
plt.gca().set_axis_off()

plt.colorbar(mappable=surf, label='z', shrink=0.6)

plt.show()

# %%
vz_ring = warp_velocity(R_ring, phi_ring, phi0=np.pi)

surf = plt.axes(projection='3d').plot_surface(x_ring, y_ring, vz_ring/5, cmap='bwr')
plt.gca().set_aspect('equal')
plt.gca().view_init(elev=90, azim=0)
plt.gca().set_axis_off()

cbar = plt.colorbar(mappable=surf, label='vz [km/s]', shrink=0.6, ticks=[-5,0,5])
cbar.ax.set_yticklabels(['-25', '0', '25'])  # vertically oriented colorbar

plt.show()

# %% [markdown]
# # Precessing warp

# %% [markdown]
# ### Coloured by $z$

# %%

plt.figure(figsize=(10, 10))

ax = plt.axes(projection='3d')

ringlist = []
for i in range(x_ring.shape[0]):
    ringlist.append(ax.plot3D(x_ring[i], y_ring[i], z_ring[i], 'w-', lw=0.5)[0])

sca = ax.scatter3D(R_star*np.cos(phi_star), R_star*np.sin(phi_star), warp_height(R_star, phi_star), marker='*', c=warp_height(R_star, phi_star), cmap='RdBu', vmin=-1, vmax=1, s=120)

#ax.plot3D(x.flatten(), y.flatten(), z.flatten(), 'k.')
ax.set_aspect('equal')
ax.view_init(elev=30, azim=0)

ax.set_axis_off()

fig = ax.figure

def animate(i):
    z_ring_t = precessing_warp_height(R_ring, phi_ring, omega=omega, t=i)
    for j in range(x_ring.shape[0]):
        #print()
        ringlist[j].set_data(x_ring[j], y_ring[j])
        ringlist[j].set_3d_properties(z_ring_t[j], 'z')
    dphi = dphidt(R_star) * i
    sca._offsets3d = (R_star*np.cos(phi_star + dphi), R_star*np.sin(phi_star + dphi), precessing_warp_height(R_star, phi_star + dphi, omega=omega, t=i))
    sca.set_array(precessing_warp_height(R_star, phi_star + dphi, omega=omega, t=i))
    return fig,

ani = animation.FuncAnimation(ax.figure, animate, frames=120, interval=50, blit=True)

plt.draw()
plt.close(fig)
display(HTML(ani.to_html5_video()))

# %% [markdown]
# ### Colour by $v_z$

# %%

plt.figure(figsize=(10, 10))

R_lon = np.linspace(-15, 15, 10)
phi_lon = np.pi * np.ones_like(R_lon)

def lon_xy(R, phi):
    return R * np.cos(phi), R * np.sin(phi)
x_lon = R_lon * np.cos(phi_lon)
y_lon = R_lon * np.sin(phi_lon)
z_lon = np.zeros_like(R_lon)

omega = -np.pi/200.


ax = plt.axes(projection='3d')

ringlist = []
for i in range(x_ring.shape[0]):
    ringlist.append(ax.plot3D(x_ring[i], y_ring[i], z_ring[i], 'w-', lw=0.5)[0])

lon = ax.plot3D(x_lon, y_lon, z_lon, 'w:')[0]


sca = ax.scatter3D(R_star*np.cos(phi_star), R_star*np.sin(phi_star), warp_height(R_star, phi_star), marker='*', c=precessing_warp_velocity(R_star, phi_star, omega=omega, t=i), cmap='bwr', vmin=-5, vmax=5, s=120)

#ax.plot3D(x.flatten(), y.flatten(), z.flatten(), 'k.')
ax.set_aspect('equal')
ax.view_init(elev=30, azim=0)

ax.set_axis_off()

fig = ax.figure

def animate(i):
    z_ring_t = precessing_warp_height(R_ring, phi_ring, omega=omega, t=i)
    for j in range(x_ring.shape[0]):
        #print()
        ringlist[j].set_data(x_ring[j], y_ring[j])
        ringlist[j].set_3d_properties(z_ring_t[j], 'z')
    dphi = dphidt(R_star) * i
    x_lon, y_lon = lon_xy(R_lon, phi_lon + omega*i)
    lon.set_data(x_lon, y_lon)
    lon.set_3d_properties(z_lon, 'z')
    sca._offsets3d = (R_star*np.cos(phi_star + dphi), R_star*np.sin(phi_star + dphi), precessing_warp_height(R_star, phi_star + dphi, omega=omega, t=i))
    sca.set_array(precessing_warp_velocity(R_star, phi_star + dphi, omega=omega, t=i))
    return fig,

ani = animation.FuncAnimation(ax.figure, animate, frames=845, interval=50, blit=True)

plt.draw()
plt.close(fig)
display(HTML(ani.to_html5_video()))

# %% [markdown]
# ### Plan view

# %%


R_ring2 = np.linspace(0, 15, 50)
phi_ring2 = np.linspace(0, 2*np.pi, 180)
R_ring2, phi_ring2 = np.meshgrid(R_ring2, phi_ring2, indexing='ij')
x_ring2 = R_ring2 * np.cos(phi_ring2)
y_ring2 = R_ring2 * np.sin(phi_ring2)
z_ring2 = np.zeros_like(x_ring2)
z_ring2 = precessing_warp_height(R_ring2, phi_ring2, phi0=np.pi, omega = -np.pi/200., t=0)
vz_ring2 = precessing_warp_velocity(R_ring2, phi_ring2, phi0=np.pi, omega = -np.pi/200., t=0)


ax = plt.axes(projection='3d')

surf = ax.plot_surface(x_ring2, y_ring2, vz_ring2, cmap='bwr', zorder=1)


lon = ax.plot3D(x_lon, y_lon, z_lon+2, 'w:', lw=3, zorder=6)


ax.set_aspect('equal')
ax.view_init(elev=90, azim=0)
ax.set_axis_off()


plt.colorbar(mappable=surf, label='vz [km/s]', shrink=0.6)

plt.show()

# %%
fig = plt.figure(figsize=(6,4))


R_ring2 = np.linspace(0, 15, 50)
phi_ring2 = np.linspace(0, 2*np.pi, 180)
R_ring2, phi_ring2 = np.meshgrid(R_ring2, phi_ring2, indexing='ij')
x_ring2 = R_ring2 * np.cos(phi_ring2)
y_ring2 = R_ring2 * np.sin(phi_ring2)
z_ring2 = np.zeros_like(x_ring2)
z_ring2 = precessing_warp_height(R_ring2, phi_ring2, phi0=np.pi, omega = -np.pi/200., t=0)
vz_ring2 = precessing_warp_velocity(R_ring2, phi_ring2, phi0=np.pi, omega = -np.pi/200., t=0)


ax = plt.axes(projection='3d')

surf = ax.plot_surface(x_ring2, y_ring2, vz_ring2, cmap='bwr', zorder=1)


lon = ax.plot3D(x_lon, y_lon, z_lon+2, 'w:', lw=3, zorder=6)


ax.set_aspect('equal')
ax.view_init(elev=90, azim=0)
ax.set_axis_off()


plt.colorbar(mappable=surf, label='vz [km/s]', shrink=0.6)



def animate(i):
    vz_ring2_t = precessing_warp_velocity(R_ring2, phi_ring2, phi0=np.pi, omega = -np.pi/200., t=i)
    #surf.set_data(x_ring2, y_ring2)
    #surf.set_3d_properties(z_ring2_t, 'z')
    ax.clear()
    surf = ax.plot_surface(x_ring2, y_ring2, vz_ring2_t, cmap='bwr', zorder=1)
    x_lon, y_lon = lon_xy(R_lon, phi_lon + omega*i)
    lon = ax.plot3D(x_lon, y_lon, z_lon+2, 'w:', lw=3, zorder=6)
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.view_init(elev=90, azim=0)
    return fig,

ani = animation.FuncAnimation(fig, animate, frames=845, interval=50, blit=True)

plt.draw()
plt.close(fig)
display(HTML(ani.to_html5_video()))

# %% [markdown]
# ## Speeding up precession

# %% [markdown]
#
