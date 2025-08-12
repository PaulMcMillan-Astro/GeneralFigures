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
import matplotlib.animation as animation
from IPython.display import HTML


plt.rcParams.update({
     "axes.facecolor": "black",     
     "figure.facecolor": "black",
     "figure.edgecolor": "black", 
     "savefig.facecolor": "black",
     "savefig.edgecolor": "black"})

# Define the parameters of the pendulum
g = 9.8 # m/s^2
Npendulums = 19
L = 0.5 + np.linspace(0,1,Npendulums) # m
omega = np.sqrt(g/L) # rad/s
T = 2*np.pi/omega # s

# Define the time array
t = np.linspace(0, 2*T, 100)

# Define the position of the pendulum as a function of time
theta = 0.5*np.cos(omega*t)

# Define the position of the pendulum as a function of time
x = L*np.sin(theta)
y = -L*np.cos(theta)

# Plot the position of the pendulum as a function of time
plt.plot(t, y)
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.show()


# %%
# Under the epicycle approximation, in a simple Galactic potential
# give the position in x,y, and z as a function of time
def x(t):
    return 0.5*np.cos(omega*t)

def y(t):
    return 0.5*np.sin(omega*t)

def z(t):
    return 0.5*np.sin(omega*t)

# Define the time array
t = np.linspace(0, 2*T, 100)


# %%
# Define the parameters of the pendulum
g = 9.8 # m/s^2
Npendulums = 19
L = 0.5 + np.linspace(0,1,Npendulums) # m
omega = np.sqrt(g/L) # rad/s
T = 2*np.pi/omega # s

# Define the time array
t = np.linspace(0, 10*np.max(T), 100)

# Define the position of the pendulum as a function of time
theta = 0.1*np.cos(omega[:,np.newaxis]*t)

# Define the position of the pendulum as a function of time
x = L[:,np.newaxis]*np.sin(theta)
y = -L[:,np.newaxis]*np.cos(theta)

fig = plt.figure()
ax = plt.axes(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
ax.axis('off')

lines = []
starting_points_x = np.linspace(-0.9, 0.9, 19)
starting_points_y = -1 + L
for i in range(19):
    line, = ax.plot([], [], lw=1, c='w')
    lines.append(line)

# initialization function: plot the background of each frame
def init():
    for line in lines:
        line.set_data([], [])
    return lines

# animation function.  This is called sequentially
def animate(i):
    for j, line in enumerate(lines):
        line.set_data([starting_points_x[j], starting_points_x[j] + x[j,i]], 
                      [starting_points_y[j], starting_points_y[j] + y[j,i]])
    return lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=100, interval=100, blit=True)

plt.close()
HTML(anim.to_html5_video())

# %%
N = 30
Nframes = 100

rng = np.random.RandomState(3)

vz0 = rng.normal(size=N)
z0 = rng.normal(size=N)
r0 = (vz0**2 + z0**2)**0.5
#ind = np.argsort(r0)
#r0, z0, vz0 = r0[ind], z0[ind], vz0[ind]

phi0 = np.arctan2(vz0,z0)
Omega = 1/(1 + 0.5 * r0)
t = 10
z, vz = r0*np.cos(phi0-Omega*t),vfactor*r0*np.sin(phi0-Omega*t)

# animate z over time from t=0 to t=10
fig = plt.figure(figsize=(3,5))
ymax = 1.1*np.max(np.abs(r0))
ax = plt.axes(xlim=(-1.1, 1.1), ylim=(-ymax, ymax))
#ax.axis('off')

Dots = []
starting_points_x = np.linspace(-0.9, 0.9, N)
dt = 0.5
ax.set_facecolor('k')
ax.spines['left'].set_color('w')
#ax.yaxis.label.set_color('w')
ax.set_ylabel('z', color='w', weight='bold', fontsize=20, labelpad=-20)
ax.plot([-1,1],[0,0], 'w--', lw=0.5)


for i in range(N):
    Dot, = ax.plot([], [], lw=1, c='gold', marker='*')
    #Line, = ax.plot([starting_points_x[i], starting_points_x[i]], 
    #                [-r0[i], r0[i]], 'w', lw=0.5)
    Dots.append(Dot)

# initialization function: plot the background of each frame
def init():
    for Dot in Dots:
        Dot.set_data([], [])
    return Dots

# animation function.  This is called sequentially
def animate(i):
    for j, Dot in enumerate(Dots):
        Dot.set_data([starting_points_x[j]], 
                      [r0[j]*np.cos(phi0[j]-Omega[j] * i*dt)])
    return Dots

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=Nframes, interval=100, blit=True)

plt.close()
HTML(anim.to_html5_video())
anim.save('Movies/JustVerticalMotion.mp4')

# %%

# %%
N = 30
Nframes = 100

rng = np.random.RandomState(3)

vz0 = rng.normal(size=N)
z0 = rng.normal(size=N)
r0 = (vz0**2 + z0**2)**0.5
#ind = np.argsort(r0)
#r0, z0, vz0 = r0[ind], z0[ind], vz0[ind]

boldstars = [np.argmin(r0), np.argmax(r0)]

phi0 = np.arctan2(vz0,z0)
Omega = 1/(1 + 0.5 * r0)
t = 10
z, vz = r0*np.cos(phi0-Omega*t),30*r0*np.sin(phi0-Omega*t)

# animate z over time from t=0 to t=10
fig = plt.figure(figsize=(3,5))
ymax = 1.1*np.max(np.abs(r0))
ax = plt.axes(xlim=(-1.1, 1.1), ylim=(-ymax, ymax))
#ax.axis('off')

Dots = []
starting_points_x = np.linspace(-0.9, 0.9, N)
dt = 0.5
ax.set_facecolor('k')
ax.spines['left'].set_color('w')
#ax.yaxis.label.set_color('w')
ax.set_ylabel('z', color='w', weight='bold', fontsize=20, labelpad=-20)
ax.plot([-1,1],[0,0], 'w--', lw=0.5)


for i in range(N):
    if i in boldstars:
        Dot, = ax.plot([], [], lw=1, c='gold', marker='*', markersize=15)
        Line, = ax.plot([starting_points_x[i], starting_points_x[i]], 
                    [-r0[i], r0[i]], 'w', lw=0.5)
    else:
        Dot, = ax.plot([], [], lw=1, c='gold', marker='*', alpha=0.2)
        Line, = ax.plot([starting_points_x[i], starting_points_x[i]], 
                        [-r0[i], r0[i]], 'w', lw=0.5, alpha=0.2)
    Dots.append(Dot)

# initialization function: plot the background of each frame
def init():
    for Dot in Dots:
        Dot.set_data([], [])
    return Dots

# animation function.  This is called sequentially
def animate(i):
    for j, Dot in enumerate(Dots):
        Dot.set_data([starting_points_x[j]], 
                      [r0[j]*np.cos(phi0[j]-Omega[j] * i*dt)])
    return Dots

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=Nframes, interval=100, blit=True)

plt.close()
anim.save('Movies/VerticalMotionEmphasised.mp4')
HTML(anim.to_html5_video())

# %%
N = 30
Nframes = 100

rng = np.random.RandomState(3)

vz0 = rng.normal(size=N)
z0 = rng.normal(size=N)
r0 = (vz0**2 + z0**2)**0.5
#ind = np.argsort(r0)
#r0, z0, vz0 = r0[ind], z0[ind], vz0[ind]

boldstars = [np.argmin(r0), np.argmax(r0)]

phi0 = np.arctan2(vz0,z0)
Omega = 1/(1 + 0.5 * r0)
t = 10
z, vz = r0*np.cos(phi0-Omega*t), 30*r0*np.sin(phi0-Omega*t)

# animate z over time from t=0 to t=10
#fig = plt.figure(figsize=(3,5))
fig, ax = plt.subplots(1,2, figsize=(8,5), gridspec_kw={'width_ratios': [3, 5]})

ymax = 1.1*np.max(np.abs(r0))
ax[0].set_xlim([-1.1, 1.1])
ax[0].set_ylim([-ymax, ymax])
ax[1].set_xlim([-30 * ymax, 30 * ymax])
ax[1].set_ylim([-ymax, ymax])


Dots = []
Dots_xv = []
starting_points_x = np.linspace(-0.9, 0.9, N)
dt = 0.5
for a in ax:
    a.set_facecolor('k')
    a.spines['left'].set_color('w')
ax[1].spines['bottom'].set_color('w')
#ax.yaxis.label.set_color('w')
for a in ax:
    a.set_ylabel('z', color='w', weight='bold', fontsize=20, labelpad=-20)

ax[1].set_xlabel(r'$v_z$', color='w', weight='bold', fontsize=20, labelpad=-10)
ax[0].plot([-1,1],[0,0], 'w--', lw=0.5)
ax[1].plot([-30 * ymax, 30 * ymax],[0,0], 'w--', lw=0.5)
ax[1].plot([0,  0],[-ymax, ymax], 'w--', lw=0.5)

# for each star, draw a circle corresponding to its value of r0
z_circ, vz_circ = r0[:,np.newaxis] * np.cos(np.linspace(0, 2*np.pi, 100)[np.newaxis,:]), \
    30 * r0[:,np.newaxis] * np.sin(np.linspace(0, 2*np.pi, 100)[np.newaxis,:])


for i in range(N):
    if i in boldstars:
        Dot, = ax[0].plot([], [], lw=1, c='gold', marker='*', markersize=15)
        Line, = ax[0].plot([starting_points_x[i], starting_points_x[i]], 
                    [-r0[i], r0[i]], 'w', lw=0.5)
        Dot_xv, = ax[1].plot([], [], lw=1, c='gold', marker='*', markersize=15)
        Line_xv, = ax[1].plot(vz_circ[i,:], z_circ[i,:], 'w', lw=0.5)
    else:
        Dot, = ax[0].plot([], [], lw=1, c='gold', marker='*', alpha=0.2)
        Line, = ax[0].plot([starting_points_x[i], starting_points_x[i]], 
                        [-r0[i], r0[i]], 'w', lw=0.5, alpha=0.2)
        Dot_xv, = ax[1].plot([], [], lw=1, c='gold', marker='*', alpha=0.2)
        Line_xv = ax[1].plot(vz_circ[i,:], z_circ[i,:], 'w', lw=0.5, alpha=0.2)
    Dots.append(Dot)
    Dots_xv.append(Dot_xv)

# initialization function: plot the background of each frame
def init():
    for Dot in Dots:
        Dot.set_data([], [])
    for Dot in Dots_xv:
        Dot.set_data([], [])
    return Dots + Dots_xv

# animation function.  This is called sequentially
def animate(i):
    for j, Dot in enumerate(Dots):
        Dot.set_data([starting_points_x[j]], 
                      [r0[j]*np.cos(phi0[j]-Omega[j] * i*dt)])
    for j, Dot in enumerate(Dots_xv):
        Dot.set_data([30*r0[j]*np.sin(phi0[j]-Omega[j] * i*dt)], [r0[j]*np.cos(phi0[j]-Omega[j] * i*dt)])
    return Dots + Dots_xv

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=Nframes, interval=100, blit=True)

plt.close()
anim.save('Movies/VerticalMotionPhaseDiagram.mp4')
HTML(anim.to_html5_video())

# %%
N = 30
Nframes = 100

rng = np.random.RandomState(3)

vz0 = rng.normal(size=N)
z0 = rng.normal(size=N)
r0 = (vz0**2 + z0**2)**0.5
#ind = np.argsort(r0)
#r0, z0, vz0 = r0[ind], z0[ind], vz0[ind]

boldstars = [np.argmin(r0), np.argmax(r0)]

phi0 = np.zeros(N) # np.arctan2(vz0,z0)
Omega = 1/(1 + 0.5 * r0)
t = 10
z, vz = r0*np.cos(phi0-Omega*t), 30*r0*np.sin(phi0-Omega*t)

# animate z over time from t=0 to t=10
#fig = plt.figure(figsize=(3,5))
fig, ax = plt.subplots(1,2, figsize=(8,5), gridspec_kw={'width_ratios': [3, 5]})

ymax = 1.1*np.max(np.abs(r0))
ax[0].set_xlim([-1.1, 1.1])
ax[0].set_ylim([-ymax, ymax])
ax[1].set_xlim([-30 * ymax, 30 * ymax])
ax[1].set_ylim([-ymax, ymax])


Dots = []
Dots_xv = []
starting_points_x = np.linspace(-0.9, 0.9, N)
dt = 0.5
for a in ax:
    a.set_facecolor('k')
    a.spines['left'].set_color('w')
ax[1].spines['bottom'].set_color('w')
#ax.yaxis.label.set_color('w')
for a in ax:
    a.set_ylabel('z', color='w', weight='bold', fontsize=20, labelpad=-20)

ax[1].set_xlabel(r'$v_z$', color='w', weight='bold', fontsize=20, labelpad=-10)
ax[0].plot([-1,1],[0,0], 'w--', lw=0.5)
ax[1].plot([-30 * ymax, 30 * ymax],[0,0], 'w--', lw=0.5)
ax[1].plot([0,  0],[-ymax, ymax], 'w--', lw=0.5)

# for each star, draw a circle corresponding to its value of r0
z_circ, vz_circ = r0[:,np.newaxis] * np.cos(np.linspace(0, 2*np.pi, 100)), \
    30 * r0[:,np.newaxis] * np.sin(np.linspace(0, 2*np.pi, 100)[np.newaxis,:])


for i in range(N):
    if i in boldstars:
        Dot, = ax[0].plot([], [], lw=1, c='gold', marker='*', markersize=15)
        Line, = ax[0].plot([starting_points_x[i], starting_points_x[i]], 
                    [-r0[i], r0[i]], 'w', lw=0.5)
        Dot_xv, = ax[1].plot([], [], lw=1, c='gold', marker='*', markersize=15)
        Line_xv, = ax[1].plot(vz_circ[i,:], z_circ[i,:], 'w', lw=0.5)
    else:
        Dot, = ax[0].plot([], [], lw=1, c='gold', marker='*', alpha=0.2)
        Line, = ax[0].plot([starting_points_x[i], starting_points_x[i]], 
                        [-r0[i], r0[i]], 'w', lw=0.5, alpha=0.5)
        Dot_xv, = ax[1].plot([], [], lw=1, c='gold', marker='*', alpha=0.5)
        Line_xv = ax[1].plot(vz_circ[i,:], z_circ[i,:], 'w', lw=0.5, alpha=0.5)
    Dots.append(Dot)
    Dots_xv.append(Dot_xv)

# initialization function: plot the background of each frame
def init():
    for Dot in Dots:
        Dot.set_data([], [])
    for Dot in Dots_xv:
        Dot.set_data([], [])
    return Dots + Dots_xv

# animation function.  This is called sequentially
def animate(i):
    for j, Dot in enumerate(Dots):
        Dot.set_data([starting_points_x[j]], 
                      [r0[j]*np.cos(phi0[j]-Omega[j] * i*dt)])
    for j, Dot in enumerate(Dots_xv):
        Dot.set_data([30*r0[j]*np.sin(phi0[j]-Omega[j] * i*dt)], [r0[j]*np.cos(phi0[j]-Omega[j] * i*dt)])
    return Dots + Dots_xv

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=Nframes, interval=100, blit=True)

plt.close()
anim.save('Movies/VerticalMotionPhaseSpiral_smallN.mp4')

HTML(anim.to_html5_video())

# %%
N = 200
Nframes = 100

rng = np.random.RandomState(3)

vz0 = rng.normal(size=N)
z0 = rng.normal(size=N)
r0 = (vz0**2 + z0**2)**0.5
#ind = np.argsort(r0)
#r0, z0, vz0 = r0[ind], z0[ind], vz0[ind]

boldstars = [np.argmin(r0), np.argmax(r0)]

phi0 = np.zeros(N) # np.arctan2(vz0,z0)
Omega = 1/(1 + 0.5 * r0)
t = 10
z, vz = r0*np.cos(phi0-Omega*t), 30*r0*np.sin(phi0-Omega*t)

# animate z over time from t=0 to t=10
#fig = plt.figure(figsize=(3,5))
fig, ax = plt.subplots(1,2, figsize=(8,5), gridspec_kw={'width_ratios': [3, 5]})

ymax = 1.1*np.max(np.abs(r0))
ax[0].set_xlim([-1.1, 1.1])
ax[0].set_ylim([-ymax, ymax])
ax[1].set_xlim([-30 * ymax, 30 * ymax])
ax[1].set_ylim([-ymax, ymax])


Dots = []
Dots_xv = []
starting_points_x = np.linspace(-0.9, 0.9, N)
dt = 0.5
for a in ax:
    a.set_facecolor('k')
    a.spines['left'].set_color('w')
ax[1].spines['bottom'].set_color('w')
#ax.yaxis.label.set_color('w')
for a in ax:
    a.set_ylabel('z', color='w', weight='bold', fontsize=20, labelpad=-20)

ax[1].set_xlabel(r'$v_z$', color='w', weight='bold', fontsize=20, labelpad=-10)
ax[0].plot([-1,1],[0,0], 'w--', lw=0.5)
ax[1].plot([-30 * ymax, 30 * ymax],[0,0], 'w--', lw=0.5)
ax[1].plot([0,  0],[-ymax, ymax], 'w--', lw=0.5)

# for each star, draw a circle corresponding to its value of r0
z_circ, vz_circ = r0[:,np.newaxis] * np.cos(np.linspace(0, 2*np.pi, 100)[np.newaxis,:]), \
    30 * r0[:,np.newaxis] * np.sin(np.linspace(0, 2*np.pi, 100)[np.newaxis,:])


for i in range(N):
    Dot, = ax[0].plot([], [], lw=1, c='gold', marker='*', alpha=0.5)
    Dot_xv, = ax[1].plot([], [], lw=1, c='gold', marker='*', alpha=0.5)
    Dots.append(Dot)
    Dots_xv.append(Dot_xv)

# initialization function: plot the background of each frame
def init():
    for Dot in Dots:
        Dot.set_data([], [])
    for Dot in Dots_xv:
        Dot.set_data([], [])
    return Dots + Dots_xv

# animation function.  This is called sequentially
def animate(i):
    for j, Dot in enumerate(Dots):
        Dot.set_data([starting_points_x[j]], 
                      [r0[j]*np.cos(phi0[j]-Omega[j] * i*dt)])
    for j, Dot in enumerate(Dots_xv):
        Dot.set_data([30*r0[j]*np.sin(phi0[j]-Omega[j] * i*dt)], [r0[j]*np.cos(phi0[j]-Omega[j] * i*dt)])
    return Dots + Dots_xv

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=Nframes, interval=100, blit=True)

plt.close()
anim.save('Movies/VerticalMotionPhaseSpiral_largeN.mp4')
HTML(anim.to_html5_video())

# %% [markdown]
# ## Phase-spiral - Simpler version

# %%
N = 1000
Nframes = 100
jumpFrame = 30


rng = np.random.RandomState(3)

vz0 = rng.normal(size=N)
z0 = rng.normal(size=N)
r0 = (vz0**2 + z0**2)**0.5
#ind = np.argsort(r0)
#r0, z0, vz0 = r0[ind], z0[ind], vz0[ind]

boldstars = [np.argmin(r0), np.argmax(r0)]

phi0 = np.arctan2(vz0,z0)
Omega = 1/(1 + 0.5 * r0)
t = 10
dt = 0.5
# z, vz = r0*np.cos(phi0-Omega*t), 30*r0*np.sin(phi0-Omega*t)
z0b = r0*np.cos(phi0-Omega * jumpFrame*dt) - 1
vz0b = r0*np.sin(phi0-Omega * jumpFrame*dt)
r0b = (vz0b**2 + z0b**2)**0.5
phi0b = np.arctan2(vz0b,z0b)
Omegab = 1/(1 + 0.5 * r0b)
# animate z over time from t=0 to t=10
#fig = plt.figure(figsize=(3,5))
fig, ax = plt.subplots(1,2, figsize=(8,5), gridspec_kw={'width_ratios': [3, 5]})

ymax = 1.1*np.max(np.abs(r0))
ax[0].set_xlim([-1.1, 1.1])
ax[0].set_ylim([-ymax, ymax])
ax[1].set_xlim([-30 * ymax, 30 * ymax])
ax[1].set_ylim([-ymax, ymax])


Dots = []
Dots_xv = []
Texts = []
starting_points_x = np.linspace(-0.9, 0.9, N)

for a in ax:
    a.set_facecolor('k')
    a.spines['left'].set_color('w')
ax[1].spines['bottom'].set_color('w')
#ax.yaxis.label.set_color('w')
for a in ax:
    a.set_ylabel('z', color='w', weight='bold', fontsize=20, labelpad=-20)

ax[1].set_xlabel(r'$v_z$', color='w', weight='bold', fontsize=20, labelpad=-10)
ax[0].plot([-1,1],[0,0], 'w--', lw=0.5)
ax[1].plot([-30 * ymax, 30 * ymax],[0,0], 'w--', lw=0.5)
ax[1].plot([0,  0],[-ymax, ymax], 'w--', lw=0.5)


Dot, = ax[0].plot([], [], lw=1, c='gold', marker='*', alpha=0.5, linestyle='None')
Dot_xv, = ax[1].plot([], [], lw=1, c='gold', marker='*', alpha=0.5, linestyle='None')
Text1 = fig.text(0.35, 0.8, '', color='w', weight='bold', fontsize=40,
                   bbox=dict(facecolor='black', alpha=0.5))
Dots.append(Dot)
Dots_xv.append(Dot_xv)
Texts.append(Text1)

# initialization function: plot the background of each frame
def init():
    for Dot in Dots:
        Dot.set_data([], [])
    for Dot in Dots_xv:
        Dot.set_data([], [])
    return Dots + Dots_xv


# animation function.  This is called sequentially
def animate(i):
    if i<jumpFrame:
        Dots[0].set_data(starting_points_x, r0*np.cos(phi0-Omega * i*dt))
        Dots_xv[0].set_data(30*r0*np.sin(phi0-Omega * i*dt), r0*np.cos(phi0-Omega * i*dt))
    else:
        if i==jumpFrame:
            Texts[0].set_text('BANG!')
        if i==jumpFrame+20:
            Texts[0].set_text('')
        for j, Dot in enumerate(Dots):
            Dot.set_data(starting_points_x, 
                        r0b*np.cos(phi0b-Omegab * (i-jumpFrame)*dt))
        for j, Dot in enumerate(Dots_xv):
            Dot.set_data(30*r0b*np.sin(phi0b-Omegab * (i-jumpFrame)*dt), r0b*np.cos(phi0b-Omegab * (i-jumpFrame)*dt))
    return Dots + Dots_xv

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=Nframes, interval=100, blit=True)

plt.close()
anim.save('Movies/VerticalMotionPhaseSpiral_realism.mp4')
HTML(anim.to_html5_video())

# %%
N = 1000
Nframes = 150
jumpFrame = 30

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

rng = np.random.RandomState(3)

vz0 = rng.normal(size=[4,N])*0.3
z0 = rng.normal(size=[4,N])*0.3
r0 = (vz0**2 + z0**2)**0.5
vfactor = 60

rescale_Omega = np.array([1.4, 1, 0.7, 0.4])

phi0 = np.arctan2(vz0,z0)
Omega = rescale_Omega[:,np.newaxis] / (1 + 0.5 * r0 )
t = 10
dt = 0.5
# z, vz = r0*np.cos(phi0-Omega*t), 30*r0*np.sin(phi0-Omega*t)
z0b = r0*np.cos(phi0-Omega * jumpFrame*dt) - 0.3
vz0b = r0*np.sin(phi0-Omega * jumpFrame*dt)
r0b = (vz0b**2 + z0b**2)**0.5
phi0b = np.arctan2(vz0b,z0b)
Omegab = rescale_Omega[:,np.newaxis]/(1 + 0.5 * r0b )
# animate z over time from t=0 to t=10
#fig = plt.figure(figsize=(3,5))
fig, ax = plt.subplots(1,4, figsize=(20,5))

xmax = 1.1*np.max(np.abs(r0))
for a in ax: 
    #a.set_xlim([-xmax, xmax])
    #a.set_ylim([-vfactor * rescale_Omega[0] * xmax, vfactor * rescale_Omega[0] * xmax])
    a.set_xlim([-1, 1])
    a.set_ylim([-60, 60])    

Dots_xv = []

ax[0].set_ylabel('Vz', color='w', weight='bold', fontsize=20, labelpad=0)

for a in ax:
    a.set_xlabel('z', color='w', weight='bold', fontsize=20, labelpad=0)
    a.plot([0,0], [-vfactor * rescale_Omega[0] * xmax, vfactor * rescale_Omega[0] * xmax], 'w--', lw=0.5)
    a.plot([-xmax, xmax], [0,0], 'w--', lw=0.5)
    Dot_xv, = a.plot([], [], lw=1, c='gold', marker='*', alpha=0.5, linestyle='None')
    Dots_xv.append(Dot_xv)


# initialization function: plot the background of each frame
def init():
    for Dot in Dots_xv:
        Dot.set_data([], [])
    return Dots_xv


# animation function.  This is called sequentially
def animate(i):
    if i<jumpFrame:
        for j, Dot in enumerate(Dots_xv):
            Dot.set_data(r0[j,:]*np.cos(phi0[j,:]-Omega[j,:] * i*dt), 
                         vfactor*rescale_Omega[j]*r0[j,:]*np.sin(phi0[j,:]-Omega[j,:] * i*dt))
    else:
        for j, Dot in enumerate(Dots_xv):
            Dot.set_data(r0b[j,:]*np.cos(phi0b[j,:]-Omegab[j,:] * (i-jumpFrame)*dt),
                         vfactor*rescale_Omega[j]*r0b[j,:]*np.sin(phi0b[j,:]-Omegab[j,:] * (i-jumpFrame)*dt))
    return Dots_xv

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                    frames=Nframes, interval=100, blit=True)

plt.close()
anim.save('Movies/VerticalMotionPhaseSpiral_quartet2.mp4')
HTML(anim.to_html5_video())
