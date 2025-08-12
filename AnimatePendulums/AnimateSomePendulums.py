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
# Give equation for position of pendulum as a function of time
# Animate the pendulum

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

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
     'ytick.direction' :'in',})


# Define the parameters of the pendulum
g = 9.8 # m/s^2
L = 1.0 # m
omega = np.sqrt(g/L) # rad/s
T = 2*np.pi/omega # s

# Define the time array
t = np.linspace(0, 2*T, 100)

# Define the position of the pendulum as a function of time
theta = 0.1*np.cos(omega*t)

# Define the position of the pendulum as a function of time
x = L*np.sin(theta)
y = -L*np.cos(theta)

# Plot the position of the pendulum as a function of time
plt.plot(t, y)
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.show()


# %%
# Now animate many identical pendulums with different initial conditions
# turn this into an animation
fig = plt.figure()
ax = plt.axes(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
lines = []
starting_points = np.linspace(-0.9, 0.9, 19)
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
        line.set_data([starting_points[j], starting_points[j] + x[i]], 
                      [0, y[i]])
    return lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=100, interval=20, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html

#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.close()
HTML(anim.to_html5_video())


# %% [markdown]
# ## Animate with coherent increase in length

# %%
# Define the parameters of the pendulum
g = 9.8 # m/s^2
Npendulums = 19
L = 0.5 + np.linspace(0,1,Npendulums) # m
omega = np.sqrt(g/L) # rad/s
T = 2*np.pi/omega # s

# Define the time array
t = np.linspace(0, 5*np.max(T), 100)

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
        line.set_data([starting_points[j], starting_points[j] + x[j,i]], 
                      [starting_points_y[j], starting_points_y[j] + y[j,i]])
    return lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=100, interval=100, blit=True)

plt.close()
HTML(anim.to_html5_video())

# %%
# Define the parameters of the pendulum
g = 9.8 # m/s^2
Npendulums = 19
L = 0.8 + 0.4 * np.random.rand(Npendulums) # m
omega = np.sqrt(g/L) # rad/s
T = 2*np.pi/omega # s

# Define the time array
t = np.linspace(0, 5*np.max(T), 100)

# Define the position of the pendulum as a function of time
theta = 0.1*np.cos(omega[:,np.newaxis]*t)



# Define the position of the pendulum as a function of time
x = L[:,np.newaxis]*np.sin(theta)
y = -L[:,np.newaxis]*np.cos(theta)

fig = plt.figure()
ax = plt.axes(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
lines = []
starting_points_x = np.linspace(-0.9, 0.9, 19)
starting_points_y = -1 + L
for i in range(19):
    line, = ax.plot([], [], lw=2, c='k')
    lines.append(line)

# initialization function: plot the background of each frame
def init():
    for line in lines:
        line.set_data([], [])
    return lines

# animation function.  This is called sequentially
def animate(i):
    for j, line in enumerate(lines):
        line.set_data([starting_points[j], starting_points[j] + x[j,i]], 
                      [starting_points_y[j], starting_points_y[j] + y[j,i]])
    return lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=100, interval=100, blit=True)

plt.close()
HTML(anim.to_html5_video())


# %%
# Make an animation of a sine-wave changing its phase and amplitude smoothly over time
fig = plt.figure()
ax = plt.axes(xlim=(0, 2*np.pi), ylim=(-1.1, 1.1))
line, = ax.plot([], [], lw=2, c='w')
lines = [line]

# initialization function: plot the background of each frame
def init():
    for line in lines:
        line.set_data([], [])
    return lines

# animation function.  This is called sequentially
def animate(i):
    
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x + 0.1*i)*np.cos(0.1*i)
    for line in lines:
        line.set_data(x, y)
    return lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=100, interval=100, blit=True)

plt.close()
HTML(anim.to_html5_video())



# %% [markdown]
# ## Use full EoM of pendulum

# %% [markdown]
# ### Single

# %%
from scipy.integrate import odeint
from matplotlib.patches import Circle

plt.rcParams.update({
     'xtick.top' : False,
     'ytick.right' : False,})


fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].set_xlim(-1.1, 1.1)
ax[0].set_ylim(-1.1, 1.1)
ax[0].axis('off')
ax[0].set_aspect('equal')
ax[0].set_title('Side view')

ax[1].set_xlim(-np.pi,np.pi)
ax[1].set_ylim(-7, 7)
ax[1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax[1].set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
#ax[1].axis('off')
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].tick_params(top=False, right=False)
ax[1].set_title('Phase space')


# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 3, 0.01
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
y0 = np.array([3*np.pi/7, 0])

def deriv_full(y, t, L1):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, zeta1 = y
    theta1dot = zeta1
    z1dot = -g*np.sin(theta1) / L1 
    return theta1dot, z1dot

L1 = 1.0
# Do the numerical integration of the equations of motion
y = odeint(deriv_full, y0, t, args=(L1,))
theta1 = y[:, 0]

x1 = L1*np.sin(theta1)
y1 = -L1*np.cos(theta1)
r = 0.05

line1, = ax[0].plot([], [], lw=1, c='w')
line2, = ax[1].plot([], [], lw=1, c='w')
c1 = Circle((0,0), r, fc='b', ec='b', zorder=10)
c1_im = ax[0].add_patch(c1)
c2 = Circle((0,0), r, fc='b', ec='b', zorder=10)
c2_im = ax[1].add_patch(c2)
lines = [line1,line2,c1_im,c2_im]

def init():
    for j in range(len(lines)-2):
        lines[j].set_data([], [])
    return lines

# animation function.  This is called sequentially
def animate(i):
    lines[0].set_data([0, x1[i]],
                        [0, y1[i]])
    lines[1].set_data([y[:i,0], y[:i,1]])

    lines[2].set_center((x1[i], y1[i]))
    lines[3].set_center((y[i,0], y[i,1]))
    return lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=len(t), interval=20, blit=True)

plt.close()
HTML(anim.to_html5_video())

# %% [markdown]
# ### Multiple - issues at angle jump

# %%
from scipy.integrate import odeint
from matplotlib.patches import Circle
import matplotlib as mpl

plt.rcParams.update({
     'xtick.top' : False,
     'ytick.right' : False,})


fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].set_xlim(-1.1, 1.1)
ax[0].set_ylim(-1.1, 1.1)
ax[0].axis('off')
ax[0].set_aspect('equal')
ax[0].set_title('Side view')

ax[1].set_xlim(-np.pi,np.pi)
ax[1].set_ylim(-11, 11)
ax[1].set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax[1].set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
#ax[1].axis('off')
ax[1].spines[['right', 'top']].set_visible(False)
ax[1].tick_params(top=False, right=False)
ax[1].set_title('Phase space')


# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 3, 0.01
npenduli = 12
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt
dthetadt_array = np.linspace(0.5,10,npenduli)


y0 = np.concatenate([np.zeros(12)[:,np.newaxis], dthetadt_array[:,np.newaxis]], axis=1)

def deriv_full(y, t, L1):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, zeta1 = y
    theta1dot = zeta1
    z1dot = -g*np.sin(theta1) / L1 
    return theta1dot, z1dot

L1 = 1.0
all_output = []
# Do the numerical integration of the equations of motion
for i in range(npenduli):
    y = odeint(deriv_full, y0[i], t, args=(L1,))
    all_output.append(y)
theta1 = y[:, 0]

lines = []
for i in range(npenduli):
    line1, = ax[0].plot([], [], lw=1, c='w')
    line2, = ax[1].plot([], [], lw=1, c='w')
    lines.append(line1)
    lines.append(line2)

color=iter(mpl.pyplot.cm.rainbow(np.linspace(1,0,npenduli)))
for i in range(npenduli):
    c = next(color)
    c1 = Circle((0,0), r, fc=c, ec=c, zorder=10)
    c1_im = ax[0].add_patch(c1)
    c2 = Circle((0,0), r, fc=c, ec=c, zorder=10)
    c2_im = ax[1].add_patch(c2)
    lines.append(c1_im)
    lines.append(c2_im)




x1 = L1*np.sin(theta1)
y1 = -L1*np.cos(theta1)
r = 0.05

def init():
    for j in range(npenduli*2):
        lines[j].set_data([], [])
    return lines

# animation function.  This is called sequentially
def animate(i):
    for j in range(npenduli):
        lines[2*j].set_data([0, L1 * np.sin(all_output[j][i,0])],
                            [0, -L1 * np.cos(all_output[j][i,0])])
        phi_in_range = all_output[j][:i,0] % (2*np.pi)
        phi_in_range[phi_in_range>np.pi] -= 2*np.pi
        lines[2*j+1].set_data(phi_in_range, all_output[j][:i,1])

        lines[2*npenduli+2*j].set_center((L1 * np.sin(all_output[j][i,0]), -L1 * np.cos(all_output[j][i,0])))
        phi_in_range = all_output[j][i,0] % (2*np.pi)
        if phi_in_range>np.pi:  phi_in_range -= 2*np.pi
        lines[2*npenduli+2*j+1].set_center((phi_in_range, all_output[j][i,1]))
    return lines

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=len(t), interval=20, blit=True)

plt.close()
HTML(anim.to_html5_video())

# %% [markdown]
# ### Fix jump

# %%
from scipy.integrate import odeint
from matplotlib.patches import Circle
import matplotlib as mpl

plt.rcParams.update(
    {
        "xtick.top": False,
        "ytick.right": False,
    }
)

DemonstrationType = "Non-circulating"
# Circulating or Non-circulating


fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_xlim(-1.1, 1.1)
ax[0].set_ylim(-1.1, 1.1)
ax[0].axis("off")
ax[0].set_aspect("equal")
ax[0].set_title("Side view")

ax[1].set_xlim(-np.pi, np.pi)
if DemonstrationType == "Circulating":
    ax[1].set_ylim(-11, 11)
elif DemonstrationType == "Non-circulating":
    ax[1].set_ylim(-7, 7)
ax[1].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax[1].set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
# ax[1].axis('off')
ax[1].spines[["right", "top"]].set_visible(False)
ax[1].tick_params(top=False, right=False)
ax[1].set_title("Phase space")


# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 3, 0.01
npenduli = 12
r=0.05

t = np.arange(0, tmax + dt, dt)
# Initial conditions: theta1, dtheta1/dt
if DemonstrationType == "Circulating":
    dthetadt_array = -np.linspace(-10, 10, npenduli)
elif DemonstrationType == "Non-circulating":
    scale = 2*np.sqrt(g) * 0.99
    dthetadt_array = scale*np.linspace(-1, 1, npenduli)
y0 = np.concatenate(
    [np.zeros(12)[:, np.newaxis], dthetadt_array[:, np.newaxis]], axis=1
)


def deriv_full(y, t, L1):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, zeta1 = y
    theta1dot = zeta1
    z1dot = -g * np.sin(theta1) / L1
    return theta1dot, z1dot


L1 = 1.0
all_output = []
# Do the numerical integration of the equations of motion
for i in range(npenduli):
    y = odeint(deriv_full, y0[i], t, args=(L1,))
    all_output.append(y)

n_split = np.sum(np.abs(dthetadt_array) >= 2 * np.sqrt(g * L1)) 


lines = []
for i in range(npenduli):
    (line1,) = ax[0].plot([], [], lw=1, c="w")
    (line2,) = ax[1].plot([], [], lw=1, c="w")
    lines.append(line1)
    lines.append(line2)

color = iter(mpl.pyplot.cm.rainbow(np.linspace(1, 0, npenduli)))
for i in range(npenduli):
    c = next(color)
    c1 = Circle((0, 0), r, fc=c, ec=c, zorder=10)
    c1_im = ax[0].add_patch(c1)
    c2 = Circle((0, 0), r, fc=c, ec=c, zorder=10)
    c2_im = ax[1].add_patch(c2)
    lines.append(c1_im)
    lines.append(c2_im)

for i in range(n_split):
    (line_new,) = ax[1].plot([], [], lw=1, c="w")
    lines.append(line_new)


r = 0.05


def init():
    for j in range(npenduli * 2):
        lines[j].set_data([], [])
    return lines


pendulum_out_of_range_index = np.zeros(npenduli, dtype=int)
pendulum_out_of_range_marker = np.zeros(npenduli, dtype=int)


# animation function.  This is called sequentially
def animate(i):
    for j in range(npenduli):
        lines[2 * j].set_data(
            [0, L1 * np.sin(all_output[j][i, 0])],
            [0, -L1 * np.cos(all_output[j][i, 0])],
        )
        phi_in_range = all_output[j][:i, 0] % (2 * np.pi)
        phi_in_range[phi_in_range > np.pi] -= 2 * np.pi
        lines[2 * j + 1].set_data(all_output[j][:i, 0], all_output[j][:i, 1])
        if (all_output[j][i - 1, 0] > np.pi) or (all_output[j][i - 1, 0] < -np.pi):
            if pendulum_out_of_range_marker[j] == 0:
                pendulum_out_of_range_index[j] = i
                pendulum_out_of_range_marker[j] = (
                    np.max(pendulum_out_of_range_marker) + 1
                )
                #print([j, all_output[j][i - 1, 0]])
            sign = np.sign(all_output[j][i - 1, 0])
            #print([all_output[j][i - 1, 0], sign])
            #print ([pendulum_out_of_range_marker[j], n_split])
            lines[4 * npenduli + pendulum_out_of_range_marker[j] - 1].set_data(
                all_output[j][pendulum_out_of_range_index[j] : i, 0] - sign * 2 * np.pi,
                all_output[j][pendulum_out_of_range_index[j] : i, 1],
            )

        lines[2 * npenduli + 2 * j].set_center(
            (L1 * np.sin(all_output[j][i, 0]), -L1 * np.cos(all_output[j][i, 0]))
        )
        phi_in_range = all_output[j][i, 0] % (2 * np.pi)
        if phi_in_range > np.pi:
            phi_in_range -= 2 * np.pi
        lines[2 * npenduli + 2 * j + 1].set_center((phi_in_range, all_output[j][i, 1]))
    return lines


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=len(t), interval=20, blit=True
)

plt.close()
HTML(anim.to_html5_video())

# %%

# Pendulum rod lengths (m), bob masses (kg).
L1, L2 = 1, 1
m1, m2 = 1, 1
# The gravitational acceleration (m.s-2).
g = 9.81

def deriv(y, t, L1, L2, m1, m2):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, z1, theta2, z2 = y

    c, s = np.cos(theta1-theta2), np.sin(theta1-theta2)

    theta1dot = z1
    z1dot = (m2*g*np.sin(theta2)*c - m2*s*(L1*z1**2*c + L2*z2**2) -
             (m1+m2)*g*np.sin(theta1)) / L1 / (m1 + m2*s**2)
    theta2dot = z2
    z2dot = ((m1+m2)*(L1*z1**2*s - g*np.sin(theta2) + g*np.sin(theta1)*c) + 
             m2*L2*z2**2*s*c) / L2 / (m1 + m2*s**2)
    return theta1dot, z1dot, theta2dot, z2dot

def calc_E(y):
    """Return the total energy of the system."""

    th1, th1d, th2, th2d = y.T
    V = -(m1+m2)*L1*g*np.cos(th1) - m2*L2*g*np.cos(th2)
    T = 0.5*m1*(L1*th1d)**2 + 0.5*m2*((L1*th1d)**2 + (L2*th2d)**2 +
            2*L1*L2*th1d*th2d*np.cos(th1-th2))
    return T + V

# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 30, 0.01
t = np.arange(0, tmax+dt, dt)
# Initial conditions: theta1, dtheta1/dt, theta2, dtheta2/dt.
y0 = np.array([3*np.pi/7, 0, 3*np.pi/4, 0])

# Do the numerical integration of the equations of motion
y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))

# Check that the calculation conserves total energy to within some tolerance.
EDRIFT = 0.05
# Total energy from the initial conditions
E = calc_E(y0)
if np.max(np.sum(np.abs(calc_E(y) - E))) > EDRIFT:
    sys.exit('Maximum energy drift of {} exceeded.'.format(EDRIFT))

# Unpack z and theta as a function of time
theta1, theta2 = y[:,0], y[:,2]

# Convert to Cartesian coordinates of the two bob positions.
x1 = L1 * np.sin(theta1)
y1 = -L1 * np.cos(theta1)
x2 = x1 + L2 * np.sin(theta2)
y2 = y1 - L2 * np.cos(theta2)

# Plotted bob circle radius
r = 0.05
# Plot a trail of the m2 bob's position for the last trail_secs seconds.
trail_secs = 1
# This corresponds to max_trail time points.
max_trail = int(trail_secs / dt)


def init():
    line, = ax.plot([], [], lw=2, c='w')[0]
    c0 = Circle((0, 0), r/2, fc='k', zorder=10)
    c1 = Circle((x1[0], y1[0]), r, fc='b', ec='b', zorder=10)
    c2 = Circle((x2[0], y2[0]), r, fc='r', ec='r', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)
    return lines

def make_plot(i):
    # Plot and save an image of the double pendulum configuration for time
    # point i.
    # The pendulum rods.
    ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')
    # Circles representing the anchor point of rod 1, and bobs 1 and 2.
    c0 = Circle((0, 0), r/2, fc='k', zorder=10)
    c1 = Circle((x1[i], y1[i]), r, fc='b', ec='b', zorder=10)
    c2 = Circle((x2[i], y2[i]), r, fc='r', ec='r', zorder=10)
    ax.add_patch(c0)
    ax.add_patch(c1)
    ax.add_patch(c2)

    # The trail will be divided into ns segments and plotted as a fading line.
    ns = 20
    s = max_trail // ns

    for j in range(ns):
        imin = i - (ns-j)*s
        if imin < 0:
            continue
        imax = imin + s + 1
        # The fading looks better if we square the fractional length along the
        # trail.
        alpha = (j/ns)**2
        ax.plot(x2[imin:imax], y2[imin:imax], c='r', solid_capstyle='butt',
                lw=2, alpha=alpha)

    # Centre the image on the fixed anchor point, and ensure the axes are equal
    ax.set_xlim(-L1-L2-r, L1+L2+r)
    ax.set_ylim(-L1-L2-r, L1+L2+r)
    ax.set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig('frames/_img{:04d}.png'.format(i//di), dpi=72)
    plt.cla()


# Make an image every di time points, corresponding to a frame rate of fps
# frames per second.
# Frame rate, s-1
fps = 10
di = int(1/fps/dt)
fig = plt.figure(figsize=(8.3333, 6.25), dpi=72)
ax = fig.add_subplot(111)

for i in range(0, t.size, di):
    print(i // di, '/', t.size // di)
    make_plot(i)

# %% [markdown]
# ## Plots for notes

# %%
plt.rcParams.keys()

# %%

plt.rcParams.update({
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white", 
     "text.color": "k",
     "axes.labelcolor": "k",
     "axes.edgecolor": "k",
     "xtick.color": "k",
     "ytick.color": "k",
     'xtick.minor.visible' : True, 
     'xtick.top' : False,
     'ytick.minor.visible' : True, 
     'ytick.right' : False,
     'xtick.direction' : 'in', 
     'ytick.direction' :'in',
     'font.size' : 14,})

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

arrows = False
#ax.spines[["right", "top"]].set_visible(False)
#ax.tick_params(top=False, right=False)

ax.set_xlim(-0.11, 0.11)
ax.set_ylim(-0.11, 0.11)
ax.set_aspect('equal')

scale = 0.015
th = np.linspace(0, 2*np.pi, 100)
x = np.sin(th)
y = np.cos(th)

rng = np.random.default_rng(12345)

randomphase = rng.integers(low=0, high=len(th), size=6)
#randint(low=0, high=len(th), size = 6)


for l in range(1, 6):
    ax.plot(scale*l*x, scale*l*y, lw=2, c='k')
    ax.plot(scale*l*x[randomphase[l-1]], scale*l*y[randomphase[l-1]], '.', c='k', markersize=20)

plt.xlabel(r'$\theta$ [rad]')
plt.ylabel(r'$p_\theta\,/\,(ml^{5/2}g^{-1/2})$')
plt.xticks([-0.1, -0.05, 0, 0.05, 0.1])
plt.yticks([-0.1, -0.05, 0, 0.05, 0.1], labels = [r'$-0.1$', r'$-0.05$', '0', r'$0.05$', r'$0.1$'])
if arrows:
    for i in range(1,6):
        plt.quiver(0.005, -scale*i, -0.1, 0, width=0.05, color='k')
    plt.tight_layout()
    plt.savefig('images/low_amplitude.pdf', dpi=300)
else:
    plt.tight_layout()
    plt.savefig('images/low_amplitude_no_arrows.pdf', dpi=300)
plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_xlim(-np.pi,np.pi)
ax.set_ylim(-11/np.sqrt(g), 11/np.sqrt(g))
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
#ax.spines[['right', 'top']].set_visible(False)
#ax.tick_params(top=False, right=False)


# Maximum time, time point spacings and the time grid (all in s).
tmax, dt = 12, 0.01
npenduli = 15
r=0.05

t = np.arange(0, tmax + dt, dt)
# Initial conditions: theta1, dtheta1/dt
scale = 2*np.sqrt(g) * 1.70
dthetadt_array = scale*np.linspace(-1, 1, npenduli)
mask = np.abs(np.abs(dthetadt_array) - 2*np.sqrt(g)) >0.5
#print(mask)
dthetadt_array = dthetadt_array[mask]
dthetadt_array = np.concatenate([dthetadt_array, [0.99999*2*np.sqrt(g)]])
npenduli = len(dthetadt_array)
y0 = np.concatenate(
    [np.zeros(npenduli)[:, np.newaxis], dthetadt_array[:, np.newaxis]], axis=1
)
print (dthetadt_array)

def deriv_full(y, t, L1):
    """Return the first derivatives of y = theta1, z1, theta2, z2."""
    theta1, zeta1 = y
    theta1dot = zeta1
    z1dot = -g * np.sin(theta1) / L1
    return theta1dot, z1dot


L1 = 1.0
all_output = []
# Do the numerical integration of the equations of motion
for i in range(npenduli):
    y = odeint(deriv_full, y0[i], t, args=(L1,))
    all_output.append(y)


rng = np.random.default_rng(12345)



for i in range(npenduli):
    if np.max(all_output[i][:,0]) > np.pi:
        all_output[i][:,0] = all_output[i][:,0] - (2*np.pi)
    elif np.min(all_output[i][:,0]) < -np.pi:
        all_output[i][:,0] = all_output[i][:,0] + (2*np.pi)
    ax.plot(all_output[i][:,0], all_output[i][:,1]/np.sqrt(g), lw=1, c='k')
    # The below didn't quite work
    #tmp_rand = rng.integers(low=0, high=len(all_output[i][:,0]), size=1)
    #ax.plot(all_output[i][tmp_rand,0], all_output[i][tmp_rand,1]/np.sqrt(g), '.', c='k', markersize=20)


for yq in dthetadt_array:
    if(yq>0):
        plt.quiver(0.005, yq/np.sqrt(g), 0.1, 0, width=0.03, color='k')
    elif(yq<-2*np.sqrt(g)):
        plt.quiver(0.005, yq/np.sqrt(g), -0.1, 0, width=0.03, color='k')

plt.xlabel(r'$\theta$ [rad]')
plt.ylabel(r'$p_\theta\,/\,(ml^{5/2}g^{-1/2})$')
plt.tight_layout()
plt.savefig('images/full_pendulum_phase_space.pdf', dpi=300)

plt.show()


# %%
# ?plt.quiver
