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
import scipy.stats as stats


# Make the plots have a bigger font size
plt.rcParams.update({
     'font.size' : 18, 
     'axes.titlesize' : 24})

# %% [markdown]
# ## Plots for PA1720 Matrices lecture 4
# Basic examples of matrix techniques used in machine learning

# %%
a,b,c,d,e = 17,-4,0.5,-0.02,0
t = np.linspace(1,15,140)
t_data = np.linspace(1,15,14)
y = a + b*t + c*t**2 + d*t**3 + e*t**4 
#+ stats.norm.rvs(0, 1, 14)
y_data = a + b*t_data + c*t_data**2 + d*t_data**3 + e*t_data**4 + stats.norm.rvs(0, 2, 14)

plt.plot(t, y)
plt.plot( t_data, y_data,'kx')
plt.errorbar(t_data, y_data, yerr=2, fmt='kx', capsize=3)
plt.xlabel('t')
plt.ylabel('y')
plt.show()

# %%
x = np.random.multivariate_normal([2,7], [[1,0.8],[0.8,0.8]], 200)
plt.plot(x[:,0], x[:,1], 'x')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal', 'box')

plt.show()

# %%

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', NoLine-=False, PCA=False, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
    eig = np.linalg.eig(cov)
    ellipse.set_transform(transf + ax.transData)
    
    if NoLine:
        return

    if PCA:
        l1 = ax.plot([mean_x, mean_x + n_std*np.sqrt((eig[0])[0]) * (eig[1])[0][0]],
                     [mean_y, mean_y + n_std*np.sqrt((eig[0])[0]) * (eig[1])[1][0]], 'k-', lw=2)
        l2 = ax.plot([mean_x, mean_x + n_std*np.sqrt((eig[0])[1]) * (eig[1])[0][1]],
                     [mean_y, mean_y + n_std*np.sqrt((eig[0])[1]) * (eig[1])[1][1]], 'k-', lw=2)

        #l2 = ax.plot([mean_x, mean_x + eig[0][1][0]],[mean_y, mean_y + eig[0][1][1]], 'r-', lw=2)
        return ax.add_patch(ellipse), l1, l2
    return ax.add_patch(ellipse)


# %%
import matplotlib.patches as patches
plt.plot(x[:,0], x[:,1], 'x')
plt.xlabel('x')
plt.ylabel('y')
confidence_ellipse(x[:,0], x[:,1], plt.gca(), n_std = 1, edgecolor='red')
confidence_ellipse(x[:,0], x[:,1], plt.gca(), n_std = 2, edgecolor='red')
plt.gca().set_aspect('equal', 'box')
plt.show()

# %%

plt.plot(x[:,0], x[:,1], 'x')
plt.xlabel('x')
plt.ylabel('y')
confidence_ellipse(x[:,0], x[:,1], plt.gca(), n_std = 1, edgecolor='red')
confidence_ellipse(x[:,0], x[:,1], plt.gca(), n_std = 2, PCA=True, edgecolor='red')
plt.gca().set_aspect('equal', 'box')

plt.show()

# %%
cov = np.cov(x[:,0], x[:,1])
eig = np.linalg.eig(cov)
eigvec = eig[1]
xy2 = np.matmul(eigvec, x.T) # rotate the data
plt.plot(xy2[0], xy2[1], 'x')
plt.gca().set_aspect('equal', 'box')

plt.show()

# %%
import matplotlib.patches as patches
plt.plot(xy2[0], xy2[1], 'x')
plt.xlabel('x')
plt.ylabel('y')
confidence_ellipse(xy2[0], xy2[1], plt.gca(), n_std = 1, edgecolor='red')
confidence_ellipse(xy2[0], xy2[1], plt.gca(), n_std = 2, edgecolor='red')
plt.gca().set_aspect('equal', 'box')
plt.show()

# %% [markdown]
# ## Plot to demonstrate problem with Uppal et al 2024

# %%
plt.rcParams['font.size'] = 18
R = np.linspace(8, 15, 200)
Cw = 0.0057
epsw = 1.4
Rw = 7.4
Z_upp = Cw * (R-Rw)**epsw
plt.plot(R, Z_upp, label='Z_uppal')
plt.xlabel('R')
plt.ylabel('Z_uppal')
plt.show()
