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
#     display_name: Agama_env
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
#df = pd.read_csv('../../../data/RVsample-result.csv')
#df.to_hdf('../../../data/RVsample-result.h5', key='table')
df = pd.read_hdf('../../../data/RVsample-result.h5', key='table')


# %%
df.columns

# %% [markdown]
# ## Select Williams stars (crude)

# %%
mask_Williams = df.parallax.values > (1/1.5)
df_Williams = df[mask_Williams]
df = []

# %%
mask_colour = np.isnan(df_Williams['phot_bp_mean_mag'].values) | np.isnan(df_Williams['phot_rp_mean_mag'].values)
df_Williams = df_Williams[~mask_colour]

# %%
len(df_Williams)

# %% [markdown]
# ## Plot HR diagram

# %%

# %%
np.sum(np.isnan(df_Williams['phot_rp_mean_mag'].values))

# %%
import matplotlib as mpl

# %%
absMag = df_Williams['phot_g_mean_mag'] - 5 * np.log10(100. / df_Williams['parallax'])
colour = df_Williams['phot_bp_mean_mag'] - df_Williams['phot_rp_mean_mag']
plt.hist2d(colour, absMag, bins=[np.linspace(-0.5, 4, 100), np.linspace(-5,12,100)], cmap='Blues', norm = mpl.colors.LogNorm())
plt.gca().invert_yaxis()  # Invert y-axis to have brighter stars at the top
plt.xlabel('Colour (BP - RP)')
plt.ylabel('Absolute Magnitude')
plt.colorbar(label='Number of stars')
plt.title('Colour-Magnitude Diagram for Williams Stars')
plt.show()

# %% [markdown]
# ### Zoom in on red clump

# %%

from matplotlib.path import Path as mpl_path

# Function to select points inside a given polygon
def inside_poly(data, vertices):
    return mpl_path(vertices).contains_points(data)


# %%
plt.hist2d(colour, absMag, bins=[np.linspace(0.5, 2.5, 100), np.linspace(-1,3,100)], cmap='Reds', norm = mpl.colors.LogNorm())
plt.gca().invert_yaxis()  # Invert y-axis to have brighter stars at the top
plt.xlabel('Colour (BP - RP)')
plt.ylabel('Absolute Magnitude')
plt.colorbar(label='Number of stars')
plt.title('Colour-Magnitude Diagram for Williams Stars')
plt.grid(True)
plt.show()

# %%
polRC = np.array([[1.05, 0.5], [1.15, 0.2], [1.6, 1], [1.4, 1.2], [1.05,0.5]])
plt.hist2d(colour, absMag, bins=[np.linspace(0.5, 2.5, 100), np.linspace(-1,3,100)], cmap='Reds', norm = mpl.colors.LogNorm())

plt.plot(polRC[:, 0], polRC[:, 1], 'k')
plt.gca().invert_yaxis()  # Invert y-axis to have brighter stars at the top
plt.xlabel('Colour (BP - RP)')
plt.ylabel('Absolute Magnitude')
plt.colorbar(label='Number of stars')
plt.show()

# %%
maskRC = inside_poly(np.array([colour, absMag]).T, polRC)

# %%
import astropy.units as u
import astropy.coordinates as coord

# %%
# Convert Gaia observations to Galactocentric coordinates using astropy

# Create SkyCoord object from Gaia data
c = coord.SkyCoord(
    ra=df_Williams['ra'].values * u.deg,
    dec=df_Williams['dec'].values * u.deg,
    distance=(1000. / df_Williams['parallax'].values) * u.pc,
    pm_ra_cosdec=df_Williams['pmra'].values * u.mas/u.yr,
    pm_dec=df_Williams['pmdec'].values * u.mas/u.yr,
    radial_velocity=df_Williams['radial_velocity'].values * u.km/u.s,
    frame='icrs'
)

# Transform to Galactocentric coordinates, in cylindrical polar coordinates
# (R, phi, z) where R is the Galactocentric radius, phi is the azimuthal angle, and z is the height above the Galactic plane
# and also the velocity components in the three directions (v_R, v_phi, v_z)
# We do this natively within the astropy coordinate transforms
c_gal = c.transform_to(coord.Galactocentric, )
c_gal.representation_type = 'cylindrical'

# Extract the Galactocentric coordinates and velocities
R = c_gal.rho.to(u.kpc).value  # Galactocentric radius
phi = c_gal.phi.to(u.rad).value  # Azimuthal angle
z = c_gal.z.to(u.kpc).value  # Height above the Galactic plane
v_R = c_gal.d_rho.to(u.km/u.s).value  # Radial velocity
v_z = c_gal.d_z.to(u.km/u.s).value  # Vertical velocity
v_phi = (c_gal.rho * c_gal.d_phi / u.rad).to(u.km/u.s).value  # Azimuthal velocity


# %%
plt.hist2d(R, z, bins=[np.linspace(6.5,9.7, 50), np.linspace(-1.5,1.5, 50)], cmap='Grays', norm=mpl.colors.LogNorm())
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('Galactocentric Radius (kpc)')
plt.ylabel('Height above Galactic Plane (kpc)')
plt.colorbar(label='Number of stars')
plt.show()

# %%
import scipy.stats as stats
# Calculate the mean v_z in 2D bins of R and z using scipy.stats.binned_statistic_2d
v_z_mean, xedges, yedges, binnumber = stats.binned_statistic_2d(
    R, z, v_z, statistic='mean', bins=[np.linspace(6.5, 9.7, 50), np.linspace(-1.5, 1.5, 50)]
)
# Plot the mean v_z in 2D bins of R and z
plt.imshow(v_z_mean.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
           cmap='coolwarm', aspect='equal', 
           interpolation='nearest', vmin=-10, vmax=10)
plt.colorbar(label='Mean v_z (km/s)')
plt.xlabel('Galactocentric Radius (kpc)')
plt.ylabel('Height above Galactic Plane (kpc)')

plt.show()



# %%
# Calculate the mean v_R in 2D bins of R and z using scipy.stats.binned_statistic_2d
v_R_mean, xedges, yedges, binnumber = stats.binned_statistic_2d(
    R, z, v_R, statistic='mean', bins=[np.linspace(6.5, 9.7, 50), np.linspace(-1.5, 1.5, 50)]
)
# Plot the mean v_R in 2D bins of R and z
plt.imshow(v_R_mean.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
           cmap='coolwarm', aspect='equal', 
           interpolation='nearest', vmin=-10, vmax=10)
plt.colorbar(label='Mean v_R (km/s)')
plt.xlabel('Galactocentric Radius (kpc)')
plt.ylabel('Height above Galactic Plane (kpc)')

plt.show()

# %% [markdown]
# ## Now the same for the RC sample

# %%
v_z_mean, xedges, yedges, binnumber = stats.binned_statistic_2d(
    R[maskRC], z[maskRC], v_z[maskRC], statistic='mean', bins=[np.linspace(6.5, 9.7, 20), np.linspace(-1.5, 1.5, 20)]
)
# Plot the mean v_z in 2D bins of R and z
plt.imshow(v_z_mean.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
           cmap='coolwarm', aspect='equal', 
           interpolation='bilinear', vmin=-6, vmax=6)
plt.colorbar(label='Mean v_z (km/s)')
plt.xlabel('Galactocentric Radius (kpc)')
plt.ylabel('Height above Galactic Plane (kpc)')

plt.show()



# %%
v_R_mean, xedges, yedges, binnumber = stats.binned_statistic_2d(
    R[maskRC], z[maskRC], v_R[maskRC], statistic='mean', bins=[np.linspace(6.5, 9.7, 20), np.linspace(-1.5, 1.5, 20)]
)
# Plot the mean v_R in 2D bins of R and z
plt.imshow(v_R_mean.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
           cmap='coolwarm', aspect='equal', 
           interpolation='nearest', vmin=-10, vmax=10)
plt.colorbar(label='Mean v_R (km/s)')
plt.xlabel('Galactocentric Radius (kpc)')
plt.ylabel('Height above Galactic Plane (kpc)')

plt.show()

