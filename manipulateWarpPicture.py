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
import numpy as np
from numpy import pi, sin, cos
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import skimage.io as sio
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "notebook"


# %%
def image2zvals(img,  n_colors=64, n_training_pixels=800, rngs = 123): 
    # Image color quantization
    # img - np.ndarray of shape (m, n, 3) or (m, n, 4)
    # n_colors: int,  number of colors for color quantization
    # n_training_pixels: int, the number of image pixels to fit a KMeans instance to them
    # returns the array of z_values for the heatmap representation, and a plotly colorscale
   
    if img.ndim != 3:
        raise ValueError(f"Your image does not appear to  be a color image. It's shape is  {img.shape}")
    rows, cols, d = img.shape
    if d < 3:
        raise ValueError(f"A color image should have the shape (m, n, d), d=3 or 4. Your  d = {d}") 
        
    range0 = img[:, :, 0].max() - img[:, :, 0].min()
    if range0 > 1: #normalize the img values
        img = np.clip(img.astype(float)/255, 0, 1)
        
    observations = img[:, :, :3].reshape(rows*cols, 3)
    training_pixels = shuffle(observations, random_state=rngs)[:n_training_pixels]
    model = KMeans(n_clusters=n_colors, random_state=rngs).fit(training_pixels)
    
    codebook = model.cluster_centers_
    indices = model.predict(observations)
    z_vals = indices.astype(float) / (n_colors-1) #normalization (i.e. map indices to  [0,1])
    z_vals = z_vals.reshape(rows, cols)
    # define the Plotly colorscale with n_colors entries    
    scale = np.linspace(0, 1, n_colors)
    colors = (codebook*255).astype(np.uint8)
    print(colors)
    pl_colorscale = [[sv, f'rgb{tuple(color)}'] for sv, color in zip(scale, colors)]
      
    # Reshape z_vals  to  img.shape[:2]
    return z_vals.reshape(rows, cols), pl_colorscale

def image2zvals_alpha(img,  n_colors=64, n_training_pixels=800, rngs = 123): 
    # Image color quantization
    # img - np.ndarray of shape (m, n, 3) or (m, n, 4)
    # n_colors: int,  number of colors for color quantization
    # n_training_pixels: int, the number of image pixels to fit a KMeans instance to them
    # returns the array of z_values for the heatmap representation, and a plotly colorscale
   
    if img.ndim != 3:
        raise ValueError(f"Your image does not appear to  be a color image. It's shape is  {img.shape}")
    rows, cols, d = img.shape
    if d < 3:
        raise ValueError(f"A color image should have the shape (m, n, d), d=3 or 4. Your  d = {d}") 
        
    range0 = img[:, :, 0].max() - img[:, :, 0].min()
    if range0 > 1: #normalize the img values
        img = np.clip(img.astype(float)/255, 0, 1)
        
    observations = img[:, :, :3].reshape(rows*cols, 3)
    training_pixels = shuffle(observations, random_state=rngs)[:n_training_pixels]
    model = KMeans(n_clusters=n_colors, random_state=rngs).fit(training_pixels)
    
    codebook = model.cluster_centers_
    indices = model.predict(observations)
    z_vals = indices.astype(float) / (n_colors-1) #normalization (i.e. map indices to  [0,1])
    z_vals = z_vals.reshape(rows, cols)
    # define the Plotly colorscale with n_colors entries    
    scale = np.linspace(0, 1, n_colors)
    brightness = get_brightness(codebook[:,0], codebook[:,1], codebook[:,2])
    #codebook = np.c_[codebook, brightness]
    colors = (codebook*255).astype(np.uint8)
    colors = np.c_[colors, brightness]
    #print(colors)
    pl_colorscale = [[sv, f'rgba{tuple(color)}'] for sv, color in zip(scale, colors)]
    print(pl_colorscale)
    # Reshape z_vals  to  img.shape[:2]
    return z_vals.reshape(rows, cols), pl_colorscale


def regular_tri(rows, cols):
    #define triangles for a np.meshgrid(np.linspace(a, b, cols), np.linspace(c,d, rows))
    triangles = []
    for i in range(rows-1):
        for j in range(cols-1):
            k = j+i*cols
            triangles.extend([[k,  k+cols, k+1+cols], [k, k+1+cols, k+1]])
    return np.array(triangles) 
       
def mesh_data(img, n_colors=32, n_training_pixels=800):
    rows, cols, _ = img.shape
    z_data, pl_colorscale = image2zvals(img, n_colors=n_colors, n_training_pixels=n_training_pixels)
    triangles = regular_tri(rows, cols) 
    I, J, K = triangles.T
    zc = z_data.flatten()[triangles] 
    tri_color_intensity = [zc[k][2] if k%2 else zc[k][1] for k in range(len(zc))]  
    return I, J, K, tri_color_intensity, pl_colorscale


def mesh_data_alpha(img, n_colors=32, n_training_pixels=800):
    rows, cols, _ = img.shape
    z_data, pl_colorscale = image2zvals_alpha(img, n_colors=n_colors, n_training_pixels=n_training_pixels)
    triangles = regular_tri(rows, cols) 
    I, J, K = triangles.T
    zc = z_data.flatten()[triangles] 
    tri_color_intensity = [zc[k][2] if k%2 else zc[k][1] for k in range(len(zc))]  
    return I, J, K, tri_color_intensity, pl_colorscale



def get_brightness(R,G,B):
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    return Y

scene_style = dict(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                   scene_aspectmode="data")

# %%
imgs = ["sun-flower.jpg" , "street-exhibition.jpg"]
img1 = sio.imread(f"images/{imgs[0]}") 
px.imshow(img1)

# %%

z_data, pl_colorscale = image2zvals(img1, n_colors=32, n_training_pixels=2000)
fig1 = go.Figure(go.Heatmap(z=np.flipud(z_data), colorscale=pl_colorscale))
fig1.update_layout(title_text="Image represented as a heatmap", title_x=0.5,
           width=600, height=600)
fig1.show()


# %%
def surface(rows, cols):
    x = np.linspace(-pi, pi, cols) 
    y = np.linspace(-pi, pi, rows)
    x, y = np.meshgrid(x,y)
    z = 0.5*cos(x/2) + 0.2*sin(y/4)
    return x, y, z


# %%
r, c, _  = img1.shape
x, y, z = surface(r, c)

# %%
fig2 = go.Figure(go.Surface(x=x[0, :], y = np.flipud(y[:, 0]), z=z, 
                            surfacecolor=z_data, colorscale=pl_colorscale))
fig2.update_layout(width=600, height=600, **scene_style,
                  scene_camera_eye=dict(x=-2, y=-2, z=1.5))
fig2.show()

# %%
I, J, K, tri_color_intensity, pl_colorscale = mesh_data(img1, n_colors=32, n_training_pixels=5000) 
fig3 = go.Figure(go.Mesh3d(x=x.flatten(), y=np.flipud(y).flatten(), z=z.flatten(), 
                            i=I, j=J, k=K, intensity=tri_color_intensity, intensitymode="cell", 
                            colorscale=pl_colorscale, showscale=False))
fig3.update_layout(width=500, height=500,
                  margin=dict(t=5, r=5, b=5, l=5), 
                  scene_camera_eye=dict(x=-2, y=-2, z=1.5),
                  **scene_style)
fig3.show()


# %%
def warp_surface(rows, cols):
    R0=8. 
    z0=0.9 
    phi0=pi
    alpha_warp = 1.1

    x = np.linspace(-20, 20, cols) 
    y = np.linspace(-20,20, rows)
    x, y = np.meshgrid(x,y)
    R = (x**2 + y**2)**0.5
    phi = np.arctan2(y, x)
    z = np.zeros_like(x)
    z[R>R0] = z0*(R[R>R0]-R0)**alpha_warp * np.sin(phi[R>R0]-phi0)
    return x, y, z


# %%
img1 = sio.imread(f"images/Stefan Payne-Wardenaar milkyway-cut.jpeg") 


# %%
r, c, _  = img1.shape
x, y, z = warp_surface(r, c)

# %%
I, J, K, tri_color_intensity, pl_colorscale = mesh_data(img1, n_colors=32, n_training_pixels=5000) 
fig3 = go.Figure(go.Mesh3d(x=x.flatten(), y=np.flipud(y).flatten(), z=z.flatten(), 
                            i=I, j=J, k=K, intensity=tri_color_intensity, intensitymode="cell", 
                            colorscale=pl_colorscale, showscale=False))
fig3.update_layout(width=500, height=500,
                   paper_bgcolor="rgba(0,0,0,1)",
                  margin=dict(t=5, r=5, b=5, l=5), 
                  scene_camera_eye=dict(x=-2, y=-2, z=1.5),
                  **scene_style)
fig3.show()
