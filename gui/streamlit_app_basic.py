import matplotlib.pyplot as plt
import torch
from astropy import constants as c
import numpy as np
from caustics.utils import get_meshgrid
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, NFW, EPL, ExternalShear, SinglePlane
from caustics.light import sersic
from caustics.light.pixelated import Pixelated
from caustics.parametrized import Parametrized
from caustics.utils import batch_lm
import caustics
import streamlit as st


def caustic_trace(delta_m, x, z_background = 3.042, simulation_size = 650, H0=67, Om0=0.32, Tcmb0=2.725, upsample_factor = 1, device = "cpu"):
    
    cosmology = FlatLambdaCDM(name = "cosmo")
    cosmology.to(dtype=torch.float32)
    epl = EPL(name = "epl", cosmology = cosmology, z_l = 0.2999)
    shear = ExternalShear(name = "shear", cosmology = cosmology, z_l = 0.2999)
    lens = SinglePlane(lenses = [epl, shear], cosmology = cosmology, z_l = 0.2999)
    fov = delta_m*simulation_size
    res = fov/simulation_size #Simulation pixel size
    thx, thy = get_meshgrid(res/upsample_factor, upsample_factor*simulation_size, upsample_factor*simulation_size, dtype=torch.float32, device = device)
    # Conveniently caustic has a function to compute the jacobian of the lens equation
    A = lens.jacobian_lens_equation(thx, thy, z_background, lens.pack(x))
    # Note that if this is too slow you can set `method = "finitediff"` to run a faster version. You will also need to provide `pixelscale` then

    # Here we compute A's determinant at every point
    detA = torch.linalg.det(A)
    levels = [0.]
    CS = plt.contour(thx.cpu().detach().numpy(), thy.cpu().detach().numpy(), detA.cpu().detach().numpy(), levels = [0.], colors = "b")
    paths = CS.collections[0].get_paths()
    caustic_paths = []
    y1s = []
    y2s = []
    for path in paths:
        # Collect the path into a descrete set of points
        vertices = path.interpolated(5).vertices
        x1 = torch.tensor(list(float(vs[0]) for vs in vertices), device = device)
        x2 = torch.tensor(list(float(vs[1]) for vs in vertices), device = device)
        # raytrace the points to the source plane
        y1,y2 = lens.raytrace(x1, x2, z_background, params = lens.pack(x))
        y1s += y1.cpu()/delta_m + simulation_size/2
        y2s += y2.cpu()/delta_m + simulation_size/2
    
    plt.close()
    return np.array(y1s),np.array(y2s)

# Streamlit app
st.title("Gravitational Lensing Simulator")

# Sliders for lens parameters
x0 = st.slider("x0", -2.0, 2.0, 0.0)
y0 = st.slider("y0", -2.0, 2.0, 0.25)
q = st.slider("q", 0.1, 1.0, 0.82)
phi = st.slider("phi", 0.0, 2*np.pi, 8*(np.pi/180) + np.pi/2)
theta_E = st.slider("theta_E", 0.0, 2.0, 1.606)
gamma = st.slider("gamma", 0.0, 2.0, 1.0)
gamma1 = st.slider("gamma1", -1.0, 1.0, 0.036*np.cos(2*3*(np.pi/180)))
gamma2 = st.slider("gamma2", -1.0, 1.0, 0.036*np.sin(2*3*(np.pi/180)))
shear1 = st.slider("shear1", -1.0, 1.0, 0.01)
shear2 = st.slider("shear2", -1.0, 1.0, 0.0)
src_x0 = st.slider("src_x0", -2.0, 2.0, 0.0)
src_y0 = st.slider("src_y0", -2.0, 2.0, -0.2+0.25)
src_q = st.slider("src_q", 0.1, 1.0, 0.5)
src_phi = st.slider("src_phi", 0.0, 2*np.pi, 0.0)
src_n = st.slider("src_n", 0.1, 10.0, 0.8)
src_Re = st.slider("src_Re", 0.0, 2.0, 1.25)
src_Ie = st.slider("src_Ie", 0.0, 2.0, 0.3)

x = torch.tensor(
    [
        x0,  #x0
        y0,  #y0
        q, #Minor/major axs
        phi, #Angle
        theta_E, #Einstein radius
        gamma, #Gamma -1 
        gamma1, #Gamma_1
        gamma2,    #Gamma_2
        shear1, # Shear x
        shear2, #Shear y
        src_x0,  # src x0
        src_y0,  # src y0
        src_q,  # src q
        src_phi,  # src phi
        src_n,  # src n
        src_Re,  # src Re
        src_Ie,  # src Ie
    ]
)
deltam = 1e-2
cosmology = FlatLambdaCDM(name = "cosmo")
epl = EPL(name = "epl", cosmology = cosmology, z_l = 0.2999)
shear = ExternalShear(name = "shear", cosmology = cosmology, z_l = 0.2999)
lens = SinglePlane(lenses = [epl, shear], cosmology = cosmology, z_l = 0.2999)
src = sersic.Sersic(name="src")
minisim = caustics.Lens_Source(lens=lens, source=src, pixelscale=deltam, pixels_x=650, z_s = 3.042)


y1s, y2s = caustic_trace(delta_m = deltam, x = x[:10], z_background = 3.042, simulation_size = 650, H0=67, Om0=0.32, Tcmb0=2.725, upsample_factor = 1, device = "cpu")
fig1, ax1 = plt.subplots(figsize=(7, 7))
ax1.plot(y1s, y2s)
ax1.imshow(np.rot90(minisim(x, lens_source=True)), origin="lower")
st.pyplot(fig1)

# Plot the lensed image
fig2, ax2 = plt.subplots(figsize=(7, 7))
ax2.imshow(np.rot90(minisim(x, lens_source=False)), origin="lower")
st.pyplot(fig2)