import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np
from caustics.utils import get_meshgrid
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SIE, EPL, ExternalShear, SinglePlane
from caustics.light import sersic
import caustics

def caustic_trace(delta_m, x, z_background = 3.042, simulation_size = 650, H0=67, Om0=0.32, Tcmb0=2.725, upsample_factor = 1, device = "cpu"):
    
    cosmology = FlatLambdaCDM(name = "cosmo")
    cosmology.to(dtype=torch.float32)
    epl = EPL(name = "epl", cosmology = cosmology)
    shear = ExternalShear(name = "shear", cosmology = cosmology)
    lens = SinglePlane(lenses = [epl, shear], cosmology = cosmology)
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

st.set_page_config(layout="wide")
css='''
<style>
    section.main > div {max-width:75rem}
</style>
'''
st.markdown(css, unsafe_allow_html=True)
logo_url = "https://github.com/Ciela-Institute/caustics/raw/main/media/caustics_logo_white.png?raw=true"
st.sidebar.image(logo_url)
docs_url = "https://caustics.readthedocs.io/"
st.sidebar.write("Check out the [documentation](%s)!" % docs_url)
user_menu = st.sidebar.radio("Select an Option", ("EPL + Shear + Sersic","Other lenses to follow"))

if user_menu == "EPL + Shear + Sersic":
    st.title("Caustics Gravitational Lensing Simulator")
    st.header("EPL + Shear Lens and Sersic Source")

    # Create a two-column layout
    col1, col2, col3 = st.columns([4, 4, 5])

    # Sliders for lens parameters in the first column
    with col1:
        st.header(r"$\textsf{\tiny Lens Parameters}$", divider = "blue")
        z_lens = st.slider("Lens redshift", min_value = 0., max_value = 10., step = 0.01)
        x0 = st.slider("EPL X position", -2.0, 2.0, 0.0)
        y0 = st.slider("EPL Y position", -2.0, 2.0, 0.25)
        q = st.slider("EPL axis ratio", 0.1, 1.0, 0.82)
        phi = st.slider("EPL rotation angle on sky", 0.0, 2*np.pi, 8*(np.pi/180) + np.pi/2)
        theta_E = st.slider("EPL Einstein radius", 0.0, 2.0, 1.606)
        t = st.slider("EPL power law slope ($\gamma - 1$)", 0.0, 2.0, 1.0)
        shearx = st.slider("Shear x position", -1.0, 1.0, 0.01)
        sheary = st.slider("Shear y position", -1.0, 1.0, 0.0)
        gamma1 = st.slider("Shear first component", -1.0, 1.0, 0.036*np.cos(2*3*(np.pi/180)))
        gamma2 = st.slider("Shear second component", -1.0, 1.0, 0.036*np.sin(2*3*(np.pi/180)))

    with col2:
        st.header(r"$\textsf{\tiny Source Parameters}$", divider = "blue")
        z_source = st.slider("Source redshift", min_value = z_lens, max_value = 10., step = 0.01)
        src_x0 = st.slider("Sersic x position", -2.0, 2.0, 0.0)
        src_y0 = st.slider("Sersic y position", -2.0, 2.0, -0.2+0.25)
        src_q = st.slider("Sersic axis ratio", 0.1, 1.0, 0.5)
        src_phi = st.slider("Sersic rotation angle on sky", 0.0, 2*np.pi, 0.0)
        src_n = st.slider("Sersic Sersic index", 0.1, 10.0, 0.8)
        src_Re = st.slider("Sersic scale length", 0.0, 2.0, 1.25)
        src_Ie = st.slider("Sersic intensity", 0.0, 2.0, 0.3)

    x = torch.tensor(
        [
            z_source, # Source z
            z_lens, # Lens z
            z_lens, # Lens z for EPL
            x0,  #x0
            y0,  #y0
            q, #Minor/major axs
            phi, #Angle
            theta_E, #Einstein radius
            t, #Gamma -1 
            z_lens, # Lens z for Shear
            shearx, # Shear x
            sheary, #Shear y
            gamma1, #Gamma_1
            gamma2,    #Gamma_2
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
    cosmology = FlatLambdaCDM(name="cosmo")
    epl = EPL(name="epl", cosmology=cosmology)
    shear = ExternalShear(name="shear", cosmology=cosmology)
    lens = SinglePlane(lenses=[epl, shear], cosmology=cosmology)
    src = sersic.Sersic(name="src")
    minisim = caustics.Lens_Source(lens=lens, source=src, pixelscale=deltam, pixels_x=650)
    y1s, y2s = caustic_trace(delta_m=deltam, x=x[1:14], z_background=z_source)

    # Plot the caustic trace and lensed image in the second column
    with col3:
        st.header(r"$\textsf{\tiny Visualization}$", divider = "blue")

        # Plot the unlensed image
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        ax2.set_title("Unlensed source and caustic", fontsize = 15)
        ax2.imshow(minisim(x, lens_source=False), origin="lower")
        ax2.plot(y1s, y2s, "-r")
        ax2.set_xticks(ticks = np.linspace(0, 650, 5).astype(int), labels = np.round(np.linspace(-650*deltam/2, 650*deltam/2, 5), 3), fontsize = 15)
        ax2.set_xlabel("Arcseconds from center", fontsize = 15)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.set_yticks(ticks = np.linspace(0, 650, 5).astype(int)[1:], labels = np.round(np.linspace(-650*deltam/2, 650*deltam/2, 5), 3)[1:], fontsize = 15, rotation=90)
        ax2.set_ylabel("Arcseconds from center", fontsize = 15)
        st.pyplot(fig2)


        fig1, ax1 = plt.subplots(figsize=(7, 7))
        ax1.set_title("Lens and caustic", fontsize = 15)
        ax1.plot(y1s, y2s, "-r")
        ax1.imshow(minisim(x, lens_source=True), origin="lower")
        ax1.set_xticks(ticks = np.linspace(0, 650, 5).astype(int), labels = np.round(np.linspace(-650*deltam/2, 650*deltam/2, 5), 3), fontsize = 15)
        ax1.set_xlabel("Arcseconds from center", fontsize = 15)
        ax1.yaxis.set_label_position("right")
        ax1.yaxis.tick_right()
        ax1.set_yticks(ticks = np.linspace(0, 650, 5).astype(int)[1:], labels = np.round(np.linspace(-650*deltam/2, 650*deltam/2, 5), 3)[1:], fontsize = 15, rotation=90)
        ax1.set_ylabel("Arcseconds from center", fontsize = 15)
        st.pyplot(fig1)
        
if user_menu == "Other lenses to follow":
    st.title("Caustics Gravitational Lensing Simulator")
    st.header("More lens configurations are on their way!")