import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np
from caustics.utils import get_meshgrid
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import EPL, ExternalShear, SinglePlane
from caustics.light import sersic
import caustics


def caustic_critical_line(
    lens, x, z_s, res, simulation_size, upsample_factor=1, device="cpu"
):
    thx, thy = get_meshgrid(
        res / upsample_factor,
        upsample_factor * simulation_size,
        upsample_factor * simulation_size,
        dtype=torch.float32,
        device=device,
    )
    A = lens.jacobian_lens_equation(thx, thy, z_s, lens.pack(x))
    # Note that if this is too slow you can set `method = "finitediff"` to run a faster version. You will also need to provide `pixelscale` then

    # Here we compute A's determinant at every point
    detA = torch.linalg.det(A)

    # Generate caustic
    CS = plt.contour(thx, thy, detA, levels=[0.0], colors="b")
    paths = CS.collections[0].get_paths()
    y1s = []
    y2s = []
    for path in paths:
        # Collect the path into a discrete set of points
        vertices = path.interpolated(5).vertices
        x1 = torch.tensor(list(float(vs[0]) for vs in vertices), device=device)
        x2 = torch.tensor(list(float(vs[1]) for vs in vertices), device=device)
        # raytrace the points to the source plane
        y1, y2 = lens.raytrace(x1, x2, z_s, params=lens.pack(x))
        y1s += y1.cpu() / res + simulation_size / 2
        y2s += y2.cpu() / res + simulation_size / 2

    plt.close()

    d_x = res * simulation_size / (thx.cpu().max() - thx.cpu().min())
    d_y = res * simulation_size / (thy.cpu().max() - thy.cpu().min())
    xcoords = (
        thx.cpu() * simulation_size / (thx.cpu().max() - thx.cpu().min())
        + simulation_size / 2
        - d_x
    )
    ycoords = (
        thy.cpu() * simulation_size / (thy.cpu().max() - thy.cpu().min())
        + simulation_size / 2
        - d_y
    )
    return xcoords, ycoords, detA, np.array(y1s), np.array(y2s)


st.set_page_config(layout="wide")
css = """
<style>
    section.main > div {max-width:75rem}
</style>
"""
st.markdown(css, unsafe_allow_html=True)
logo_url = "https://github.com/Ciela-Institute/caustics/raw/main/media/caustics_logo_white.png?raw=true"
st.sidebar.image(logo_url)
docs_url = "https://caustics.readthedocs.io/"
st.sidebar.write("Check out the [documentation](%s)!" % docs_url)
user_menu = st.sidebar.radio(
    "Select an Option", ("EPL + Shear + Sersic", "Other lenses to follow")
)
st.sidebar.write(
    "Note: if you see an error about contour plots, just reload the webpage and it will go away."
)

if user_menu == "EPL + Shear + Sersic":
    st.title("Caustics Gravitational Lensing Simulator")
    st.header("EPL + Shear Lens and Sersic Source")
    simulation_size = st.number_input("Simulation resolution", min_value=64, value=256)
    fov = 6.5
    deltam = fov / simulation_size
    # Create a two-column layout
    col1, col2, col3 = st.columns([4, 4, 5])

    # Sliders for lens parameters in the first column
    with col1:
        st.header(r"$\textsf{\tiny Lens Parameters}$", divider="blue")
        # z_lens = st.slider("Lens redshift", min_value=0.0, max_value=10.0, step=0.01)
        z_lens = 0
        x0 = st.slider("EPL X position", -2.0, 2.0, 0.0)
        y0 = st.slider("EPL Y position", -2.0, 2.0, 0.25)
        q = st.slider("EPL axis ratio", 0.1, 1.0, 0.82)
        phi = st.slider(
            "EPL rotation angle on sky", 0.0, np.pi, 8 * (np.pi / 180) + np.pi / 2
        )
        theta_E = st.slider("EPL Einstein radius", 0.0, 2.0, 1.606)
        t = st.slider("EPL power law slope ($\gamma - 1$)", 0.0, 2.0, 1.0)
        # shearx = st.slider("Shear x position", -1.0, 1.0, 0.01)
        shearx = 0
        # sheary = st.slider("Shear y position", -1.0, 1.0, 0.0)
        sheary = 0
        gamma1 = st.slider(
            "Shear first component",
            -1.0,
            1.0,
            0.036 * np.cos(2 * 3 * (np.pi / 180)),
        )
        gamma2 = st.slider(
            "Shear second component",
            -1.0,
            1.0,
            0.036 * np.sin(2 * 3 * (np.pi / 180)),
        )

    with col2:
        st.header(r"$\textsf{\tiny Source Parameters}$", divider="blue")
        # z_source = st.slider("Source redshift", min_value=z_lens, max_value=10.0, step=0.01)
        z_source = 1
        src_x0 = st.slider("Sersic x position", -2.0, 2.0, 0.0)
        src_y0 = st.slider("Sersic y position", -2.0, 2.0, -0.2 + 0.25)
        src_q = st.slider("Sersic axis ratio", 0.1, 1.0, 0.5)
        src_phi = st.slider("Sersic rotation angle on sky", 0.0, np.pi, 0.0)
        src_n = st.slider("Sersic Sersic index", 0.1, 10.0, 0.8)
        src_Re = st.slider("Sersic scale length", 0.0, 2.0, 1.25)
        # src_Ie = st.slider("Sersic intensity", 0.0, 2.0, 0.3)
        src_Ie = 10.0

    x = torch.tensor(
        [
            z_source,  # Source z
            z_lens,  # Lens z
            z_lens,  # Lens z for EPL
            x0,  # x0
            y0,  # y0
            q,  # Minor/major axs
            phi,  # Angle
            theta_E,  # Einstein radius
            t,  # Gamma -1
            z_lens,  # Lens z for Shear
            shearx,  # Shear x
            sheary,  # Shear y
            gamma1,  # Gamma_1
            gamma2,  # Gamma_2
            src_x0,  # src x0
            src_y0,  # src y0
            src_q,  # src q
            src_phi,  # src phi
            src_n,  # src n
            src_Re,  # src Re
            src_Ie,  # src Ie
        ]
    )

    cosmology = FlatLambdaCDM(name="cosmo")
    epl = EPL(name="epl", cosmology=cosmology)
    shear = ExternalShear(name="shear", cosmology=cosmology)
    lens = SinglePlane(lenses=[epl, shear], cosmology=cosmology)
    src = sersic.Sersic(name="src")
    minisim = caustics.Lens_Source(
        lens=lens, source=src, pixelscale=deltam, pixels_x=simulation_size
    )
    thx, thy, detA, y1s, y2s = caustic_critical_line(
        lens=lens, x=x[1:14], z_s=z_source, res=deltam, simulation_size=simulation_size
    )

    # Plot the caustic trace and lensed image in the second column
    with col3:
        st.header(r"$\textsf{\tiny Visualization}$", divider="blue")

        # Plot the unlensed image
        fig2, ax2 = plt.subplots(figsize=(7, 7))
        ax2.set_title("Unlensed source and caustic", fontsize=15)
        ax2.imshow(minisim(x, lens_source=False), origin="lower", cmap="inferno")
        ax2.plot(y1s, y2s, "-w")
        ax2.set_xticks(
            ticks=np.linspace(0, simulation_size, 5).astype(int),
            labels=np.round(
                np.linspace(
                    -simulation_size * deltam / 2, simulation_size * deltam / 2, 5
                ),
                3,
            ),
            fontsize=15,
        )
        ax2.set_xlabel("Arcseconds from center", fontsize=15)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.set_yticks(
            ticks=np.linspace(0, simulation_size, 5).astype(int)[1:],
            labels=np.round(
                np.linspace(
                    -simulation_size * deltam / 2, simulation_size * deltam / 2, 5
                ),
                3,
            )[1:],
            fontsize=15,
            rotation=90,
        )
        ax2.set_ylabel("Arcseconds from center", fontsize=15)
        st.pyplot(fig2)

        fig1, ax1 = plt.subplots(figsize=(7, 7))
        ax1.set_title("Lens and critical curve", fontsize=15)
        ax1.contour(thx, thy, detA, levels=[0.0], colors="w")
        ax1.imshow(minisim(x, lens_source=True), origin="lower", cmap="inferno")
        ax1.set_xticks(
            ticks=np.linspace(0, simulation_size, 5).astype(int),
            labels=np.round(
                np.linspace(
                    -simulation_size * deltam / 2, simulation_size * deltam / 2, 5
                ),
                3,
            ),
            fontsize=15,
        )
        ax1.set_xlabel("Arcseconds from center", fontsize=15)
        ax1.yaxis.set_label_position("right")
        ax1.yaxis.tick_right()
        ax1.set_yticks(
            ticks=np.linspace(0, simulation_size, 5).astype(int)[1:],
            labels=np.round(
                np.linspace(
                    -simulation_size * deltam / 2, simulation_size * deltam / 2, 5
                ),
                3,
            )[1:],
            fontsize=15,
            rotation=90,
        )
        ax1.set_ylabel("Arcseconds from center", fontsize=15)
        st.pyplot(fig1)

if user_menu == "Other lenses to follow":
    st.title("Caustics Gravitational Lensing Simulator")
    st.header("More lens configurations are on their way!")
