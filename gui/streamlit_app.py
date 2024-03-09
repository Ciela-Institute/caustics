import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np
from caustics.utils import get_meshgrid
from caustics.cosmology import FlatLambdaCDM
from caustics.lenses import SinglePlane
import caustics
from app_configs import (
    lens_slider_configs,
    source_slider_configs,
    name_map,
    default_params,
)


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
    paths = CS.allsegs[0]
    x1s = []
    x2s = []
    y1s = []
    y2s = []
    for path in paths:
        # Collect the path into a discrete set of points
        x1 = torch.tensor(list(float(vs[0]) for vs in path), device=device)
        x2 = torch.tensor(list(float(vs[1]) for vs in path), device=device)
        # raytrace the points to the source plane
        y1, y2 = lens.raytrace(x1, x2, z_s, params=lens.pack(x))
        y1s.append((y1.cpu() / res + simulation_size / 2).numpy())
        y2s.append((y2.cpu() / res + simulation_size / 2).numpy())
        x1s.append((x1.cpu() / res + simulation_size / 2).numpy())
        x2s.append((x2.cpu() / res + simulation_size / 2).numpy())

    plt.close()

    return x1s, x2s, y1s, y2s


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
lens_menu = st.sidebar.multiselect(
    "Select your Lens(es)", lens_slider_configs.keys(), default=["EPL", "Shear"]
)
source_menu = st.sidebar.radio(
    "Select your Source (more to come)", source_slider_configs.keys()
)
st.sidebar.write(
    "Note: if you see an error about contour plots, just reload the webpage and it will go away."
)

st.title("Caustics Gravitational Lensing Simulator")
st.header(f"{'+'.join(lens_menu)} and {source_menu} Source")
simulation_size = st.number_input("Simulation resolution", min_value=64, value=256)
fov = 6.5
deltam = fov / simulation_size
# Create a two-column layout
col1, col2, col3 = st.columns([4, 4, 5])

# Sliders for lens parameters in the first column
with col1:
    st.header(r"$\textsf{\tiny Lens Parameters}$", divider="blue")
    # z_lens = st.slider("Lens redshift", min_value=0.0, max_value=10.0, step=0.01)
    x_lens = []
    for lens in lens_menu:
        for param, label, bounds in lens_slider_configs[lens]:
            x_lens.append(
                st.slider(
                    label, min_value=bounds[0], max_value=bounds[1], value=bounds[2]
                )
            )

    x_lens = torch.tensor(x_lens)

with col2:
    st.header(r"$\textsf{\tiny Source Parameters}$", divider="blue")
    # z_source = st.slider("Source redshift", min_value=z_lens, max_value=10.0, step=0.01)
    x_source = []
    for param, label, bounds in source_slider_configs[source_menu]:
        x_source.append(
            st.slider(label, min_value=bounds[0], max_value=bounds[1], value=bounds[2])
        )
    x_source = torch.tensor(x_source)
x_all = torch.cat((x_lens, x_source))
z_lens = 1.0
z_source = 2.0
cosmology = FlatLambdaCDM(name="cosmo")
lenses = []
for lens in lens_menu:
    lenses.append(name_map[lens](cosmology, **default_params[lens], z_l=z_lens))
lens = SinglePlane(lenses=lenses, cosmology=cosmology, z_l=z_lens)
src = name_map[source_menu](name="src", **default_params[source_menu])
minisim = caustics.Lens_Source(
    lens=lens, source=src, pixelscale=deltam, pixels_x=simulation_size, z_s=z_source
)
x1s, x2s, y1s, y2s = caustic_critical_line(
    lens=lens, x=x_lens, z_s=z_source, res=deltam, simulation_size=simulation_size
)

# Plot the caustic trace and lensed image in the second column
with col3:
    st.header(r"$\textsf{\tiny Visualization}$", divider="blue")

    # Plot the unlensed image
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.set_title("Unlensed source and caustic", fontsize=15)
    ax2.imshow(minisim(x_all, lens_source=False), origin="lower", cmap="inferno")
    for c in range(len(y1s)):
        ax2.plot(y1s[c], y2s[c], "-w")
    ax2.set_xticks(
        ticks=np.linspace(0, simulation_size, 5).astype(int),
        labels=np.round(
            np.linspace(-simulation_size * deltam / 2, simulation_size * deltam / 2, 5),
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
            np.linspace(-simulation_size * deltam / 2, simulation_size * deltam / 2, 5),
            3,
        )[1:],
        fontsize=15,
        rotation=90,
    )
    ax2.set_ylabel("Arcseconds from center", fontsize=15)
    st.pyplot(fig2)

    fig1, ax1 = plt.subplots(figsize=(7, 7))
    ax1.set_title("Lens and critical curve", fontsize=15)
    for c in range(len(x1s)):
        ax1.plot(x1s[c], x2s[c], "-w")
    ax1.imshow(minisim(x_all, lens_source=True), origin="lower", cmap="inferno")
    ax1.set_xticks(
        ticks=np.linspace(0, simulation_size, 5).astype(int),
        labels=np.round(
            np.linspace(-simulation_size * deltam / 2, simulation_size * deltam / 2, 5),
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
            np.linspace(-simulation_size * deltam / 2, simulation_size * deltam / 2, 5),
            3,
        )[1:],
        fontsize=15,
        rotation=90,
    )
    ax1.set_ylabel("Arcseconds from center", fontsize=15)
    st.pyplot(fig1)
