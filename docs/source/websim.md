# Live Caustics Playground

Here is a little online simulator run using caustics! It's slow because it's
running on a free server, but it's a good way to play with the simulator without
having to install anything.

Pro tip: check out the "Pixelated" source to lens any image you want!

<div style="display: flex; height: 500px;">
    <iframe src="https://ciela-institute-caustics-webapp-guistreamlit-app-yanhhm.streamlit.app/?embed=true" style="flex-grow: 1; width: 100%;"></iframe>
</div>

If the window doesn't show up just follow this link:
[caustics-webapp](https://ciela-institute-caustics-webapp-guistreamlit-app-yanhhm.streamlit.app)

For frequent simulator users (e.g., if you plan on exploring the parameter space
of a lens), we recommend installing the simulator locally and running it in your
browser. Follow the steps below:

1. Install Caustics. Please follow the instructions on the
   [install page](https://caustics.readthedocs.io/en/latest/install.html).
2. `pip install streamlit`
3. `git clone https://github.com/Ciela-Institute/caustics-webapp.git`
4. Move into the `caustics-webapp/gui/` directory and run the following command:
   `streamlit run streamlit_app.py --server.enableXsrfProtection=false`

If you were successful in installing the simulator, Step 4 should automatically
open the simulator in your default browser.
