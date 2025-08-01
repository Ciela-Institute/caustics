
Installation
============

Regular Install
---------------

The easiest way to install is to make a new ``Python`` virtual environment (``Python 3.9`` through ``3.12`` are recommended; compatibility with versions ``>=3.13`` is currently untested). Then run::

    pip install caustics

this will install all the required libraries and then install ``caustics`` and you are ready to go! You can check out the tutorials afterwards to see some of ``caustics``' capabilities.


Developer Install
-----------------

First clone the repo with::

    git clone git@github.com:Ciela-Institute/caustics.git

this will create a directory ``caustics`` wherever you ran the command. Next go into the directory and install in developer mode::

   pip install -e ".[dev]"

this will install all relevant libraries and then install ``caustics`` in an editable format so any changes you make to the code will be included next time you import the package. To start making changes you should immediately create a new branch::

   git checkout -b <new_branch_name>

you can edit this branch however you like. If you are happy with the results and want to share with the rest of the community, then follow the contributors guide to create a pull request!
