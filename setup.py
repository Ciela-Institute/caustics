import os
from setuptools import setup, find_packages
import caustic.__init__ as caustic

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()
def read_lines(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).readlines()

setup(
    name = "caustic",
    version=caustic.__version__,
    description="A gravitational lensing simulator for the future",
    long_description=read("README.md"),
    long_description_content_type='text/markdown',
    url="https://github.com/Ciela-Institute/caustic",
    author=caustic.__author__,
    license="MIT license",
    packages=find_packages(),
    install_requires=read_lines("requirements.txt"),
    keywords = [
        "gravitational lensing",
        "astrophysics",
        "differentiable programming",
        "pytorch",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
    ],    
)
