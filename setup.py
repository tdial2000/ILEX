from setuptools import setup
import os

package_list = ['ilex', 'ilex.script_core', 'ilex.addons']
# check if SHRINE code is installed
if os.path.isdir("./src/SHRINE"):
    print("Found source code for [SHRINE], adding to installed package list!")
    package_list += ['src.SHRINE']

setup(
    name = 'ILEX',
    version = '1.0.0',
    description = 'A Python packages for analysing High Time Resolution FRB data, dynamic spectra and polarisation.',
    url = 'https://github.com/tdial2000/ILEX',
    author = 'Tyson Dial',
    author_email = 'tdial@swin.edu.au',
    license = 'BSD',
    packages = package_list,
    package_data = {
            "ILEX": ["files/*"],
    },
    install_requires = ['numpy==1.26.4', 'matplotlib==3.9.1', 'PyYAML==6.0.1', 'scipy==1.13.1', 'bilby==2.3.0', 
                        'RM-Tools==1.4.6', 'pyparsing==2.4.7', 'ipython==8.18.1', 'ruamel.yaml==0.18.0', 'cmasher==1.8.0'],

    classifiers = [
        'Development Status :: 1 - Alpha Testing',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Programming Language :: Python :: 3.9+',
    ]
)
