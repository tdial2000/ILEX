from setuptools import setup

setup(
    name = 'ILEX',
    version = '0.9.0',
    description = 'A Python packages for analysing High Time Resolution FRB data, dynamic spectra and polarisation.',
    url = 'https://github.com/tdial2000/ILEX',
    author = 'Tyson Dial',
    author_email = 'tdial@swin.edu.au',
    license = 'BSD',
    packages = ['ilex', 'ilex.script_core'],
    install_requires = ['numpy<=1.26.4', 'matplotlib<=3.9.1', 'PyYAML<=5.4.1', 'scipy<=1.13.1', 'bilby<=2.3.0', 
                        'RM-Tools=1.4.6', 'pyparsing=2.4.7', 'ipython<=8.18.1', 'ruamel.yaml<0.18.0', 'cmasher=1.8.0'],

    classifiers = [
        'Development Status :: 1 - Alpha Testing',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Programming Language :: Python :: 3.10+',
    ]
)
