from setuptools import setup

setup(
    name = 'ILEX',
    version = '0.9.0',
    description = 'A Python packages for analysing High Time Resolution FRB data, dynamic spectra and polarisation.',
    url = 'https://github.com/tdial2000/ILEX',
    author = 'Tyson Dial',
    author_email = 'tdial@swin.edu.au',
    license = 'BSD',
    packages = ['ilex'],
    install_requires = ['numpy', 'matplotlib', 'PyYAML', 'scipy', 'bilby', 'RM-Tools', 'pyparsing', 'ipython'],

    classifiers = [
        'Development Status :: 1 - Alpha Testing',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Programming Language :: Python :: 3.10+',
    ]
)