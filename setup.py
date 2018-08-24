#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='mwtools',
      version='0.1',
      description='Tools for dealing with Milky Way data',
      author='Chris Wegg',
      author_email='chriswegg+mwtools@gmail.com',
      url='https://gitlab.com/chriswegg/mwtools',
      packages=find_packages(),
      install_requires=['python-dotenv', 'astropy', 'astroquery', 'galpy', 'pandas', 'requests','numpy']
      )

print("""
-----------------------------------------------------------------------------------------
If you didnt install nemo, and want to use it to compute potentials then do it now!
Then you also need to update the .env file with the paths to nemo, and the dehnen 
executables as NEMO_LOCATION and NEMO_DEHNEN_LOCATION
------------------------------------------------------------------------------------------""")