from setuptools import setup

setup(
   name='ikea',
   version='0.2',
   description='Python implementation of the Flexible ES-MDA',
   keywords='ensemble smoother data assimilation',
   author='Frederick Bennett',
   author_email='frederick.bennett@des.qld.gov.au',
   packages=['ikea'],  #same as name
   install_requires=[
            'numpy',
            'pandas',
            'matplotlib',
            'scipy'
    ]
)