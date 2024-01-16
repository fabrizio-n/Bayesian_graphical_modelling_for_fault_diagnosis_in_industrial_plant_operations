# Bayesian graphical modelling for fault diagnosis in industrial plant operations

## Description

This is a python implementation of the algorithm for "Newtork structure learning under uncertain interventions", a bayesian mcmc algorithm developed by Castelletti and Peluso.

## Installation

Packages required: 

- Numpy
- SciPy
- Pandas
- tqdm
- Sklearn

## Project structure

- data_analysis: contains data pre-processing code
- python_mcmcdagtargets
    - mcmcdagtargets.py is the main implementaion script of the algorithm and relative utility functions. 
    - orc_study.ipynb: application of the algorithm on real data
    - simulation.py: contains simulation function 
    - simulation_study.ipynb: contains a simulation study on 4 synthetic datasets


