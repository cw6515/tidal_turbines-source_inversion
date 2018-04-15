# tidal_turbines-source_inversion
A repository for solving tidal stream turbine optimization with mixed-integer pde-constrained optimization.

### Project Description
This repository contains files used in an Imperial College MSc project entitled 
'Mixed-Integer PDE-Constrained Optimization'. All algorithms referred to and parameters are specified therein.
There are two main modules, one for running various optimization algorithms for the Source Inversion problem 
in two dimensions, and another for running optimization algorithms for tidal stream turbine optimization.

### Pre-requisites and installation steps
- Python 2.7.12
- Anaconda for python
- The contents of the requirements.yaml file

1) If you do not have the open-source package manager anaconda, install it by going to 
https://docs.continuum.io/anaconda/install/ and following the relevant links for your operating system.

2) Once installed, create a new virtual environment with the project requirements by opening a terminal and running
```conda env create -f requirements.yaml tidal_farm``` where ```tidal_farm``` is the name of the new virtual environment.

3) To activate the virtual environment, open a terminal and run ```source activate tidal_farm```. To deactivate the 
environment, simply run ```source deactivate```.

### Source Inversion 
There is one main file, `source_inversion_pipeline.py`, and two helper files contained in the `source_inversion` folder.
In order to run optimization routines, simply open a terminal, activate the virtual environment, and run
```python run_source_inversion.py``` with the specified arguments. This will run the continuous algorithm (w) control, as
well as the heuristic integer algorithm with the default parameters.

The parameters are:
- n: n^2 is the total number of source functions embedded in the mesh, defaul is n=4
- m: The computational mesh is a unit square discretized into m x m regions, default is m=16
- alpha: Boolean specifying whether to take penalty parameter from precalculated table, default=True
- penalty: A boolean denoting whether to run the penalty algorithm or not, default=False
- two_step: A boolean denoting whether to run the two-step algorithm or not, default=False
- b_and_b: A boolean denoting whether to run the two-step algorithm or not, default=False
- tolerance: The tolerance to be used for BFGS/SLSQP, default=1e-4

In order to specify arguments, run the file with argparse specifications, ie:

```python source_inversion_pipeline.py --n=8 --m=32 --two_step --b_and_b```

The results of the optimization routine will be printed out to the terminal in which the file is run.

### Tidal Stream Turbine Optimization
There is one main file ```run_tidal_turbine_pipeline.py```, four folders containing files used in various computational meshes,
and a folder named `opentidalfarm_extensions`, which contains helper methods and classes which provide mixed-integer pde-constrained
optimization support for the existing opentidalfarm framework. Each mesh folder contains several files which allow for modification
or creation of new gmsh meshes aside from the four predefined ones. In order to run optimization routines, activate the virtual
environment and run ```python run_tidal_turbine_pipeline.py```. This will run the continuous algorithm with the default parameters.


The parameters are:

- turbine_num: The number of discrete turbines to use in the optimization routine, default=30
- mesh_name: The name of the folder containing the desired mesh (one of `small_mesh`, `middle_mesh`, `large_mesh`, `alt_large_mesh`,
             unless you define your own mesh, default=`small_mesh`
- discrete: A boolean denoting whether to run the discrete algorithm or no, default=False
- init_two_step: A boolean denoting whether to evaluate the profit of the optimal continuous output after it is converted to a 
                 discrete field via a stochastic algorithm, default=False
- full_two_step: A boolean denoting whether to run the discrete algorithm using the init_two_step layout as the 
                 initial turbine layout, default=False
- mipdeco: A boolean denoting whether to run the two-step mipdeco algorithm or not, default=False
- turbine_friction: The friction of an idealized turbine
- blade_radius: The radius of an idealized turbine's blade, default=8.35
- minimum_distance: Minimum distance between idealized turbines (in meters), default=40
- efficiency: Efficiency of energy extraction, default=.5
- lcoe: Levelized cost of energy (in 2016 GBP), default=107.89
- discount_rate: Value at which future cash flows are discounted, default=.1
- income_per_unit: Income per MWh of energy (GBP), determined by the UK Electricity Market Reform CfD scheme, default=330.51
- timesteps: Number of years of which to evaluate profit of the farm, default=5
- velocity: Constant velocity of inflow from the western boundary of the farm (in m/s), default=2

The results of the optimization routine will be printed out to the terminal in which the file is run.

##### Please note that much of this work is based off of the existing open-source software opentidalfarm:
Opentidalfarm has excellent documentation and may be used as a reference framework for the discrete and continuous models.


http://opentidalfarm.readthedocs.io/en/latest/
