# Adversarial estimation of Riesz representers

## Instructions to reproduce the results
To reproduce the results of the paper:[^*]
1. Clone this repository. (If using a HPC cluster, you should upload this repository to your HPC cluster.)
2. Use the `requirements.txt` file to create a virtual environment in the directory containing the respository files. See `README2.txt` for help on this. If you are using a HPC cluster, you will need to first change the working directory to the directory containing the the repository files and then create the virtual environment. To change the working directory on a SLURM-based HPC cluster, use
```
cd <PATH OF WORKING DIRECTORY>
```
3. Create a directory named `gcv_results` in the working directory if it is not already present.
4. Run `Results_GCV.py`. Due to the computational intensiveness of the code, it is recommended to do this on a HPC cluster using a batch file. The batch file template `advriesz_gcv.sbatch` is provided for convenience – simply replace the `<...>` placeholders. To run `Results_GCV.py` on the HPC cluster on a SLURM-based HPC cluster, run
```
sbatch <PATH TO advriesz_gcv.sbatch>
```
5. If using a HPC cluster, download the `gcv_results` directory to your local device. In the directory containing the `gcv_results` directory, run the code in `format_simulation_results.ipynb` and `format_empirical_results.ipynb` to format the results for the tables and produce figures.


The code is confirmed to work with Python 3.10.16.

The empirical data used in this project is the Karlan and List (2007) dataset, publicly available from [this link](https://github.com/gsbDBI/ExperimentData/raw/master/Charitable/RawData/AER%20merged.dta). For convenience, a copy of it is already contained within this repository, and you do not need to separately download it.
