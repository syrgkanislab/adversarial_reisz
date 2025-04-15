# How to set up a virtual environment on a SLURM-based HPC cluster
This mini-guide contains instructions for using either virtualenv or mamba to set up a virtual environment. Use whichever is more convenient, which may depend on what comes pre-installed on your HPC cluster. Credit: William Liu and Marvin Lob.

## virtualenv
To set up the environment using the `requirements.txt`, run the following code after replacing the `<...>` placeholders:
```
cd <WORKING DIRECTORY>
module load <PATH OF PYTHON INSTALLATION>
python3 -m pip install virtualenv --user
virtualenv -p python3 venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```

If you get an error saying that the directory in which virtualenv is installed is not on PATH, add it using:
```
export PATH=$PATH:<PATH OF DIRECTORY>
```
If you do not know the directory, you will be told it if you (re)install virtualenv.

To activate the environment, run:
```
cd <WORKING DIRECTORY>
module load <PATH OF PYTHON INSTALLATION>
source venv/bin/activate
```

## mamba
### Step 1: Set Up the Environment
To create and set up the environment using `requirements.txt`, run the following commands  
(replace `<...>` with your specific values):

```
cd <WORKING_DIRECTORY> 
module load <PATH_TO_PYTHON_MODULE> 
mamba create -n <ENV_NAME> python=3.10.16 
mamba activate <ENV_NAME> 
python3 -m pip install -r requirements.txt
```

### Step 2: Configure Your `.sbatch` Script
Make sure to activate the environment and load the correct Python version within your SLURM `.sbatch` script:

```
module load python/3.10.16-fasrc01 
source ~/.bashrc 
eval "$(conda shell.bash hook)" 
conda activate <ENV_NAME>
```
