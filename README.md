# Doubly Mixed-Effects Gaussian Process Regression
![Decomposition Figure](https://github.com/SeyoungKimLab/DMGP/blob/main/figures/decomp.png)<br>
*Doubly mixed-effects Gaussian process (DMGP)* is a multi-task GP regression model that learns fixed and random effects across both samples and tasks (decomposition demonstrated in the figure above). Along with an example dataset, this repository includes implementations of:
- Doubly mixed-effects Gaussian process (DMGP)
- Translated mixed-effects Gaussian process (TMGP)
- Mixed-effects Gaussian process (MGP)  

This work is introduced in the following paper:  
> Jun Ho Yoon, Daniel P. Jeong, and Seyoung Kim. Doubly Mixed-Effects Gaussian Process Regression. *Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS)*, 2022.  

## Setup
All of the models were implemented in [GPflow](https://github.com/GPflow/GPflow) & [TensorFlow](https://www.tensorflow.org) and tested on Linux & Mac OS. The following package versions were used:

| TensorFlow Ver. | GPflow Ver. | Python Ver. | 
| --------------- | ----------- | ----------- |
| 2.4.1           | 2.1.4       | 3.8.5       |

Other dependencies:
- scikit-learn
- numpy
- pyyaml
- tqdm
- matplotlib

To set up the environment that we used, you can take the following steps:

### 1. Install Conda (Optional)
If you do not already have conda installed on your local machine, please install conda following the instructions [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

### 2. Import Conda Environment for DMGP
You can find the exported [conda](https://docs.conda.io/en/latest/) environment `.yaml` files under `./env`, which can be used to replicate the environment that we used to develop our code. To import and create a new conda environment on your local machine, run on the command line:
```
conda env create -f ./env/dmgp_env_linux.yaml (if you are using Linux)
conda env create -f ./env/dmgp_env_mac.yaml (if you are using Mac OS)
```

You can also refer to conda's [documentation on managing environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more information.

### 3. Activate the Conda Environment
After Step 2, you can check whether the environment (named `dmgp_env`) was successfully created by running:
```
conda env list
```
which lists all of the conda environments available on your local machine. If `dmgp_env` is also listed, then you can activate it by running:
```
conda activate dmgp_env
```

## Running the Code (Demo)

For demonstration, we include an example simulation dataset with 30 tasks and 100 samples under `./data`. We performed a random 80-20 split along the samples to create the train and test data.

### 1. Specifying Settings for Training (Optional)
For each model, all training settings such as learning rate, the number of inducing points, and dataset path are to be specified in the corresponding yaml file under `./dmgp/params`. For example, before running `./dmgp/train_dmgp.py`, you can edit the `./dmgp/params/train_dmgp_params.yaml` file to make changes to the default setting. 

### 2. Training
After specifying what settings to use for training, move to `./dmgp` and simply run on the command line one of the following scripts corresponding to your model of choice:

#### DMGP
```
python3 train_dmgp.py
```

#### TMGP
```
python3 train_tmgp.py
```

#### MGP
```
python3 train_mgp.py
```

All of the checkpoints and training results will be saved in the same working directory under a new folder called `./dmgp/logs`.
