# Off Environment Evaluation using Convex Risk Minimization

## Installation Instructions
**Requirements**
- Python 3
- Pytorch 1.5.1
- gym 
- pybullet
for the rest of the packages simply follow the commands below  
```
git clone https://github.com/pulkitkatdare/offenveval.git
cd offenveval
pip install -r requirements.txt 
```

## Training Instructions
We are releasing code for 5 different environments (beta, reacher, cartpole, gridworld, reacher)
To run either of this code run the following
- To run *beta* experiment
```
python -m experiments.train.beta
```
- To run *dartenv* experiment
```
python -m experiments.train.dartenv
```
- To run *gridworld* experiment
```
python -m experiments.train.gridworld
```
- To run *cartpole* experiment
```
python -m experiments.train.cartpole
```
- To run *reacher* experiment
```
python -m experiments.train.reacher
```
Configuration files for each of these experiments are in their respective folders. For example, for the gridworld they can be found in *./gridworld/config.py* file
To play around these parameters, feel free to change parameters in the config files and re-run the code again



