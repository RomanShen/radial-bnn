<div align="center">

# Radial Bayesian Neural Networks

</div>
 
## Description   
This repository contains the code for the paper [Radial Bayesian Neural Networks: Beyond Discrete Support in Large-Scale Bayesian Deep Learning](https://arxiv.org/abs/1907.00865).

We only run experiments on the MNIST dataset.

## How to run   
First, install dependencies   
```bash
# clone src   
git clone https://github.com/RomanShen/radial-bnn.git

# install dependencies 
cd radial-bnn
pip install -r requirements.txt
 ```   
Next, run either convolutional or radial version MNIST experiments.   
 ```bash
# convolutional version
python run_conv.py    
```
For multiple runs with different seeds, go to [WandB Sweeps](https://docs.wandb.ai/guides/sweeps/quickstart) for help.

Basically, run following commands for convolutional version.
```bash
wandb sweep sweep_conv.yaml
wandb agent your-sweep-id
```

## Results
All experimental results are available online [here](https://wandb.ai/xqshen/radial-bnn?workspace=user-xqshen).
### Citation   
```
@InProceedings{pmlr-v108-farquhar20a, 
title = {Radial Bayesian Neural Networks: Beyond Discrete Support In Large-Scale Bayesian Deep Learning}, author = {Farquhar, Sebastian and Osborne, Michael A. and Gal, Yarin}, booktitle = {Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics}, 
pages = {1352--1362}, 
year = {2020}, 
editor = {Silvia Chiappa and Roberto Calandra}, 
volume = {108}, 
series = {Proceedings of Machine Learning Research}, 
month = {26--28 Aug}, 
publisher = {PMLR}, 
pdf = {http://proceedings.mlr.press/v108/farquhar20a/farquhar20a.pdf}, 
url = { http://proceedings.mlr.press/v108/farquhar20a.html }, 
```   
