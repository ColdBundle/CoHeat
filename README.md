# CoHeat 
This project is a PyTorch implementation of CoHeat which is submitted to The Web Conference 2024.

## Prerequisties 
The implementation is based on Python 3.10 and PyTorch 2.0.1
A complete list of required packages can be found in the `requirements.txt` file.
Please install the necessary packages before running the code.

## Datasets
We use 3 datasets in our work: Youshu, NetEase, and iFashion.
The preprocessed dataset is included in the repository: `./data`.
We separate the dataset into three scenarios: cold, warm, and all.

## Configuration
To customize the configuration, please edit the `./src/config.yaml` file.
For guidance on setting the hyperparameters, please refer to our paper.

## Running the code
To execute the code, use the command `python main.py` with the arguments `--data` and `--seed`.
For convenience, we provide a `demo.sh` script that reproduces the experiments presented in our work.