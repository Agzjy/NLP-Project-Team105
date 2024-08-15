# Model Testing and Training Guide

This repository contains the necessary scripts and configurations to test and train a machine learning model. Please follow the instructions below to get started.

## Requirements
To run the scripts, you will need Python 3 and the following dependencies:

- PyTorch (< 3.0)
- NumPy
- HuggingFace Transformers (< 3.0)
- HuggingFace Datasets (< 3.0)
- TikToken
- tqdm

## Installation

Clone the repository to your local machine:

pip install torch==2.5.0
pip install numpy transformers==2.11.0 datasets==2.0.0 tiktoken tqdm

## Testing the model

python3 test.py

## Train the model

### Prepare the data and train the model

python3 prepare.py

python3 train.py config/train_shakespeare_char.py




