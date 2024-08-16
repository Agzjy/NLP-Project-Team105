# Lyrics Generator

This repository contains the implementation and resources for the Lyrics Generator model based on NanoGPT. The model is trained to generate song lyrics using [Spotify Million Song dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset). Below is a brief overview of the files included in this directory and instructions on how to use the model.

## Files

- **`demo.ipynb`**: This Jupyter notebook provides a demonstration of the Lyrics Generator model in action. It walks through the process of loading the model, and generating lyrics.

- **`lyric_generator_model.pth`**: The pre-trained model file for the Lyrics Generator. This file contains the weights and configuration needed to reproduce the results demonstrated in the `demo.ipynb` notebook.

- **`nanogpt-lyrics.ipynb`**: This notebook contains the main training and fine-tuning code for the Lyrics Generator model. It includes data preprocessing, model configuration, training loop, and evaluation.

- **`unit_test.ipynb`**: This notebook includes unit tests for verifying the functionality of the Lyrics Generator model. It ensures that the key components of the model perform as expected.

## How to Use

1. **Load the Model**: Open the `demo.ipynb` notebook and execute the cells to load the pre-trained model (`lyric_generator_model.pth`).

2. **Generate Songs**: Use the provided `generate_song` function in the `demo.ipynb` notebook to generate lyrics. You can generate multiple songs by calling the function multiple times, each time producing different results based on the model’s predictions.

3. **Fine-tune the Model**: If you wish to fine-tune the model with your own data or adjust the training parameters, you can do so by modifying the `nanogpt-lyrics.ipynb` notebook.

4. **Run Unit Tests**: To ensure everything is functioning correctly, run the cells in the `unit_test.ipynb` notebook. This will execute a series of tests to validate the model's performance.

## Prerequisites

- Python 3.8 or higher
- PyTorch
- Jupyter Notebook
- Required Python libraries (see `requirements.txt` if available)

## Getting Started

1. Clone the repository to your local machine.
2. Install the necessary dependencies.
3. Open the `demo.ipynb` notebook to start generating lyrics.

