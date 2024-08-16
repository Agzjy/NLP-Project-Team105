# NanoGPT from Scratch

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Details](#model-details)
    - [Tiny Shakespeare](https://github.com/Agzjy/NLP-Project-Team105/tree/master/NanoGPT%20Models/Tiny-Shakespeare)
    - [Lyrics Generator](https://github.com/Agzjy/NLP-Project-Team105/tree/master/NanoGPT%20Models/Lyrics%20Generator)
    - [Story Generator](https://github.com/Agzjy/NLP-Project-Team105/tree/master/NanoGPT%20Models/Story%20Generator)
6. [Results](#results)
7. [Contributing](#contributing)

## Overview

This project focuses on the implementation of **NanoGPT** from scratch, a streamlined and efficient adaptation of the GPT (Generative Pre-trained Transformer) architecture. Designed to generate coherent and contextually relevant text with minimal computational overhead, NanoGPT serves as a versatile tool for various text generation tasks.

## Requirements

To run the scripts, you will need Python 3 and the following dependencies:

- PyTorch (< 3.0)
- NumPy
- HuggingFace Transformers (< 3.0)
- HuggingFace Datasets (< 3.0)
- TikToken
- tqdm

## Installation

Clone the repository to your local machine and install the required dependencies:

```bash
git clone https://github.com/yourusername/NanoGPT-Project.git
cd NanoGPT-Project
pip install torch==2.5.0
pip install numpy transformers==2.11.0 datasets==2.0.0 tiktoken tqdm
```

## Usage

### Running Models

Each model has its own directory under `NanoGPT Models`. To run a model, navigate to its directory and follow the instructions in the corresponding `README.md` file.

### Example:

```bash
cd NanoGPT-Project/NanoGPT Models/Lyrics Generator
# Open demo.ipynb in Jupyter Notebook or run any of the scripts
```

## Model Details
**Tiny Shakespeare**:

NanoGPT was trained on the [Tiny Shakespeare dataset](https://www.kaggle.com/datasets/thedevastator/the-bards-best-a-character-modeling-dataset) to generate Shakespeare-like text.

Training Script: train.py
Model Script: model.py
Configuration: config/train_shakespeare_char.py
ReadMe: Detailed instructions are in the README.md file within the directory.

**Lyrics Generator**:

NanoGPT was trained on the [Spotify Million Song Dataset](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset) to generate song lyrics.

Jupyter Notebooks: demo.ipynb, nanogpt-lyrics.ipynb
Unit Tests: unit_test.ipynb
ReadMe: Detailed instructions are in the README.md file within the directory.

**Story Generator**:

NanoGPT was trained on the [Story Cloze dataset](https://huggingface.co/datasets/LSDSem/story_cloze) to generate 5-sentence stories based on prompts.

Story Generator Script: story_generator.py
Loading Model Script: load_model.py
Unit Tests: unit_test.py
ReadMe: Detailed instructions are in the README.md file within the directory.

## Results
The performance of NanoGPT was evaluated using perplexity as the primary metric. Detailed results for each model can be found in the respective directories.

## Contribution
Pooja Laxmi Sankarakameswaran, Poornima Jaykumar Dharamdasani, and Jiaying Zheng.
