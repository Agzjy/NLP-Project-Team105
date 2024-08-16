# Story Generator Project

This repository contains a GPT-based story generator that leverages a SentencePiece tokenizer for preprocessing and a Transformer architecture for generating coherent stories based on a prompt. The project includes the following components:

## Folder Contents

1. **`story_generator.py`**: 
   - The core script that trains the story generation model using a GPT-like architecture. The training process is designed to generate a sequence of tokens given a prompt, which is then converted into a human-readable story.

2. **`spm_model.model`**:
   - The SentencePiece tokenizer model used for encoding and decoding text. This model is essential for both training and inference stages.

3. **`unit_test.py`**:
   - The unit testing script that includes various test cases to validate the different functions in `story_generator.py`. You can run this script using:
     ```bash
     python unit_test.py
     ```

4. **`demo.py`**:
   - A demo script that contains 3 example test cases for story generation. You can run this file as is or modify the test cases as needed. To run the demo:
     ```bash
     python demo.py
     ```

5. **`load_model.py`**:
   - A script that handles loading the trained model with the correct tokenizer and weights for inference.

6. **`story_generator.ipynb`**:
   - The Jupyter Notebook containing the model training code. Executing this notebook trains the story generation model and generates the weights used during inference.

## Setup Instructions

1. **Cloning the Repository**:
   ```bash
   git clone https://github.com/Agzjy/NLP-Project-Team105.git
   cd NLP-Project-Team105
