# BioMolX - Biological Complexity Curriculum Learning for Molecular Activity Prediction in Humans

## Overview
This project contains the implementation of the model associated with the paper ``Biological Complexity Curriculum Learning for Molecular Activity Prediction in Humans``.
The checkpoints for all the models presented in the paper can be found in the ``results`` folder.

## Usage

### Pre-trained Models
If you only want to use the pre-trained models, you can find the checkpoints in the ``results`` folder. These models are ready for evaluation or prediction.

### Train Your Own Model
If you want to train your own model, follow the steps below:

1. **Set Up Environment:**
   - Create a virtual environment using the provided `env.yml` file:
     ```bash
     conda env create -f env.yml
     ```

2. **Prepare Datasets:**
   - Navigate to the ``datasets`` folder.
   - Unzip the dataset corresponding to the tissue you are interested in.

3. **Run Training:**
   - Go back to the main project directory.
   - Open `main.py` and scroll down to the main command.
   - Choose the tissue you want to train on by uncommenting the corresponding line.
   - Run the training script using the following command:
     ```bash
     python main.py
     ```
