# ECG Federated Learning

## Overview
This project implements Federated Learning for ECG signal classification using the PhysioNet/CinC Challenge 2017 dataset. The objective is to categorize ECG signals into three classes: Normal, Atrial Fibrillation, and Other.

## Environment
  - OS: Linux Mint 22
  - Python: 3.11.11

## How to run.
1. Download the dataset.

    ```bash
    bash download_dataset.sh
    ```
2. Install dependencies.

    ```bash
    pip install -r requirements.txt
    ```
3. Login to Wandb.
    
    Choose one of the following methods:
    - Via command line.
        ```bash
        wandb login
        ```
    - Using .env file.
        ```bash
        echo "WANDB_API_KEY=your_wandb_api_key" > .env
        ```
    - Using environment variable.
        ```bash
        export WANDB_API_KEY=your_wandb_api_key
        ```
    If you prefer not to use Wandb (not recommended), you can disable it:
    - Using .env file.
        ```bash
        echo "WANDB_MODE=disabled" > .env
        ```
    - Using environment variable.
        ```bash
        export WANDB_MODE=disabled
        ```

4. Run the following .ipynb files in the order:
    - Data preprocessing: [preprocessing.ipynb](preprocessing.ipynb)
    - Split dataset: [split_dataset.ipynb](split_dataset.ipynb)
    - Train and evaluate the model:
        - For Centralized Learning: [train_centralized.ipynb](train_centralized.ipynb)
        - For Federated Learning: [train_federated.ipynb](train_federated.ipynb)
