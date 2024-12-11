# ECG Federated Learning

## Overview
This is a federated learning project based on the PhysioNet/CinC Challenge 2017. The goal is to classify ECG signals into three categories: Normal, Atrial Fibrillation (AF), and Other.

## Environment
  - OS: Linux Mint 22
  - Python: 3.11.10

## How to run.
1. Download the dataset.

    ```bash
    bash download_dataset.sh
    ```
2. Install the requirements.

    ```bash
    pip install -r requirements.txt
    ```
3. Login to Wandb.
    
    Choose one of the following methods:
    - Login to wandb using the following command.
        ```bash
        wandb login
        ```
    - Using .env file.
        ```bash
        echo "WANDB_API_KEY=your_wandb_api_key" > .env
        ```
    - Using the environment variable.
        ```bash
        export WANDB_API_KEY=your_wandb_api_key
        ```
    If you don't want to use wandb (Not Recommended), you can disable it using the following methods:
    - Using .env file.
        ```bash
        echo "WANDB_MODE=disabled" > .env
        ```
    - Using the environment variable.
        ```bash
        export WANDB_MODE=disabled
        ```

4. Run the following .ipynb files in the order:
    - Data preprocessing: [preprocessing.ipynb](preprocessing.ipynb)
    - Split dataset: [split_dataset.ipynb](split_dataset.ipynb)
    - Train and evaluate the model:
        - For Centralized Learning: [train_centralized.ipynb](train_centralized.ipynb)
        - For Federated Learning: [train_federated.ipynb](train_federated.ipynb)
