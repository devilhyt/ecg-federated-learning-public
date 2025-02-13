# Note

## Data Preprocessing
1. Denoising
    - High-frequency noise (>40hz).
    - Baseline wander (<0.5hz).
2. Inversion correction
    - If the signal is inverted, correct it.
3. Downsampling
    - From 300hz to 100hz.

## Spilt Dataset
1. Split the dataset into `Train`, `Valid`, and `Test` sets.
    - Train : Valid : Test = 80 : 10 : 10
2. For federated learning
    - Further split the `Train` set equally among each client (called `Client_N` set).
    - Each client then splits its portion into `Client_Train_N` and `Client_Valid_N` set.
        - Client_Train_N : Client_Valid_N = 90 : 10

## Tasks performed in Torch Dataset Class
1. Label Encoding

## Tasks performed in Lightning Data Module Class
1. Data Augmentation
    - Random Time Scale
    - Random Noise
    - Random Invert (Vertical Flip)
    - ~~Random Mask~~
2. ~~Outlier Handling~~
    - ~~Clip the signal to the range of ±3 standard deviations.~~
3. Data Length Standardization 
    - Target time is 60sec
    - Target length is 6000 (Target time * resampled frequency)
    - If signal length > Target length
        - Discard the redundant signal.
    - If signal length < Target length 
        - Zero padding.
    - If signal length = Target length
        - Do nothing.
4. Voltage Normalization
    - min-max normalization
    - ~~mean normalization~~
    - ~~z-score normalization~~
5. Convert To Tensor
6. Unsqueeze
7. Imbalance Dataset Handling
    - Using `WeightedRandomSampler`.
8. Federated Dataset Partitioning
    - Using one of methods in `flwr_datasets.partitioner`.