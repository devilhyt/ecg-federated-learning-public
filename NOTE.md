# Note

## Data Preprocessing
1. Denoising
    - Skip the first 2 seconds of the signal.
    - High-frequency noise (>40hz).
    - Baseline wander (<0.5hz).
2. Inversion correction
3. Downsampling
    - From 300hz to 100hz.

## Spilt Dataset
1. Split the dataset into `Train`, `Valid`, and `Test` sets.
    - Train : Valid : Test = 70 : 15 : 15
2. For federated learning
    - Further split the `Train` set for each client (called `Client_N` set).
        - The number of clients is 5.

## Tasks performed in Torch Dataset Class
1. Label Encoding

## Tasks performed in Lightning Data Module Class
1. Data Augmentation
    - Random Time Scale
    - Random Noise
    - ~~Random Invert (Vertical Flip)~~
    - ~~Random Mask~~
2. ~~Outlier Handling~~
    - ~~Clip the signal to the range of ±3 standard deviations.~~
3. Convert To Tensor
4. Cropping
    - Target time is 58sec
    - Target length is 5800 (Target time * resampled frequency)
    - If signal length > Target length
        - Train set: Random cropping.
        - Valid set: Head cropping.
        - Test set: Head cropping.
    - If signal length < Target length 
        - Zero padding.
    - If signal length = Target length
        - Do nothing.
5. Voltage Normalization
    - Min-Max Normalization
    - ~~Mean Normalization~~
    - ~~Z-Score Normalization~~
6. Unsqueeze
7. Imbalance Dataset Handling
    - Using `WeightedRandomSampler`.
8. Federated Dataset Partitioning
    - Non_IID: using `DirichletPartitioner`.
    - ~~IID: using `IidPartitioner`.~~