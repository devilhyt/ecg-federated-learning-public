#!/bin/bash
mkdir -p dataset
cd dataset
wget -O REFERENCE-v3.csv "https://physionet.org/files/challenge-2017/1.0.0/REFERENCE-v3.csv?download"
wget -O training2017.zip "https://physionet.org/files/challenge-2017/1.0.0/training2017.zip?download"
unzip training2017.zip