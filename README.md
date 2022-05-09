# Benchmarking Deep Learning Architectures for predicting Readmission to the ICU and Describing patients-at-Risk

This repository is the attempt at reproducing the results of the paper: [Benchmarking Deep Learning Architectures for predicting Readmission to the ICU and Describing patients-at-Risk](https://www.nature.com/articles/s41598-020-58053-z). 

The official repo is located here: https://github.com/sebbarb/time_aware_attention

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

These results were obtained using MiniConda on a Windows machine and installing the requirements mentioned above.

## Dataset

The dataset used for this analysis is the publicly available [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) dataset. The instructions on how to obtain access are provided at the [end](https://physionet.org/content/mimiciii/1.4/#files) of the page. The dataset, once uncompressed, expands to ~50GB.

## Directory Structure

The code expects the following directory structure:

.
├── DLH-NeuralODEs                  # Repo Root Folder
|   |── data                        # Preprocessed files land here. Also copy embeddings here.
|   |── logdir                      # Trained models land here.
|   |── related_code                # Code for everything: pre-processing, training, 
|   |   |── embeddings              # Medical code embeddings
|   |── trained_models              # Pre-Trained Models
├── MIMIC-III Clinical Database     # Dataset Root Folder
|   ├── uncompressed                # Folder holding the Uncompressed version of the dataset's CSV files

## Pre-Processing

To pre-process the MIMIC-III dataset, run this command:

```pre-process
python preprocessing_reduce_charts.py
python preprocessing_reduce_outputs.py
python preprocessing_merge_charts_outputs.py
python preprocessing_ICU_PAT_ADMIT.py
python preprocessing_CHARTS_PRESCRIPTIONS.py
python preprocessing_DIAGNOSES_PROCEDURES.py
python preprocessing_create_arrays.py
```

## Training

To train the model(s) in the paper, run this command:

```train
python train.py
```

Select the model to train in `hyperparameters.py`

## Testing

To test the previously trained model, run:

```test
python test.py
```

This will produce the average precision and AUROC of the model.

## Pre-trained Models

The pre-trained models are available in the '<reporoot>\trained_models\' folder.

## Results

The following results were achieved on running the different models :


| Model name                       |  Avg. Precision   |      AUROC       |
| -------------------------------- |------------------ | ---------------- |
| ODE + RNN                        |     0.317         |      0.738       |
| ODE + RNN + Attention            |     0.309         |      0.737       |
| RNN (Exp Time Decay) + Attention |     0.302         |      0.733       |
| RNN (Exp Time Decay)             |     0.308         |      0.733       |
| ODE + Attention                  |     0.291         |      0.717       |
| Attention (concatenated time)    |     0.278         |      0.701       |
| Logistic Regression              |     0.254         |      0.659       |
