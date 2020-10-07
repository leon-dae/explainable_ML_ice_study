# Unfolding the complexity of ice with explainable machine learning models

This repository is the official implementation of [Unfolding the complexity of ice withexplainable machine learning models](https://arxiv.org/abs/2030.12345). 

## Requirements

This research was done in an Anaconda environment, packages were installed with pip and conda. An environment.yml file can be found in the repository. The main required packages are

| Name               | Version         | Build          | Channel        |
| ------------------ |---------------- | -------------- | -------------- | 
| xgboost   |     1.2.0-SNAPSHOT         |      pypi_0       |     pypi    
| tensorflow          |      2.2.0         |           pypi_0 |   pypi |
| pandas                |    1.0.3         |   py37h47e9c7a_0| | 

Visualization packages are matplotlib and seaborn.

In order for the scripts to run, data_preprocessing.py and auxiliary_functions.py are required. 

## Corresponding data

The data required to train the models is in data_points.xlsx, version 1.12. This file is not available in the repository, but you can send an e-mail to leon.kellner@tuhh.de or ehlers@tuhh.de. 

## Exploratory data analysis

The exploratory analysis is done with the following python codes: `data_preprocessing.py`, `exploratory_all_data.py` and `exploratory_strength_values.py`. 

## Training

To train the models in the paper, run the following notebooks: `behavior_XGBoost.ipynb`, `strength_XGBoost.ipynb`, `behavior_ANN.ipnyb`, `strength_ANN.ipynb`. 

## Comparison models

Different approaches were compared to the machine-learning models. Regarding the strength regression ML models, the comparison models were empirical formulas. Their computation can be found in `strength_empirical.py`. Regarding the behavior classification ML models, the analytical comparison model is in `behavior_analytical.py`. 

## Evaluation

The results were evaluated within the model training notebooks and in `model_performance.py` regarding taking the log of the target strength values or not.

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

Trained models are available in pickle format from the repository. They can be run with 

```
model_path_name_template = 'models/rgsr_xgb_sw_pickle.dat'
model = pickle.load(open(model_path_name_template, "rb"))  
predictions = model.predict(xgboost.DMatrix(X))
```

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
