# deep-learning-challenge

Machine learning techniques and neural networks were used to train and evaluate a model that helps Alphabet Soup selecting the applicants for funding with the best chance of success in their ventures.

## Overview of the Analysis

The purpose of the analysis is to identify whether the applicant's venture is successful with the use of the historical information about the applications.

Based on the provided data the model is able to predict "status of success". The data includes:
- EIN,
- NAME,
- APPLICATION_TYPE,
- AFFILIATION,
- CLASSIFICATION (classification code),
- USE_CASE,
- ORGANIZATION,
- STATUS (status of application),
- INCOME_AMT,
- SPECIAL_CONSIDERATIONS,
- ASK_AMT
- IS_SUCCESSFUL (target variable)

The dataset incuded 34,299 records in total. 

## Results

### Data Preprocessing

Data preprocessing steps included:
1. Splitting dataset into target and feature columns.
    1.1. Values in column IS_SUCCESSFUL were used as target variables
    1.2. Feature columns:
        - APPLICATION_TYPE,
        - AFFILIATION,
        - CLASSIFICATION (classification code),
        - USE_CASE,
        - ORGANIZATION,
        - STATUS (status of application),
        - INCOME_AMT,
        - SPECIAL_CONSIDERATIONS,
        - ASK_AMT

    For feature columns that included more than 10 unique values, the cut off points were identyfied to reduce the number of classes in a given features, further binning was conducted for the categories that had less datapoints than a set threshold and grouped into 'Other'.
    pd.get_dummies() method was used to encode columns with categorical values.

    1.3. Columns that are neither features nor targets were removed from the model:
        - EIN,
        - NAME.

### Compiling, Training, and Evaluating the Model

TensorFlow was used to design a neural network to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. 
The model includes 2 hidden layers with 8 and 5 neurons respectively, the activation functions that were used in the model are sigmoid. 

The original model prediction results - Loss: 0.56, Accuracy: 0.722

The following steps were taken to improve the prediction accuracy:
1. Changing # of "Application Types"
2. Encreasing # of epochs
3. Change the activation function for the hidden layers.

The optimized model prediction results - Loss: 0.56, Accuracy: 0.727

## Summary

The optimized model achieves slightly higher results, however, not being able to reach the level of 75%. 
The auto optimized should be applied to further tune the hyperparameters to achieve the higher results. 
Potentially, Supervised learning models as Logistic regresssion or SVM are also suitable for solving the classification problem.