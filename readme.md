# Tensorflow Classifier For Forecast Prediction

This project is an attempt to predict the future sales of a large IT company. The algorithm is trained on historic data of bussines oportunities, some of which where won and some weren't. The output file generates a list with all the opened bussines oportunities and the probability of them being won.

## Folder structure

1. ./scripts the scripts are the same code of the Jupyter Notebooks of the same name. These are for production purposes.

2. ./helper_fn and ./lstm.py are helper functions needed to treat data or to build the model. 

3. ./Preprocessing.ipynb is the notebook used to preprocess the data, eliminate unuseful variables based on statistical analysis. 