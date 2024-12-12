# DiabetesPrediction
This is a project for BU EC503: Learning from Data 2024 Fall. We are team 7: A comparison of Machine Learning and Deep Learning Algorithms for Diabetes Prediction. Team members consists of Oliver Li, Bohan Zhang, Aowei Zhao and Xindong Zhou.

Three datasets are used in this project. diabetes.csv (https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset), diabetes_data.csv (https://www.kaggle.com/datasets/rabieelkharoua/diabetes-health-dataset-analysis) and diabetes_2000.csv. Correspondingly, pima.ipynb, data.ipynb and 2000.ipynb include the code of traning five different models on those datasets, comparing their CCR and plotting results. For three .ipynb files, all you need to do is installing required python libraries and clicking "run all" button to reproduce results.

lightGBMComplete.py: To use and train different features for the model, please change the dataset_info variable at line 14, and add the features to drop for the model training. Then run the program to show the graphs for CCR change in training and validation. 

wideAndDeepComplete.py: To use and train different features for the model, please refer to change categorical_features and numerical_features, for features to drop, please add to line 23. Then run the program to show the graphs for CCR change in training and validation. 

