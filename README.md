# Bangla Fake News Detection Using Machine Learning and Inferancing with FastApi
This project aims to create a simple api that will use output from a machine learning model to predict if a news is real or fake. 

## Data Source
The data used in this project was from the paper BanFakeNews: A Dataset for Detecting Fake News in Bangla (Hossain et al., LREC 2020) published In Proceedings of the 12th Language Resources and Evaluation Conference, pages 2862–2871, Marseille, France. European Language Resources Association.

## Data Cleaning and Preprocessing
First downsample technique was used to balance the dataset. Then NLTK and BNLTK tookit was used to clean and preprocess the data. 

## Machine Learning Models
Different machine learning model was used and evaluted based on their accuracy matrics

| Model | % Accuracy | 
|---|---|
| Naive Bayes | 89.42% |
| SVM | 91.82% |
| XGBoost | 94.80% |
| AdaBoost | 85.96% |
| Decision Tree | 87.98% |
| Random Forest | 95.28% |
| SGD | 92.40% |
| KNN | 82.78% |
| Logistics Regression | 90.19% |
