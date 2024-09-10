# Geographical Analysis and Prediction of Chicago Crime

This project focuses on analyzing and predicting crime in Chicago based on geographical and temporal data. By leveraging machine learning techniques such as decision trees, random forests, and K-Nearest Neighbors (KNN), the project aims to enhance public safety by predicting crime occurrences based on time, location, and other factors.

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Datasets](#datasets)
4. [Data Processing](#data-processing)
5. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
6. [Machine Learning Models](#machine-learning-models)
    - [Decision Tree](#decision-tree)
    - [Random Forest](#random-forest)
    - [K-Nearest Neighbor](#k-nearest-neighbor)
7. [Sampling Techniques](#sampling-techniques)
8. [Dimensionality Reduction: PCA](#dimensionality-reduction-pca)
9. [Model Performance](#model-performance)
10. [Visualization](#visualization)
11. [Future Work](#future-work)
12. [References](#references)

## Introduction
Crime is a persistent social issue that affects public safety and socio-economic well-being. Inspired by real-world events, this project aims to analyze crime trends in Chicago based on time and location, and predict future crime occurrences using machine learning techniques. The goal is to assist law enforcement in making informed decisions and improving public safety.

## Project Structure
The project is divided into several components:
- **Data Preprocessing**: Handling missing values, filtering irrelevant features, and reclassifying crime types.
- **Exploratory Data Analysis (EDA)**: Visualizing crime patterns based on time, location, and type.
- **Modeling**: Implementing machine learning models (Decision Tree, Random Forest, KNN) for crime prediction.
- **Evaluation**: Using metrics like accuracy, F1-score, and ROC-AUC to evaluate the models.
- **Prediction**: Forecasting crime types for the year 2024 during major events in Chicago.

## Datasets
The project uses crime data from Chicago, spanning from 2001 to 2024. This dataset includes around 8 million incidents with variables like date, type, description, location, etc.

- **Source**: [Chicago Crime Data](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data)

## Data Processing
1. **Handling Missing Data**: Less than 1% missing values were removed.
2. **Feature Selection**: Relevant features like crime type, location (latitude and longitude), and time were retained for modeling.
3. **Crime Classification**: Crimes were reclassified into two categories (UCR Part I and Part II) using the IUCR codes.

## Exploratory Data Analysis (EDA)
Data visualizations were performed to understand crime trends. Key insights include:
- UCR Part I crimes (violent crimes) are more frequent than UCR Part II crimes (less serious crimes).
- Crime peaks during the afternoon and in summer months (July and August).
  
**Tools**: `ggplot2` in R, and [kepler.gl](http://www.kepler.gl/) for spatial visualizations.

## Machine Learning Models

### Decision Tree
A decision tree was trained on selected features (e.g., time, location). The model achieved:
- **Accuracy**: 64.57%
- **F1-Score**: 64.89%
  
### Random Forest
Random Forest outperformed the decision tree after hyperparameter tuning:
- **Accuracy**: 75.23%
- **F1-Score**: 67.31%
  
### K-Nearest Neighbor (KNN)
KNN was also explored, achieving:
- **Accuracy**: 73.69%
- **F1-Score**: 65.05%

## Sampling Techniques
To handle class imbalance, oversampling (SMOTE) and undersampling techniques were employed. However, these techniques did not lead to significant improvements in model performance.

## Dimensionality Reduction: PCA
Principal Component Analysis (PCA) was applied for dimensionality reduction. Models trained on PCA-reduced data did not outperform models without PCA.

## Model Performance
| Model          | Accuracy  | F1-Score |
|----------------|-----------|----------|
| Decision Tree  | 64.57%    | 64.89%   |
| Random Forest  | 75.23%    | 67.31%   |
| KNN            | 73.69%    | 65.05%   |

The Random Forest model with tuned hyperparameters provided the best performance in terms of both accuracy and F1-score.

## Visualization
Multiple visualizations were created to better understand the data:
1. Crime trends over time.
2. Crime distribution by type.
3. Geospatial maps of crime occurrences using latitude and longitude.

## Future Work
- Incorporating additional features such as economic indicators, demographic statistics, and weather data.
- Applying deep learning techniques to improve prediction accuracy.
- Expanding the dataset and including more complex features to address the limitations in current predictive models.

## References
1. Sharma, A., & Singh, D. (2021). Machine learning-based analytical approach for geographical analysis and prediction of Boston City crime using geospatial dataset. *Geojournal*. [DOI: 10.1007/s10708-021-10485-4](https://doi.org/10.1007/s10708-021-10485-4)
2. Hossain, S., et al. (2020). Crime Prediction Using Spatio-Temporal Data. *Communications in Computer and Information Science*. [DOI: 10.1007/978-981-15-6648-6_221](https://doi.org/10.1007/978-981-15-6648-6_221)
3. Yadav, S., et al. (2017). Crime pattern detection, analysis & prediction. *2017 International Conference of Electronics, Communication and Aerospace Technology (ICECA)*. [DOI: 10.1007/s10708-021-10485-4](https://doi.org/10.1007/s10708-021-10485-4)

