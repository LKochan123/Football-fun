# Football-fun

### Description:
Football-fun is a project where im doing machine learning prediction for matches in most common european football leagues. Model is predicting 3 results: home win, draw or away win. Model is universal so with good format of data it can be used for any league. Data in this application was used from [Football-Data.co.uk](https://www.football-data.co.uk/) website.

### Project structure:
- **data/**
  - **raw/** `Unprocessed data` - Original data files as obtained from the source.
  - **processed/** `Cleaned data` - Data that has been cleaned and transformed, ready for analysis.
    
- **notebooks/**
  - **merge_seasons/** `Seasons combination` - Notebook for combining different seasons' data.
  - **prediction/** `Model predictions` - Notebook used for making and evaluating predictions.
  - **statistic/** `Data exploration` - Notebook for statistical analysis of the data.

- **src/**
  - **core/** `Main functionality` - Core functions used across the project.
  - **train/** `Model training` - Scripts to train the machine learning models.
  - **statistic/** `Analysis utilities` - Functions for statistical data analysis.

- **tests/**
  - **test_data_validity/** `Data tests` - Scripts to validate the integrity and correctness of data.
 
### Results:
In this project I used data from 4 leagues: Premier League, La Liga, Serie A and Ligue 1. The training set was for seasons 2008/09 - 2019/20 and test set was for 2020/21 - 2022/23. I used several different algorithms and below I will show the best result for each of leagues.

| League         | Season    | Accuracy | Algorithm           |
|----------------|-----------|----------|---------------------|
| Premier League | 2022/2023   | 52.5%    | Gradient boosting |
| La Liga        | 2022/2023   | 55.5%    | Logistic reg.     |
| Serie A        | 2020/2021   | 53.7%    | Logistic reg.     |
| Ligue 1        | 2022/2023   | 50.1%    | SVM               |
