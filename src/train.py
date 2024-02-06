from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

league_names = ['Premier League', 'La Liga', 'Serie A', 'Ligue 1']

def clean_data(df: pd.DataFrame):
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

def my_train_test_split(df: pd.DataFrame, season_start_year: int, features: list[str]):
    season_start_date = pd.to_datetime(f'01/08/{season_start_year}', format=f'%d/%m/%y')
    season_end_date = pd.to_datetime(f'31/07/{season_start_year + 1}', format=f'%d/%m/%y')

    train_df = df[df['Date'] <= season_start_date]
    test_df = df[(season_start_date < df['Date']) & (df['Date'] < season_end_date)]

    X_train, X_test = standarize_data(train_df[features], test_df[features], features)

    return X_train, X_test, train_df['target'], test_df['target']

def standarize_data(X_train: pd.DataFrame, X_test: pd.DataFrame, features: list[str]): 
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns=features)
    X_test = pd.DataFrame(X_test_scaled, columns=features)

    return X_train, X_test

def league_predictions_core(df: pd.DataFrame, years: list[int], features: list[str], clf):
    acc_results = []
    for year in years:
        X_train, X_test, y_train, y_test = my_train_test_split(df, year, features)

        clf.fit(X_train, y_train)
        y_pred_lr = clf.predict(X_test)
    
        acc_results.append(np.round(accuracy_score(y_test, y_pred_lr), 3))

    data = {'Season': [f"{year}/{year+1}" for year in years], 'Accuracy': acc_results}
    return pd.DataFrame(data)

def plot_leagues_predictions(dfs: list[pd.DataFrame], years: list[int], features: list[str], clf):
    arr = []

    for league in dfs:
        arr.append(league_predictions_core(league, years, features, clf))

    return pd.concat(arr, axis=1, keys=league_names)

def plot_confustion_matrix(y_test, y_pred, clf, method: str):
    conf_matrix_lr = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_lr, display_labels=clf.classes_)
    disp.plot()
    plt.title(f'Confusion matrix - {method}')
    plt.xlabel('Prediction')
    plt.ylabel('Result')
    plt.show()