from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def my_train_test_split(df: pd.DataFrame, season_start_year: int, features: list[str]):
    season_start_date = pd.to_datetime(f'01/08/{season_start_year}', format=f'%d/%m/%y')
    season_end_date = pd.to_datetime(f'31/07/{season_start_year + 1}', format=f'%d/%m/%y')

    train_df = df[df['Date'] <= season_start_date]
    test_df = df[(season_start_date < df['Date']) & (df['Date'] < season_end_date)]

    return train_df[features], test_df[features], train_df['target'], test_df['target']

def league_predictions(df: pd.DataFrame, years: list[int], features: list[str], clf):
    acc_results = []
    for year in years:
        X_train, X_test, y_train, y_test = my_train_test_split(df, year, features)

        clf.fit(X_train, y_train)
        y_pred_lr = clf.predict(X_test)
    
        acc_results.append(np.round(accuracy_score(y_test, y_pred_lr), 3))

    data = {'Season': [f"{year}/{year+1}" for year in years], 'Accuracy': acc_results}
    return pd.DataFrame(data)

def plot_confustion_matrix(y_test, y_pred, clf, method: str):
    conf_matrix_lr = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_lr, display_labels=clf.classes_)
    disp.plot()
    plt.title(f'Confusion matrix - {method}')
    plt.xlabel('Prediction')
    plt.ylabel('Result')
    plt.show()

def process_data(df: pd.DataFrame):
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)