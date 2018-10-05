from typing import List
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from ITCS6156MachineLearningCourse.josiah_laivins_assignment_2.BaseModel import Classifier


def show_correlation(columns_to_look_at: List[str], data: pd.DataFrame, n_rows=200, is_labeled=False):
    if not is_labeled:
        data = data.reset_index(drop=True)
        # Convert String columns into one-hot encoded columns
        for column in data:
            if data[column].dtype == object:
                # print(f'Encoding one-hot for column: {column} \r')
                data[column] = LabelEncoder().fit_transform(y=data[column].fillna('0'))
    else:
        warnings.warn("Assuming the dataset is already labeled. Note, if it has strings, then this will be blank.",
                      category=RuntimeWarning)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(data[columns_to_look_at].corr())
    plt.xticks(range(len(columns_to_look_at)), [column for column in data[columns_to_look_at].columns], rotation=45)
    plt.yticks(range(len(columns_to_look_at)), [column for column in data[columns_to_look_at].columns], rotation=45)
    plt.title(f'Prediction using: {n_rows} samples', y=1.15)
    plt.ylabel('Columns Y')
    plt.xlabel('Columns X')
    plt.margins(.1)
    plt.title(f'Prediction using: {n_rows}. Shows the correlation between different columns')
    plt.show()


def show_accuracy(x, t):
    # x, t = x[np.argsort(t, axis=0).flatten()], np.sort(t, axis=0)

    plt.plot(t, label="Ground Truth")
    plt.plot(x, label="Result")
    plt.legend()
    plt.show()
