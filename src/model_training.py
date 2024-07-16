from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd
import numpy as np
import os
from dvclive import Live
df = pd.read_csv('data/mobile_data.csv')
df=df[['ram','battery_power','px_width','px_height','price_range']]
X = df.drop(columns = 'price_range',axis=1)
y = df['price_range']

n_estimators=150
max_depth=50

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


with Live(save_dvc_exp=True) as live:

    live.log_metric('Accuracy',"Accuracy:", accuracy_score(y_test, y_pred))
    live.log_metric('Precision',"Precision (macro):", precision_score(y_test, y_pred, average='macro'))
    live.log_metric('Recall',"Recall (macro):", recall_score(y_test, y_pred, average='macro'))
    live.log_metric('F1-Score',"F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))

    live.log_param('n_estimator',n_estimators)
    live.log_param('max_depth',max_depth)


