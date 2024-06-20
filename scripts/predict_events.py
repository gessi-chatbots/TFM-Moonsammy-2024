import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
import os
import argparse

parser = argparse.ArgumentParser(description='Predict events using machine learning models.')
parser.add_argument('-m', type=str, required=True, help='Directory containing metrics files')
parser.add_argument('-t', type=str, required=True, help='Truth set file')
parser.add_argument('-i', type=str, required=True, help='Directory containing inference files')
args = parser.parse_args()

base_dir_metrics = args.m
base_dir_truth = args.t
base_dir_inference = args.i 

files = [
    "negative_review_count.csv",
    "negative_review_percentage.csv",
    "negative_sentiment_count.csv",
    "negative_sentiment_percentage.csv",
    "neutral_sentiment_count.csv",
    "neutral_sentiment_percentage.csv",
    "positive_review_count.csv",
    "positive_review_percentage.csv",
    "positive_sentiment_count.csv",
    "positive_sentiment_percentage.csv",
    "review_count.csv",
    "review_polarity.csv",
    "review_rating.csv",
    "review_word_count.csv"
]
feature_files = [os.path.join(base_dir_metrics, file) for file in files]

features = pd.DataFrame()
for file in feature_files:
    df = pd.read_csv(file)
    feature_name = file.split("/")[-1].split(".")[0]
    df.columns = ["App"] + list(df.columns[1:]) 
    df = df.melt(id_vars=["App"], var_name="Time window", value_name=feature_name)
    if features.empty:
        features = df
    else:
        features = pd.merge(features, df, on=["App", "Time window"], how="outer")

features["App"] = features["App"].astype(str)
features["Time window"] = features["Time window"].astype(str)

#unique identifier for merging
features["unique_id"] = features["App"] + "_" + features["Time window"]

truth_set = pd.read_csv(base_dir_truth)
truth_set["unique_id"] = truth_set["App"].astype(str) + "_" + truth_set["Time window"].astype(str)

#Merge
merged_data = pd.merge(features, truth_set, on="unique_id")

X = merged_data.drop(columns=["App_x", "unique_id", "Event", "Time window_x", "App_y", "Time window_y"])
y = merged_data["Event"].map({"yes": 1, "no": 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

event = np.sum(y == 1)
non_event = np.sum(y == 0)
print("Received from the truth set:")
print(f"Number of predicted events: {event}")
print(f"Number of predicted non-events: {non_event}")
print("\nPredicting events. This may take a while...")

pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])

param_grid = [
    {
        'classifier': [SVC(kernel='rbf')],
        'preprocessing': [StandardScaler(), MinMaxScaler(), None],
        'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100, 'auto', 'scale'],
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
    },
    {
        'classifier': [RandomForestClassifier()],
        'preprocessing': [None],
        'classifier__n_estimators': [50, 100, 150, 200, 250],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    {
        'classifier': [GaussianNB()],
        'preprocessing': [None]
    },
    {
        'classifier': [MLPClassifier(max_iter=3000, random_state=42)],
        'preprocessing': [StandardScaler(), None],
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100,100)],
        'classifier__alpha': [0.0001, 0.001, 0.01, 0.1],
        'classifier__learning_rate_init': [0.001, 0.01, 0.1],
        'classifier__activation': ['tanh', 'relu']
    }
]

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)

print("Best classifier:\n{}\n".format(grid.best_params_))
print("Best cross-validation F1 score: {:.2f}".format(grid.best_score_))

y_pred = grid.predict(X_test)
test_score = f1_score(y_test, y_pred)
print("Test-set F1 score: {:.2f}".format(test_score))
print("\nTest set Performance report:")
print(classification_report(y_test, y_pred))

print()
events = np.sum(y_pred == 1)
non_events = np.sum(y_pred == 0)
print("For test set:")
print(f"Number of predicted events: {events}")
print(f"Number of predicted non-events: {non_events}")

indices = [index for index, value in enumerate(y_pred) if value == 1]

apps = merged_data.loc[X_test.index[indices], 'App_x']
time_windows = merged_data.loc[X_test.index[indices], 'Time window_x']

predicted_events = pd.DataFrame({'App': apps, 'Time window': time_windows})
print("\nInstances of Predicted events:")
print(predicted_events)
print()

#inference
feature_files = [os.path.join(base_dir_inference, file) for file in files]

features = pd.DataFrame()
for file in feature_files:
    df = pd.read_csv(file)
    feature_name = file.split("/")[-1].split(".")[0]
    df.columns = ["App"] + list(df.columns[1:]) 
    df = df.melt(id_vars=["App"], var_name="Time window", value_name=feature_name)
    if features.empty:
        features = df
    else:
        features = pd.merge(features, df, on=["App", "Time window"], how="outer")
        
X_new = features.drop(columns=["App", "Time window"])

y_pred = grid.predict(X_new)

events = np.sum(y_pred == 1)
non_events = np.sum(y_pred == 0)
print("Now predicting the next weeks:")
print(f"Number of predicted events: {events}")
print(f"Number of predicted non-events: {non_events}")

indices = [index for index, value in enumerate(y_pred) if value == 1]

apps = features.loc[X_new.index[indices], 'App']
time_windows = features.loc[X_new.index[indices], 'Time window']

predicted_events = pd.DataFrame({'App': apps, 'Time window': time_windows})
print("\nInstances of Predicted events:")
print(predicted_events)