import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

lung_cancer = pd.read_csv("lung cancer.csv")

lung_cancer.replace({"GENDER": {"M": 0, "F": 1}}, inplace=True)
lung_cancer.replace({"LUNG_CANCER": {"NO": 0, "YES": 1}}, inplace=True)

lung_cancer = lung_cancer[
    [
        "GENDER",
        "SMOKING",
        "YELLOW_FINGERS",
        "ANXIETY",
        "PEER_PRESSURE",
        "CHRONIC DISEASE",
        "FATIGUE ",
        "ALLERGY ",
        "WHEEZING",
        "ALCOHOL CONSUMING",
        "COUGHING",
        "SHORTNESS OF BREATH",
        "SWALLOWING DIFFICULTY",
        "CHEST PAIN",
        "LUNG_CANCER",
    ]
]

X = lung_cancer.drop(columns="LUNG_CANCER", axis=1)
Y = lung_cancer["LUNG_CANCER"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=3, stratify=Y
)

model = LogisticRegression()
model.fit(X_train, Y_train)

filename = "lung_cancer_model.pkl"
pickle.dump(model, open(filename, "wb"))
