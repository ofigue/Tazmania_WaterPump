# Ref. https://github.com/abhishekkrthakur/mlframework
# to run it: % sh predict.sh

import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
from xgboost import XGBClassifier
from scipy import stats
import joblib
import numpy as np

from . import dispatcher

def predict(test_data_path, model_type, model_path):
    df = pd.read_csv(test_data_path)
    test_idx = df["id"].values
    lista = list()
    vote_df = pd.DataFrame() # ADDED
    vote_df['id'] = df["id"].values # ADDED

    for FOLD in range(5):
        df = pd.read_csv(test_data_path)
        df = df.drop(["id"], axis=1) # ADDED

        cols = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_columns.pkl"))      
        clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))
        
        df = df[cols]
        preds = clf.predict(df)

        vote_df = pd.concat([vote_df, pd.DataFrame(preds)], ignore_index=True, axis=1) # ADDED

    for i in range(vote_df.shape[0]): # ADDED
        lista.append(stats.mode(vote_df.loc[i, 1:5])[0][0]) # ADDED
    
    vote_df = pd.concat([vote_df, pd.DataFrame(lista)], ignore_index=True, axis=1) # ADDED
    preds = vote_df.loc[:, 6].values
    # print(vote_df.loc[0:50, :]) # ADDED

    sub = pd.DataFrame(np.column_stack((test_idx, preds)), columns=["id", "target"])
    return sub
    

if __name__ == "__main__":
    submission = predict(test_data_path="input/testSet.csv", 
                         model_type="randomforest", 
                         model_path="models/")
    submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)
    submission.to_csv(f"models/rf_submission.csv", index=False)