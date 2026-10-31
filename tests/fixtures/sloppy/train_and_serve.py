"""Sloppy training and serving script used as a scanner fixture."""
import pickle

import pandas as pd
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from imblearn.over_sampling import SMOTE
from pydantic import BaseModel
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

password = "hunter2hunter2"


class Clipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.clip(0, 100)


def engineer(df):
    pass


df = pd.read_csv("data/transactions.csv")
df["future_amount"] = df["amount"].shift(-1)
df["merchant_rate"] = df.groupby("merchant")["is_fraud"].transform("mean")
y = df["is_fraud"]
X = SelectKBest(k=10).fit_transform(df.drop(columns=["is_fraud"]), y)
X_res, y_res = SMOTE().fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)
search = GridSearchCV(SVC(probability=True), {"C": [1, 10]}, scoring="accuracy")
search.fit(X_test, y_test)
print(search.best_score_)
print("ROC-AUC", roc_auc_score(y_test, search.predict_proba(X_test)[:, 1]))
t, p = stats.ttest_ind(df.amount[y == 1], df.amount[y == 0])
print("p-value", p)
pickle.dump(search.best_estimator_, open("model.pkl", "wb"))

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True)


class ExplainRequest(BaseModel):
    amount: float
    shap_values: list[float]


@app.post("/predict")
async def predict(req: ExplainRequest):
    model = pickle.load(open("model.pkl", "rb"))
    return {"p": model.predict_proba([[req.amount]])[0, 1]}


@app.post("/load")
def load(file: UploadFile):
    return {"loaded": str(pickle.load(file.file))}
