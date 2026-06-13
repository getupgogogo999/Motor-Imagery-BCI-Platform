"""Quick benchmark: what's the accuracy ceiling on current CSV data?"""
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

SR = 250
FB = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
BASE = Path("patients (1)/patients")


def load_patient(pid: int) -> pd.DataFrame:
    return pd.read_csv(BASE / f"BCICIV_2a_{pid}.csv")


def fbcsp_features(df: pd.DataFrame, tmin: float = 0.0):
    feat = [c for c in df.columns if c.startswith("EEG")]
    mi = df[df.time >= tmin]
    rows, labels = [], []
    for (_, _), g in mi.groupby(["patient", "epoch"]):
        sig = g.sort_values("time")[feat].values.T
        vec = []
        for fmin, fmax in FB:
            b, a = butter(4, [fmin / (SR / 2), fmax / (SR / 2)], btype="band")
            f = filtfilt(b, a, sig, axis=1)
            vec.extend(np.log(np.var(f, axis=1) + 1e-8))
        rows.append(vec)
        labels.append(g["label"].iloc[0])
    return np.array(rows), np.array(labels)


print("=== Within-subject Filter Bank + SVM (80/20 holdout) ===")
for pid in [1, 2, 3, 5, 6, 7, 8, 9]:
    X, y = fbcsp_features(load_patient(pid))
    y_enc = LabelEncoder().fit_transform(y)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )
    pipe = Pipeline([("s", StandardScaler()), ("c", SVC(kernel="rbf"))])
    pipe.fit(Xtr, ytr)
    holdout = pipe.score(Xte, yte)
    cv = cross_val_score(
        Pipeline([("s", StandardScaler()), ("c", SVC(kernel="rbf"))]),
        X,
        y_enc,
        cv=5,
    )
    print(f" patient {pid}: holdout={holdout:.3f}, 5-fold={cv.mean():.3f} +/- {cv.std():.3f}")

print("\n=== Best patients: LDA 5-fold CV ===")
for pid in [7, 8, 9]:
    X, y = fbcsp_features(load_patient(pid))
    y_enc = LabelEncoder().fit_transform(y)
    cv = cross_val_score(
        Pipeline([("s", StandardScaler()), ("c", LinearDiscriminantAnalysis())]),
        X,
        y_enc,
        cv=5,
    )
    print(f" patient {pid}: LDA 5-fold={cv.mean():.3f}")

print("\n=== Time window sweep on patient 9 ===")
df = load_patient(9)
for tmin in [-0.1, 0.0, 0.2, 0.4]:
    X, y = fbcsp_features(df, tmin=tmin)
    y_enc = LabelEncoder().fit_transform(y)
    cv = cross_val_score(
        Pipeline([("s", StandardScaler()), ("c", SVC(kernel="rbf"))]),
        X,
        y_enc,
        cv=5,
    )
    print(f" tmin={tmin}: 5-fold={cv.mean():.3f}")

g = df.groupby(["patient", "epoch"])
sample = next(iter(g))[1].sort_values("time")
print(f"\nSamples per trial: {sample.shape[0]}, time range: {sample['time'].min()} ~ {sample['time'].max()}s")
