import sys
import os.path as op
from sklearn.externals import joblib as jl
from glob import glob
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif, SelectPercentile, VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from skbold.postproc import MvpResults, MvpAverageResults
from scipy.stats import ttest_1samp
from utils import load_test_behav
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

subjects = sorted(glob(op.join('MVP', '???')))

pipe = Pipeline([
    ('varthresh', VarianceThreshold()),
    ('ufs', SelectPercentile(score_func=f_classif, percentile=100)),
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='linear', C=1.0))
])

results = []

for sub in subjects:

    train_file = op.join(sub, 'mvp_train_allvox.jl')
    test_file = op.join(sub, 'mvp_test_allvox.jl')

    if not op.isfile(train_file) or not op.isfile(test_file):
        print("Skipping sub %s" % op.basename(sub))
        continue

    perf_all, perf_lr = load_test_behav(sub)

    print("Processing sub-%s" % op.basename(sub))
    mvp_train = jl.load(train_file)
    mvp_test = jl.load(test_file)

    skf = StratifiedKFold(n_splits=10)

    columns = ['accuracy', 'f1_score', 'recall', 'precision', 'prop_correct',
               'prop_imag', 'score_type']

    results_inner = []
    for train_idx, test_idx in skf.split(X=mvp_train.X, y=mvp_train.y):

        X_train, y_train = mvp_train.X[train_idx], mvp_train.y[train_idx]
        X_test, y_test = mvp_test.X, mvp_test.y
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        # Append regular scores
        results_inner.append(dict(
            accuracy=accuracy_score(perf_all, pred),
            f1_score=f1_score(perf_all, pred),
            recall=recall_score(perf_all, pred),
            precision=precision_score(perf_all, pred),
            prop_correct=perf_all.mean(),
            prop_imag=pred.mean(),
            score_type='all'
        ))

        # Append lr scores
        results_inner.append(dict(
            accuracy=accuracy_score(perf_lr, pred),
            f1_score=f1_score(perf_lr, pred),
            recall=recall_score(perf_lr, pred),
            precision=precision_score(perf_lr, pred),
            prop_correct=perf_lr.mean(),
            prop_imag=pred.mean(),
            score_type='lr'
        ))

    concat = pd.concat([pd.DataFrame(f, index=[op.basename(sub)])
                        for f in results_inner])
    results.append(concat)
concat_all = pd.concat(results)
fn = 'RESULTS/CROSS/WITHIN_SUBS/results.tsv'
concat_all.to_csv(fn, sep='\t', index=True)
