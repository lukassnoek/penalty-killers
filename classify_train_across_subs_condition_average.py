import os.path as op
from sklearn.externals import joblib as jl
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif, SelectPercentile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from skbold.postproc import MvpResults
import numpy as np


pipe = Pipeline([
    ('ufs', SelectPercentile(score_func=f_classif, percentile=10)),
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='linear', C=1.0))
])

mvp = jl.load('MVP/mvp_across_subjects.jl')
unique_dirs = np.unique(mvp.directories)

new_X = np.zeros((len(unique_dirs), mvp.X.shape[1]))
new_y = np.zeros(len(unique_dirs))
for i, fdir in enumerate(unique_dirs):

    idx = np.array([fdir == this_dir for this_dir in mvp.directories])
    new_X[i, :] = mvp.X[idx, :].mean(axis=0)
    new_y[i] = np.unique(mvp.y[idx])

mvp.y = new_y
mvp.X = new_X

mvp_results = MvpResults(mvp=mvp, type_model='classification', n_iter=10,
                         feature_scoring='forward', verbose=True,
                         accuracy=accuracy_score,
                         f1_score=f1_score)

out_dir = op.join('RESULTS', 'TRAIN', 'ACROSS_SUBS', 'CONDITION_AVERAGE')

skf = StratifiedKFold(n_splits=10)
for train_idx, test_idx in skf.split(X=mvp.X, y=mvp.y):

    X_train, y_train = mvp.X[train_idx], mvp.y[train_idx]
    X_test, y_test = mvp.X[test_idx], mvp.y[test_idx]
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mvp_results.update(pipeline=pipe, test_idx=test_idx, y_pred=pred)

mvp_results.compute_scores(maps_to_tstat=False)
mvp_results.write(out_path=out_dir)
