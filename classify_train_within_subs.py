import os
import os.path as op
from sklearn.externals import joblib as jl
from glob import glob
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_classif, SelectPercentile
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from skbold.postproc import MvpResults, MvpAverageResults

subjects = sorted(glob(op.join('MVP', '???', 'mvp_train_nonzero.jl')))

pipe = Pipeline([
    ('ufs', SelectPercentile(score_func=f_classif, percentile=100)),
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='linear'))
])

out_dir = op.join('RESULTS', 'TRAIN', 'WITHIN_SUBS')

results = []
for sub in subjects:
    sub_name = op.basename(op.dirname(sub))
    print("Processing sub-%s" % sub_name)
    mvp = jl.load(sub)
    mvp_results = MvpResults(mvp=mvp, type_model='classification', n_iter=10,
                             feature_scoring='fwm', verbose=False,
                             accuracy=accuracy_score,
                             f1_score=f1_score)

    skf = StratifiedKFold(n_splits=10)
    for train_idx, test_idx in skf.split(X=mvp.X, y=mvp.y):

        X_train, y_train = mvp.X[train_idx], mvp.y[train_idx]
        X_test, y_test = mvp.X[test_idx], mvp.y[test_idx]
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mvp_results.update(pipeline=pipe, test_idx=test_idx, y_pred=pred)

    mvp_results.compute_scores(maps_to_tstat=False)
    tmp_out_dir = op.join(out_dir, sub_name)

    if not op.isdir(tmp_out_dir):
        os.makedirs(tmp_out_dir)

    mvp_results.write(out_path=tmp_out_dir)
    results.append(mvp_results)

mvp_av_results = MvpAverageResults(identifiers=subjects,
                                   mvp_results_list=results)
mvp_av_results.compute_statistics(metric='accuracy', h0=0.5)
mvp_av_results.write(path=out_dir, name='average_results')
