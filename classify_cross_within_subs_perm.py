import sys
import os
import os.path as op
import numpy as np
from sklearn.externals import joblib as jl
from glob import glob
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from utils import load_test_behav
from joblib import Parallel, delayed
from skbold.utils import ArrayPermuter
from sklearn.feature_selection import (f_classif, SelectPercentile,
                                       VarianceThreshold)


def run_subject(sub, PERCENTILE, N_PERMS):

    pipe = Pipeline([
        ('varthresh', VarianceThreshold()),
        ('ufs', SelectPercentile(score_func=f_classif, percentile=PERCENTILE)),
        ('scaler', StandardScaler()),
        ('permuter', ArrayPermuter()),
        ('clf', SVC(kernel='linear'))
    ])

    train_file = op.join(sub, 'mvp_train_allvox.jl')
    test_file = op.join(sub, 'mvp_test_allvox.jl')

    if not op.isfile(train_file) or not op.isfile(test_file):
        print("Skipping sub %s" % op.basename(sub))
        return None

    perf = load_test_behav(sub)

    print("Processing sub-%s" % op.basename(sub))
    mvp_train = jl.load(train_file)
    mvp_test = jl.load(test_file)

    perm_scores = np.zeros(N_PERMS)
    for i in np.arange(N_PERMS):
        skf = StratifiedKFold(n_splits=10)
        props = []
        for train_idx, test_idx in skf.split(X=mvp_train.X, y=mvp_train.y):

            X_train, y_train = mvp_train.X[train_idx], mvp_train.y[train_idx]
            X_test, y_test = mvp_test.X, mvp_train.y
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            preds_given_correct_guess = pred[perf.astype(bool)]
            proportion_predicted_class_1 = preds_given_correct_guess.mean()
            props.append(proportion_predicted_class_1)

        perm_scores[i] = np.mean(props)

    out_dir = op.join('RESULTS', 'CROSS', 'WITHIN_SUBS', op.basename(sub))
    if not op.isdir(out_dir):
        os.makedirs(out_dir)

    np.save(op.join(out_dir, 'perms.npy'), perm_scores)


if __name__ == '__main__':

    PERCENTILE = int(sys.argv[1])
    N_PERMS = 1000
    subjects = sorted(glob(op.join('MVP', '???')))

    _ = Parallel(n_jobs=4)(delayed(run_subject)(sub, PERCENTILE, N_PERMS)
                            for sub in subjects)
