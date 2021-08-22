import os.path as op
import pandas as pd
import numpy as np


def load_test_behav(sub):

    sub = op.basename(sub)
    f1 = op.join('BEHAV', 'clean_logs', 'sub-%s_task-test1.tsv' % sub)
    f1 = pd.read_csv(f1, sep='\t')[['result', 'result_lr']].values

    try:
        f2 = op.join('BEHAV', 'clean_logs', 'sub-%s_task-test2.tsv' % sub)
        f2 = pd.read_csv(f2, sep='\t')[['result', 'result_lr']].values
        f1 = np.concatenate((f1, f2))
    except FileNotFoundError:
        pass

    return f1[:, 0], f1[:, 1]
