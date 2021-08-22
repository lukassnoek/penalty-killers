import os
import os.path as op
import pandas as pd
import numpy as np
from glob import glob

log_dir = 'BEHAV/logs'
logs = sorted(glob(op.join(log_dir, '*.txt')))

tsv_dir = 'BEHAV/clean_logs'
if not op.isdir(tsv_dir):
    os.makedirs(tsv_dir)

category = {'training_A': 'trainattention',
            'training_B': 'trainimagery',
            'test_A': 'test1',
            'test_B': 'test2'}

for log in logs:

    pulsefile = log.replace('.txt', '.log')
    if not op.isfile(pulsefile):
        pulsefile = list(pulsefile)
        pulsefile[14] = '-'
        pulsefile = ''.join(pulsefile)

    try:
        pulsedf = pd.read_csv(pulsefile, skiprows=2, sep='\t')
    except FileNotFoundError:
        print("Cannot find %s" % pulsefile)

    print("Processing %s" % op.basename(log))
    logname = op.basename(log)

    pulsedf = pulsedf[pulsedf['Event Type'] == 'Pulse']
    pulsetime = pulsedf.iloc[0]['Time'] / 10000

    if len(log.split('-')) == 2:
        sub_nr, cat = logname.split('-')
    elif len(log.split('_')) == 3:
        sub_nr, cat1, cat2 = logname.split('_')
        cat = "_".join([cat1, cat2])
    else:
        raise ValueError('Something wrong with log %s' % log)

    this_cat = category[cat.split('.')[0]]
    df = pd.read_csv(log, sep='\t', skiprows=2)

    df_concat = []
    for i in range(3):
        dfx = df[[col for col in df.columns if str(i+1) in col]]
        dfx.columns = [col.replace(str(i+1), '') for col in dfx.columns]
        dfx = dfx.assign(trial_type=[this_cat + '_block_%s' % ((2 - len(str(ii+1))) * '0' + str(ii+1)) for ii in np.arange(len(dfx))])
        df_concat.append(dfx)

    df_concat = pd.concat(df_concat, axis=0).sort_values(by='filmStart')
    df_concat = df_concat.rename(columns={'filmStart': 'onset', 'film': 'stim_file'})
    df_concat['duration'] = df_concat['filmStop'] - df_concat['onset']
    df_concat.drop(['filmStop', 'ITI', 'RT'], inplace=True, axis=1)
    df_concat = df_concat.assign(weight=np.ones(len(df_concat)))
    df_concat['duration'] /= 1000
    df_concat['onset'] = df_concat['onset'] / 1000 - pulsetime
    lr_lut = {0: 99, 1: 1, 2: 1, 3: 2, 4: 2}

    if 'test' in log:
        df_concat['result_lr'] = np.array([lr_lut[cat] == lr_lut[resp]
                                           for cat, resp in zip(df_concat['cat'],
                                                                df_concat['RB'])],
                                          dtype=int)
    fn = 'sub-%s_task-%s.tsv' % (sub_nr, this_cat)
    df_concat.to_csv(op.join(tsv_dir, fn), sep='\t')
