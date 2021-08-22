import os
import os.path as op
from glob import glob
from skbold.core import MvpWithin

firstlevel_dir = 'FIRSTLEVEL'
sub_dirs = sorted(glob(op.join(firstlevel_dir, 'sub_id-sub*')))

for sub_dir in sub_dirs:

    sub_name = op.basename(sub_dir).split('-')[-1]
    print('Processing %s' % sub_name)
    test_data_dirs = sorted(glob(op.join(sub_dir, 'task-test?',
                                         'firstlevel_FEAT', '*.feat')))
    path_to_save = op.join('MVP', sub_name)
    if not op.isdir(path_to_save):
        os.makedirs(path_to_save)

    if test_data_dirs:
        print("Processing %r" % test_data_dirs)
        mvp = MvpWithin(source=test_data_dirs, read_labels=True,
                        ref_space='epi', statistic='zstat', remove_zeros=True)
        mvp.create()
        print(mvp.X.shape)
        mvp.write(path=path_to_save, name='mvp_test_nonzero')

    train_data_dirs = sorted(glob(op.join(sub_dir, 'task-train*',
                                          'firstlevel_FEAT', '*.feat')))

    if train_data_dirs:
        print("Processing %r" % train_data_dirs)
        mvp = MvpWithin(source=train_data_dirs, read_labels=True,
                        ref_space='epi', statistic='zstat', remove_zeros=True)
        try:
            mvp.create()
            print(mvp.X.shape)

            mvp.write(path=path_to_save, name='mvp_train_nonzero')
        except:
            print("Error: %s" % sub_name)
            pass
