import os
import os.path as op
from glob import glob
from skbold.core import MvpWithin

firstlevel_dir = 'FIRSTLEVEL'
sub_dirs = sorted(glob(op.join(firstlevel_dir, 'sub_id-sub*', 'task-train*',
                               'firstlevel_FEAT', '*.feat')))
sub_dirs = [sub_dir for sub_dir in sub_dirs if '208' not in sub_dir]
mvp = MvpWithin(source=sub_dirs, read_labels=True, ref_space='epi',
                statistic='zstat', remove_zeros=True)
mvp.create()
mvp.write(path='MVP', name='mvp_across_subjects')
