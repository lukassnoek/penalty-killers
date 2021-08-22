import os
import os.path as op
import nibabel as nib
import numpy as np
from glob import glob

base_dir = 'RESULTS/TRAIN/WITHIN_SUBS'
subs = sorted(glob(op.join(base_dir, '???')))
CUTOFF = 0.001

for sub in subs:
    print(sub)
    obs_img = nib.load(op.join(sub, 'MvpWithin.nii.gz'))
    obs_data = obs_img.get_data()
    obs_idx = obs_data.ravel() != 0.0

    perm_dirs = glob(op.join(sub, 'perm_*'))
    perm_data = np.zeros((len(perm_dirs), obs_idx.sum()))
    for i, perm_dir in enumerate(perm_dirs):
        perm_img = op.join(perm_dir, 'MvpWithin.nii.gz')
        perm_data[i, :] = nib.load(perm_img).get_data().ravel()[obs_idx]

    n_times_selected = (perm_data != 0.0).sum(axis=0)

    perm_stats_pos = np.sum(perm_data > obs_data.ravel()[obs_idx], axis=0) / n_times_selected
    perm_stats_neg = np.sum(perm_data < obs_data.ravel()[obs_idx], axis=0) / n_times_selected

    pvals = np.zeros(perm_data.shape[1])
    pvals[perm_stats_pos < CUTOFF] = 1
    pvals[perm_stats_neg < CUTOFF] = 1

    img2write = np.zeros(obs_data.shape).ravel()
    img2write[obs_idx] = pvals
    img2write = img2write.reshape(obs_data.shape)
    fn = op.join(sub, 'pvals_perm.nii.gz')
    nib.save(nib.Nifti1Image(img2write, obs_img.affine), fn)

    tmp_data = obs_data.ravel()[obs_idx]
    tmp_data[np.invert(pvals.astype(bool))] = 0
    obs_data[obs_idx.reshape(obs_data.shape)] = tmp_data
    fn = op.join(sub, 'thresh_weights.nii.gz')
    nib.save(nib.Nifti1Image(obs_data, obs_img.affine), fn)
