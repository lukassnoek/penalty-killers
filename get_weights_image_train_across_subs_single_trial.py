import os
import os.path as op
import nibabel as nib
import numpy as np
from glob import glob

base_dir = 'RESULTS/TRAIN/ACROSS_SUBS/SINGLE_TRIAL'

obs_img = nib.load(op.join(base_dir, 'MvpWithin.nii.gz'))
obs_data = obs_img.get_data()
obs_idx = obs_data.ravel() != 0.0

perm_dirs = glob(op.join(base_dir, 'perm_*'))

perm_data = np.zeros((len(perm_dirs), obs_idx.sum()))
for i, perm_dir in enumerate(perm_dirs):
    perm_img = op.join(perm_dir, 'MvpWithin.nii.gz')
    perm_data[i, :] = nib.load(perm_img).get_data().ravel()[obs_idx]

pvals = 1 - np.sum(perm_data > obs_data.ravel()[obs_idx], axis=0) / len(perm_dirs)

img2write = np.zeros(obs_data.shape).ravel()
img2write[obs_idx] = pvals
img2write = img2write.reshape(obs_data.shape)
nib.save(nib.Nifti1Image(img2write, obs_img.affine), 'test.nii.gz')
