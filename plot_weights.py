import os
import os.path as op
import numpy as np
import nibabel as nib
from nilearn.plotting import plot_glass_brain
from skbold.core import convert2mni
import matplotlib.pyplot as plt
from glob import glob

base_dir = 'RESULTS/TRAIN/WITHIN_SUBS'

subs = sorted(glob(op.join(base_dir, '???')))

avg_data = np.zeros((len(subs), 91, 109, 91))
fig, axs = plt.subplots(6, 6, figsize=(50, 25), dpi=800)
for i, ax in enumerate(axs.reshape(-1)[:-1]):
    sub = subs[i]
    subname = op.basename(sub)
    print("Processing sub %s" % subname)
    reg_dir = op.join('FIRSTLEVEL', 'sub_id-sub-%s' % subname, 'task-test1',
                      'firstlevel_FEAT', 'test1.feat', 'reg')
    data = op.join(sub, 'thresh_weights.nii.gz')
    data = convert2mni(data, reg_dir=reg_dir, suffix='mni', overwrite=True)
    plot_glass_brain(data, threshold=0.0, axes=ax, title="Subject %s" % subname,
                     colorbar=False, plot_abs=False, display_mode='lyrz',
                     annotate=False)
    avg_data[i, :, :, :] = nib.load(data).get_data()

mean_data, std_data = avg_data.mean(axis=0), avg_data.std(axis=0)
t_data = mean_data / (std_data / np.sqrt(avg_data.shape[0] - 1))
t_data = nib.Nifti1Image(t_data, nib.load(data).affine)
nib.save(t_data, 'RESULTS/TRAIN/WITHIN_SUBS/tmap_average_weights.nii.gz')
plot_glass_brain(t_data, threshold=2.3, axes=axs[5, 5], title="AVERAGE",
                 colorbar=False, plot_abs=False, display_mode='lyrz',
                 annotate=False)

fig.savefig('weight_maps.pdf')
