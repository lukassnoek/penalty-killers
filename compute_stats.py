import numpy as np
import os.path as op
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import ttest_1samp, pearsonr

subs = sorted(glob(op.join('RESULTS', 'CROSS', 'WITHIN_SUBS', '???',
                           'perms.npy')))
seq_var = np.array([int(op.basename(op.dirname(f))[0]) for f in subs]) - 1
results = pd.read_csv('RESULTS/CROSS/WITHIN_SUBS/results.tsv', index_col=0, sep='\t')
results_all = results[results.score_type == 'all']
results_all = results_all.groupby(results_all.index).mean()
results_lr = results[results.score_type == 'lr']
results_lr = results_lr.groupby(results_lr.index).mean()

observed_scores_all = results_all.recall
observed_scores_lr = results_lr.recall

# Regular t-test
exog = np.ones(len(subs))
model = sm.OLS(endog=observed_scores_all - 0.5, exog=exog)
res = model.fit()
print("Using GLM, t-val = %.3f, pval = %.3f" % (res.tvalues, res.pvalues))

# Adding sequence to model
exog = np.hstack((np.ones((len(subs), 1)), seq_var[:, np.newaxis]))
model = sm.OLS(endog=observed_scores_all - 0.5, exog=exog)
res = model.fit()
print("Using GLM w/seq_var, t-val = %r, pval = %r" % (res.tvalues.tolist(),
                                                      res.pvalues.tolist()))

# Regular t-test
exog = np.ones(len(subs))
model = sm.OLS(endog=observed_scores_lr - 0.5, exog=exog)
res = model.fit()
print("Using GLM, t-val = %.3f, pval = %.3f" % (res.tvalues, res.pvalues))

# Adding sequence to model
exog = np.hstack((np.ones((len(subs), 1)), seq_var[:, np.newaxis], results_all.prop_correct[:, np.newaxis]))
model = sm.OLS(endog=observed_scores_lr - 0.5, exog=exog)
res = model.fit()
print("Using GLM w/seq_var, t-val = %r, pval = %r" % (res.tvalues.tolist(),
                                                      res.pvalues.tolist()))


# Now, using WLS
exog = np.ones(len(subs))
model = sm.WLS(endog=observed_scores_all - 0.5, exog=exog, weights=results_all.prop_correct)
res = model.fit()
print("Using WLS, t-val = %.3f, pval = %.3f" % (res.tvalues, res.pvalues))

exog = np.hstack((np.ones((len(subs), 1)), seq_var[:, np.newaxis]))
model = sm.WLS(endog=observed_scores_all - 0.5, exog=exog, weights=results_all.prop_correct)
res = model.fit()
print("Using WLS, t-val = %r, pval = %r" % (res.tvalues, res.pvalues))

# Now, using WLS
exog = np.ones(len(subs))
model = sm.WLS(endog=observed_scores_lr - 0.5, exog=exog, weights=results_lr.prop_correct)
res = model.fit()
print("Using WLS, t-val = %.3f, pval = %.3f" % (res.tvalues, res.pvalues))

exog = np.hstack((np.ones((len(subs), 1)), seq_var[:, np.newaxis]))
model = sm.WLS(endog=observed_scores_lr - 0.5, exog=exog, weights=results_lr.prop_correct)
res = model.fit()
print("Using WLS, t-val = %r, pval = %r" % (res.tvalues, res.pvalues))


quit()

all_perms = np.zeros((len(observed_scores), 1000))
perm_stats, perm_pvals = [], []

for i, sub in enumerate(subs):
    sub_name = op.basename(op.dirname(sub))
    perms = np.load(sub)
    perm_stat = (observed_scores[i] < perms).sum()
    print("Observed stat=%.3f, perm_stat=%i" % (observed_scores[i], perm_stat))
    all_perms[i, :] = perms
    perm_stats.append(perm_stat)
    perm_pvals.append(perm_stat / 1000)

f, ax_arr = plt.subplots(6, 6, figsize=(20, 20), sharex=False, sharey=False)
for i, ax in enumerate(ax_arr.flatten()[:-1]):

    ax.set_title('Subject: %s' % op.basename(op.dirname(subs[i])))
    ax.hist(all_perms[i, :], bins=20)
    ax.axvline(observed_scores[i], color='r', ls='--')
    ax.text(0.05, 27, 'P-value: %.3f' % perm_pvals[i])
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 175))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax_arr[5, 5].hist(observed_scores, bins=10, color='darkgreen')
ax_arr[5, 5].spines['top'].set_visible(False)
ax_arr[5, 5].spines['right'].set_visible(False)
ax_arr[5, 5].set_xlim((0, 1))
ax_arr[5, 5].set_ylim((0, 10))
ax_arr[5, 5].set_title("Observed scores across subjects")
ax_arr[5, 5].axvline(observed_scores.mean(), color='b', ls='--')

f.suptitle("Permutation distributions and observed scores", fontsize=20)
f.tight_layout()
f.subplots_adjust(top=0.93)

f.savefig('perm_dists.png')
