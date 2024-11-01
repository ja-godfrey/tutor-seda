#%%
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_ind_from_stats, mannwhitneyu
from sklearn.utils import resample

df_tutor = pd.read_json('./../data/derived/data_cleaned.json', dtype={"FIPS": str}, orient='records', lines=True)
df_seda = pd.read_csv('./../data/derived/seda_geodist_poolsub_gcs_5.0_updated_20240319.csv')
# %% 
# prepare data

# get the data for just districts that committed money to tutoring
filtered_seda = df_seda[(df_seda['subcat'] == 'all') & (df_seda['subgroup'] == 'all')]
merged_df = df_tutor.merge(filtered_seda, left_on='nces id', right_on='sedalea', how='inner')

# get just the data for districts that didn't commit to tutoring
sedalea_in_merged = merged_df['sedalea'].unique()
filtered_seda_noTutor = filtered_seda[~filtered_seda['sedalea'].isin(sedalea_in_merged)]

# %%
# clean data for statistical analysis. view means.

def clean_column(column):
    column = pd.to_numeric(column, errors='coerce')
    column = column.dropna()
    column = column[column != 0]  
    return column.astype(float)

group_1 = clean_column(merged_df['gcs_mn_coh_rla_ol'])
group_0 = clean_column(filtered_seda_noTutor['gcs_mn_coh_rla_ol'])

# Calculate mean and median for each group
mean_1 = group_1.mean()
mean_0 = group_0.mean()

median_1 = group_1.median()
median_0 = group_0.median()

# Display the results
print(f"Mean for 1s: {mean_1}")
print(f"Mean for 0s: {mean_0}")
print(f"Median for 1s: {median_1}")
print(f"Median for 0s: {median_0}")

#%%

# Independent Samples t-test (assuming equal variances)
t_stat, p_value_ttest = ttest_ind(group_1, group_0, equal_var=True)

# Welch's t-test (assuming unequal variances)
t_stat_welch, p_value_welch = ttest_ind(group_1, group_0, equal_var=False)

# Mann-Whitney U Test
u_stat, p_value_mannwhitney = mannwhitneyu(group_1, group_0, alternative='two-sided')

# Permutation Test
# Combine both groups
combined = np.concatenate([group_1, group_0])
observed_diff = np.mean(group_1) - np.mean(group_0)

# Perform permutation test
n_permutations = 10000
count = 0
for _ in range(n_permutations):
    np.random.shuffle(combined)
    new_group_1 = combined[:len(group_1)]
    new_group_0 = combined[len(group_1):]
    permuted_diff = np.mean(new_group_1) - np.mean(new_group_0)
    if abs(permuted_diff) >= abs(observed_diff):
        count += 1

p_value_permutation = count / n_permutations

# Bootstrap Confidence Intervals
bootstrap_diffs = []
n_bootstraps = 10000
for _ in range(n_bootstraps):
    boot_group_1 = resample(group_1, replace=True)
    boot_group_0 = resample(group_0, replace=True)
    bootstrap_diffs.append(np.mean(boot_group_1) - np.mean(boot_group_0))

# Calculate the 95% confidence interval
ci_lower = np.percentile(bootstrap_diffs, 2.5)
ci_upper = np.percentile(bootstrap_diffs, 97.5)

# Display Results
results = {
    "Independent t-test": (t_stat, p_value_ttest),
    "Welch's t-test": (t_stat_welch, p_value_welch),
    "Mann-Whitney U test": (u_stat, p_value_mannwhitney),
    "Permutation test p-value": p_value_permutation,
    "Bootstrap CI (95%)": (ci_lower, ci_upper)
}

for test, result in results.items():
    print(f"{test}: {result}")

# %%
