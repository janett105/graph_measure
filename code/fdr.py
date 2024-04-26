import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

def calculate_fdr(p_values):
    sorted_p_values = np.sort(p_values)
    N = len(p_values)
    ranks = np.arange(1, N+1)
    adjusted_p_values = np.minimum.accumulate(sorted_p_values * N / ranks)
    return adjusted_p_values

df = pd.read_csv('results/drop_subs/drop_corr.csv')

# 0~99(100), 100~199(200)..., 800~899 
for i in range(9):
    start = 100*i
    end = start+100
    print(df.iloc[start:end,0])

    measure_p = df.iloc[start:end, 2]
    measure_p_fdr = multipletests(measure_p, alpha=0.05, method='fdr_bh')
    
    df.iloc[start:end, 3] = measure_p_fdr[1]
    df.to_csv('results/drop_subs/drop_corr.csv', index=False)