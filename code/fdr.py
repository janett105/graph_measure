import numpy as np
import pandas as pd

def calculate_fdr(p_values):
    sorted_p_values = np.sort(p_values)
    N = len(p_values)
    ranks = np.arange(1, N+1)
    adjusted_p_values = np.minimum.accumulate(sorted_p_values * N / ranks)
    return adjusted_p_values

df = pd.read_csv('corr_wow.csv')


measure_p = df.iloc[5:105, 2]

measure_p_fdr = pd.DataFrame(calculate_fdr(measure_p), columns=['p_value_fdr'])
print(measure_p_fdr)