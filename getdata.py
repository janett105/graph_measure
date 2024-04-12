import numpy as np
import pandas as pd
import pickle
from network import Network

UCLA_CNP_df=pd.read_csv('data/UCLA_CNP/UCLA_CNP_100FC_phenotype_cognition.csv')[['participant_id', 'diagnosis', 'Cohert', 'PC1']]

local_measure_dict={
    # 'betweenness': pd.DataFrame(),
    # 'strength': pd.DataFrame(),
    # 'strengths_nodal_positive': pd.DataFrame(),
    'strengths_nodal_negative': pd.DataFrame(),
    # 'clustering_coefficient_positive': pd.DataFrame(),
    # 'clustering_coefficient_negative': pd.DataFrame(),
    # 'local_assortativity_positive': pd.DataFrame(),
    # 'clustering_coefficient': pd.DataFrame(),
    # 'local_efficiency': pd.DataFrame(),
}

global_measure_dict={
    'transitivity': [],
    'assortativity': [],
    'strengths_total_positive': [],
    'strengths_total_negative': [],
    'global_efficiency': [],
}

def local(network, local_measure_dict):   
    network.compute_local_graph_measures()
    network.apped_local_measures_df(local_measure_dict)
    
def global_(network, global_measure_dict):
    network.compute_global_graph_measures()
    network.apped_global_measures_list(global_measure_dict)

if __name__=='__main__':
    is_local=True
    is_global_=False   

    for subj_idx, subj in enumerate(UCLA_CNP_df['participant_id']):
        print(f'{subj_idx+1} / {len(UCLA_CNP_df["participant_id"])}')

        corr = np.load(f'data/UCLA_CNP/FC100_{subj}.npy') # correlation matrix should be zero diagonal
        adj = np.load(f'data/UCLA_CNP/adjacency/FC100_{subj}_adj.npy')

        # abs_corr = np.abs(corr)
        # top_10_percentile_threshold = np.percentile(abs_corr, 90)
        # corr_th = np.where((corr > top_10_percentile_threshold) | (corr < -top_10_percentile_threshold), corr, 0)
        # adj_th = compute_KNN_graph(corr_th)
        
        network = Network(corr, adj)
        if is_local:local(network, local_measure_dict)
        if is_global_:global_(network, global_measure_dict)

    if is_local:
        for key in local_measure_dict.keys():
            column_names = [f"{key}_{i+1}" for i in range(100)]
            local_measure_dict[key].columns=column_names
            local_measure_dict[key].to_csv(f'results/local/{key}.csv')
    if is_global_:
        with open('results/global/globals.pickle', 'wb') as handle:
            pickle.dump(global_measure_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)