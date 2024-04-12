import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
from scipy.stats import spearmanr

from network import Network
from AdjacencyMat import compute_KNN_graph

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

local_measure_dict={
    'betweenness': pd.DataFrame(),
    #'eigenvector_centrality': pd.DataFrame(),
    'node_degree': pd.DataFrame(),
    'strength': pd.DataFrame(),
    'strengths_nodal_positive': pd.DataFrame(),
    #'strengths_nodal_negative': pd.DataFrame(),
    'clustering_coefficient_positive': pd.DataFrame(),
    'clustering_coefficient_negative': pd.DataFrame(),
    'local_assortativity_positive': pd.DataFrame(),
    'local_assortativity_negative': pd.DataFrame(),
    'clustering_coefficient': pd.DataFrame(),
    'local_efficiency': pd.DataFrame(),
}

global_measure_dict={
    'transitivity': [],
    'assortativity': [],
    'strengths_total_positive': [],
    'strengths_total_negative': [],
    'global_efficiency': [],
    'density':[],
}

spearman_corr_df=pd.DataFrame(columns=['r', 'p_value', 'isCorr'])

UCLA_CNP_df=pd.read_csv('data/UCLA_CNP/UCLA_CNP_100FC_phenotype_cognition.csv')[['participant_id', 'diagnosis', 'Cohert', 'PC1']]

def local(network):
    network.compute_local_graph_measures()
    network.apped_local_measures_df(local_measure_dict)

def local_sum(local_measure_dict, UCLA_CNP_df):
    global spearman_corr_df

    for key in local_measure_dict.keys():
        column_names = [f"{key}_{i+1}" for i in range(100)]
        local_measure_dict[key].columns=column_names

    for measure_name, measure_df in local_measure_dict.items():
        UCLA_CNP_df = pd.concat([UCLA_CNP_df, measure_df], axis=1)

    for measure_idx, measure_name in enumerate(local_measure_dict.keys()):
        print(f"measure :{measure_name} ")
        for measure in UCLA_CNP_df.columns.to_list()[(measure_idx*100)+4:(measure_idx+1)*100+3]:
            corr, p_value = spearmanr(UCLA_CNP_df['PC1'], UCLA_CNP_df[measure])
            measure_df = pd.DataFrame({ 'r': [corr],
                                        'p_value': [p_value],
                                        'isCorr': [p_value < 0.05]}, index=[measure])
            measure_df_cleaned = measure_df.dropna(axis=1, how='all')
            spearman_corr_df = pd.concat([spearman_corr_df, measure_df_cleaned])

            plt.figure(figsize=(5, 5))
            scatter = plt.scatter(UCLA_CNP_df[measure], UCLA_CNP_df['PC1'])
            plt.xlabel(f'{measure}')
            plt.ylabel('Cognition Score')

            measure_name = measure
            dir = f'results/local/{measure_name}'
            createDirectory(dir)
            plt.savefig(Path(dir)/f'{measure}.png')
            #plt.show()
            
            plt.close()

def global_(network):
    network.compute_global_graph_measures()
    network.apped_global_measures_list(global_measure_dict)

def global_sum(global_measure_dict, UCLA_CNP_df):
    global spearman_corr_df

    print(UCLA_CNP_df)
    for measure_name, measure_list in global_measure_dict.items():
        UCLA_CNP_df = pd.concat([UCLA_CNP_df, pd.DataFrame(measure_list, columns=[measure_name])], axis=1)
        print(UCLA_CNP_df)
    
    UCLA_CNP_df.drop(UCLA_CNP_df[UCLA_CNP_df['transitivity']>0.6].index, axis=0, inplace=True)
    
    for measure_idx, measure_name in enumerate(global_measure_dict.keys()):
        print(f"measure :{measure_name} ")

        corr, p_value = spearmanr(UCLA_CNP_df['PC1'], UCLA_CNP_df[measure_name])
        measure_df = pd.DataFrame({ 'r': [corr],
                                    'p_value': [p_value],
                                    'isCorr': [p_value < 0.05]}, index=[measure_name])
        measure_df_cleaned = measure_df.dropna(axis=1, how='all')
        spearman_corr_df = pd.concat([spearman_corr_df, measure_df_cleaned])

        plt.figure(figsize=(5, 5))
        scatter = plt.scatter(UCLA_CNP_df[measure_name], UCLA_CNP_df['PC1'])
        plt.xlabel(f'{measure_name}')
        plt.ylabel('Cognition Score')

        dir = f'results/global'
        createDirectory(dir)
        plt.savefig(Path(dir)/f'{measure_name}.png')
        #plt.show()
        
        plt.close()

if __name__=='__main__':
    is_local=True
    is_global_=False

    for subj_idx, subj in enumerate(UCLA_CNP_df['participant_id']):
        print(f'{subj_idx+1} / {len(UCLA_CNP_df["participant_id"])}')

        corr = np.load(f'data/UCLA_CNP/FC100_{subj}.npy') # correlation matrix should be zero diagonal
        adj = np.load(f'data/UCLA_CNP/adjacency/FC100_{subj}_adj.npy')

        abs_corr = np.abs(corr)
        top_10_percentile_threshold = np.percentile(abs_corr, 90)
        corr_th = np.where((corr > top_10_percentile_threshold) | (corr < -top_10_percentile_threshold), corr, 0)
        adj_th = compute_KNN_graph(corr_th)
        
        network = Network(corr, adj)
        if is_local:local(network)
        if is_global_:global_(network)
    if is_global_:global_sum(global_measure_dict, UCLA_CNP_df)
    if is_local:local_sum(local_measure_dict, UCLA_CNP_df)
    
    print(spearman_corr_df)