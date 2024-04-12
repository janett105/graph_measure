import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
import os
from scipy.stats import spearmanr
import json
import pickle


from network import Network
from AdjacencyMat import compute_KNN_graph

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

local_measure_dict={
    # 'betweenness': pd.DataFrame(),
    # 'strength': pd.DataFrame(),
    # 'strengths_nodal_positive': pd.DataFrame(),
    # 'clustering_coefficient_positive': pd.DataFrame(),
    # 'clustering_coefficient_negative': pd.DataFrame(),
    # 'local_assortativity_positive': pd.DataFrame(),
    # 'clustering_coefficient': pd.DataFrame(),
    'local_efficiency': pd.DataFrame(),
}

global_measure_dict={
    'transitivity': [],
    'assortativity': [],
    'strengths_total_positive': [],
    'strengths_total_negative': [],
    'global_efficiency': [],
}

# local_measure_dict={
#     'betweenness': pd.read_csv('results/local/betweenness.csv'),
#     'strength': pd.read_csv('results/local/strength.csv'),
#     'strengths_nodal_positive':pd.read_csv('results/local/strengths_nodal_positive.csv'),
#     'clustering_coefficient_positive':pd.read_csv('results/local/clustering_coefficient_positive.csv'),
#     'clustering_coefficient_negative':pd.read_csv('results/local/clustering_coefficient_negative.csv'),
#     'local_assortativity_positive':pd.read_csv('results/local/local_assortativity_positive.csv'),
#     'clustering_coefficient':pd.read_csv('results/local/clustering_coefficient.csv'),
#     'local_efficiency':pd.read_csv('results/local/local_efficiency.csv'),
# }

spearman_corr_df=pd.DataFrame(columns=['r', 'p_value', 'isCorr'])

UCLA_CNP_df=pd.read_csv('data/UCLA_CNP/UCLA_CNP_100FC_phenotype_cognition.csv')[['participant_id', 'diagnosis', 'Cohert', 'PC1']]

def local(network, local_measure_dict):   
    network.compute_local_graph_measures()
    network.apped_local_measures_df(local_measure_dict)


def local_sum(local_measure_dict, UCLA_CNP_df):
    global spearman_corr_df
    for measure_name, measure_df in local_measure_dict.items():
        UCLA_CNP_df = pd.concat([UCLA_CNP_df, measure_df], axis=1)
    
    # UCLA_CNP_df.drop(UCLA_CNP_df[UCLA_CNP_df['strengths_nodal_positive']>60].index, axis=0, inplace=True)
    # UCLA_CNP_df.drop(UCLA_CNP_df[UCLA_CNP_df['strength']>50].index, axis=0, inplace=True)
    # UCLA_CNP_df.drop(UCLA_CNP_df[UCLA_CNP_df['clustering_coefficient']>0.6].index, axis=0, inplace=True)
    # UCLA_CNP_df.drop(UCLA_CNP_df[UCLA_CNP_df['clustering_coefficient_positive']>0.6].index, axis=0, inplace=True)

    for measure_idx, measure_name in enumerate(local_measure_dict.keys()):
        print(f"measure :{measure_name} ")
        for measure in UCLA_CNP_df.columns.to_list()[(measure_idx*100)+4:(measure_idx+1)*100+4]:
            corr, p_value = spearmanr(UCLA_CNP_df['PC1'], UCLA_CNP_df[measure])
            measure_df = pd.DataFrame({ 'r': [corr],
                                        'p_value': [p_value],
                                        'isCorr': [p_value < 0.05]}, index=[measure])
            measure_df_cleaned = measure_df.dropna(axis=1, how='all')
            spearman_corr_df = pd.concat([spearman_corr_df, measure_df_cleaned])

            plt.figure(figsize=(5, 5))
            #scatter = plt.scatter(UCLA_CNP_df[measure], UCLA_CNP_df['PC1'])
            scatter = sns.lmplot(x=measure, y='PC1', data=UCLA_CNP_df,line_kws={'color':'yellow'})

            Q1 = UCLA_CNP_df[measure].quantile(0.25)
            Q3 = UCLA_CNP_df[measure].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (UCLA_CNP_df[measure] < Q1 - 3 * IQR) | (UCLA_CNP_df[measure] > Q3 + 3 * IQR)
            for idx, row in UCLA_CNP_df[outliers].iterrows():
                    x_random = np.random.uniform(-0.02, 0.02)
                    y_random = np.random.uniform(-0.02, 0.02)
                    plt.scatter(row[measure], row['PC1'], color='red', s=50)
                    plt.text(row[measure]+x_random, row['PC1']+y_random, f'({row[measure]:.2f}, {row["PC1"]:.2f})', color='black', ha='right') 

            plt.xlabel(f'{measure}')
            plt.ylabel('Cognition Score')

            dir = f'results/local/{measure_name}'
            createDirectory(dir)
            plt.savefig(Path(dir)/f'{measure}.png')
            #plt.show()
            
            plt.close()

def global_(network, global_measure_dict):
    network.compute_global_graph_measures()
    network.apped_global_measures_list(global_measure_dict)


def global_sum(global_measure_dict, UCLA_CNP_df):
    global spearman_corr_df

    for measure_name, measure_list in global_measure_dict.items():
        UCLA_CNP_df = pd.concat([UCLA_CNP_df, pd.DataFrame(measure_list, columns=[measure_name])], axis=1)
    
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
        #scatter = plt.scatter(UCLA_CNP_df[measure_name], UCLA_CNP_df['PC1'])
        scatter = sns.lmplot(x=measure_name, y='PC1', data=UCLA_CNP_df,line_kws={'color':'yellow'})
        
        Q1 = UCLA_CNP_df[measure_name].quantile(0.25)
        Q3 = UCLA_CNP_df[measure_name].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (UCLA_CNP_df[measure_name] < Q1 - 3 * IQR) | (UCLA_CNP_df[measure_name] > Q3 + 3 * IQR)
        for idx, row in UCLA_CNP_df[outliers].iterrows():
                x_random = np.random.uniform(-0.02, 0.02)
                y_random = np.random.uniform(-0.02, 0.02)
                plt.scatter(row[measure_name], row['PC1'], color='red', s=50)
                plt.text(row[measure_name]+x_random, row['PC1']+y_random, f'({row[measure_name]:.2f}, {row["PC1"]:.2f})', color='black', ha='right') 

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

        # abs_corr = np.abs(corr)
        # top_10_percentile_threshold = np.percentile(abs_corr, 90)
        # corr_th = np.where((corr > top_10_percentile_threshold) | (corr < -top_10_percentile_threshold), corr, 0)
        # adj_th = compute_KNN_graph(corr_th)
        
        network = Network(corr, adj)
        if is_local:local(network, local_measure_dict)
        if is_global_:global_(network, global_measure_dict)

    for key in local_measure_dict.keys():
        column_names = [f"{key}_{i+1}" for i in range(100)]
        local_measure_dict[key].columns=column_names
        local_measure_dict[key].to_csv(f'results/local/{key}.csv')

    with open('results/local/globals.pickle', 'wb') as handle:
        pickle.dump(global_measure_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # if is_global_:
    #     with open('results/local/globals.pickle', 'rb') as handle:
    #         loglobal_measure_dict = pickle.load(handle)
    #     global_sum(global_measure_dict, UCLA_CNP_df)
    # if is_local:local_sum(local_measure_dict, UCLA_CNP_df)
    
    # spearman_corr_df.to_csv("corr_new.csv")