import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
import os
from scipy.stats import spearmanr
import pickle

from AdjacencyMat import compute_KNN_graph

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

local_measure_dict={
    # 'betweenness': pd.read_csv('results/betweenness.csv'),
    'strength': pd.read_csv('results/strength.csv'),
    # 'strengths_nodal_positive':pd.read_csv('results/strengths_nodal_positive.csv'),
    # 'strengths_nodal_negative':pd.read_csv('results/strengths_nodal_negative.csv'),
    # 'clustering_coefficient_positive':pd.read_csv('results/clustering_coefficient_positive.csv'),
    # 'clustering_coefficient_negative':pd.read_csv('results/clustering_coefficient_negative.csv'),
    # 'local_assortativity_positive':pd.read_csv('results/local_assortativity_positive.csv'),
    # 'clustering_coefficient':pd.read_csv('results/clustering_coefficient.csv'),
    # 'local_efficiency':pd.read_csv('results/local_efficiency.csv'),
}

outlier_subjects={}

spearman_corr_df=pd.DataFrame(columns=['r', 'p_value'])

# atlas_names=[]
# f = open('data/atlas/schaefer_2018/Schaefer2018_100Parcels_7Networks_order.txt')
# lines = f.readlines()
# for line in lines:
#     columns = line.strip().split('\t')
#     atlas_names.append(columns[1])

UCLA_CNP_df=pd.read_csv('data/UCLA_CNP/UCLA_CNP_100FC_phenotype_cognition.csv')[['participant_id', 'diagnosis', 'Cohert', 'PC1']]

def local_sum(local_measure_dict, UCLA_CNP_df):
    global spearman_corr_df

    # UCLA_CNP df와 모든 measure값 합치기
    for measure_name, measure_df in local_measure_dict.items():
        UCLA_CNP_df = pd.concat([UCLA_CNP_df, measure_df.iloc[:,1:]], axis=1)

    # FC 이상한 subjects 제거
    drop_subs = ['sub-50029', 'sub-10271', 'sub-10998', 'sub-50006', 'sub-70020', 'sub-50008', 'sub-60078']
    UCLA_CNP_df = UCLA_CNP_df[~UCLA_CNP_df['participant_id'].isin(drop_subs)]

    for measure_idx, measure_name in enumerate(local_measure_dict.keys()):
        print(f"measure :{measure_name} ")

        for measure in UCLA_CNP_df.columns.to_list()[(measure_idx*100)+4:(measure_idx+1)*100+4]:
            corr, p_value = spearmanr(UCLA_CNP_df['PC1'], UCLA_CNP_df[measure])
            measure_df = pd.DataFrame({ 'r': [corr], 'p_value': [p_value],}, index=[measure])
            measure_df_cleaned = measure_df.dropna(axis=1, how='all')
            spearman_corr_df = pd.concat([spearman_corr_df, measure_df_cleaned])

            plt.figure(figsize=(5, 5))
            #scatter = plt.scatter(UCLA_CNP_df[measure], UCLA_CNP_df['PC1'])
            scatter = sns.lmplot(x=measure, y='PC1', data=UCLA_CNP_df, scatter_kws={'color':'black', 's':10}, line_kws={'color':'red'})

            Q1 = UCLA_CNP_df[measure].quantile(0.25)
            Q3 = UCLA_CNP_df[measure].quantile(0.75)
            IQR = Q3 - Q1
            outliers = (UCLA_CNP_df[measure] < Q1 - 3 * IQR) | (UCLA_CNP_df[measure] > Q3 + 3 * IQR)  
            
            # measure outlier subjects와 100ROI에서 outlier되는 횟수, 한 measure에 대해서만 측정해야 함.
            for sub in UCLA_CNP_df[outliers]['participant_id']:
                if sub in outlier_subjects:
                    outlier_subjects[sub] += 1
                else:
                    outlier_subjects[sub] = 1    
            # for idx, row in UCLA_CNP_df[outliers].iterrows():
            #         x_random = np.random.uniform(-0.02, 0.02)
            #         y_random = np.random.uniform(-0.02, 0.02)
            #         plt.scatter(row[measure], row['PC1'], color='black', s=50)
            #         plt.text(row[measure]+x_random, row['PC1']+y_random, f'({row[measure]:.2f}, {row["PC1"]:.2f})', color='black', ha='right') 

            plt.xlabel(f'{measure}')
            plt.ylabel('Cognition Score')

            dir = f'results/drop_subs/local/{measure_name}'
            createDirectory(dir)
            plt.savefig(Path(dir)/f'{measure}.png')
            #plt.show()
            plt.close()

def global_sum(global_measure_dict, UCLA_CNP_df):
    global spearman_corr_df

    for measure_name, measure_list in global_measure_dict.items():
        UCLA_CNP_df = pd.concat([UCLA_CNP_df, pd.DataFrame(measure_list, columns=[measure_name])], axis=1)
    
    UCLA_CNP_df.drop(UCLA_CNP_df[UCLA_CNP_df['transitivity']>0.6].index, axis=0, inplace=True)
    
    for measure_idx, measure_name in enumerate(global_measure_dict.keys()):
        print(f"measure :{measure_name} ")

        corr, p_value = spearmanr(UCLA_CNP_df['PC1'], UCLA_CNP_df[measure_name])
        measure_df = pd.DataFrame({ 'r': [corr],'p_value': [p_value],}, index=[measure_name])
        measure_df_cleaned = measure_df.dropna(axis=1, how='all')
        spearman_corr_df = pd.concat([spearman_corr_df, measure_df_cleaned])

        plt.figure(figsize=(5, 5))
        #scatter = plt.scatter(UCLA_CNP_df[measure_name], UCLA_CNP_df['PC1'])
        scatter = sns.lmplot(x=measure_name, y='PC1', data=UCLA_CNP_df, scatter_kws={'color':'black', 's':10}, line_kws={'color':'red'})
        
        # Q1 = UCLA_CNP_df[measure_name].quantile(0.25)
        # Q3 = UCLA_CNP_df[measure_name].quantile(0.75)
        # IQR = Q3 - Q1
        # outliers = (UCLA_CNP_df[measure_name] < Q1 - 3 * IQR) | (UCLA_CNP_df[measure_name] > Q3 + 3 * IQR)
        # #print(f"outliter subject(global) : {UCLA_CNP_df[outliers]['participant_id']}")
        # for idx, row in UCLA_CNP_df[outliers].iterrows():
        #         x_random = np.random.uniform(-0.02, 0.02)
        #         y_random = np.random.uniform(-0.02, 0.02)
        #         plt.scatter(row[measure_name], row['PC1'], color='red', s=50)
        #         plt.text(row[measure_name]+x_random, row['PC1']+y_random, f'({row[measure_name]:.2f}, {row["PC1"]:.2f})', color='black', ha='right') 

        plt.xlabel(f'{measure_name}')
        plt.ylabel('Cognition Score')

        dir = f'results/drop_subs/global'
        createDirectory(dir)
        plt.savefig(Path(dir)/f'{measure_name}.png')
        #plt.show()
        
        plt.close()

if __name__=='__main__':
    is_local=True
    is_global_=False

    if is_global_:
        with open('results/globals.pickle', 'rb') as handle:
            global_measure_dict = pickle.load(handle)
        global_sum(global_measure_dict, UCLA_CNP_df)
    if is_local:local_sum(local_measure_dict, UCLA_CNP_df)
    print(outlier_subjects)
    
    #spearman_corr_df.to_csv("results/non_drop_subs/corr.csv")