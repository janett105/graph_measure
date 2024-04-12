import brainconn
import pandas as pd

class Network(object):
    def __init__(self, corr, adj):
        self.corr = corr
        self.adj = adj

        self.betweenness=None
        self.strength=None
        self.strengths_nodal_positive=None
        self.strengths_nodal_negative=None
        self.clustering_coefficient=None
        self.clustering_coefficient_positive=None
        self.clustering_coefficient_negative=None
        self.local_assortativity_positive=None
        self.local_efficiency=None
        
        self.transitivity=None
        self.assortativity=None
        self.strengths_total_positive=None
        self.strengths_total_negative=None
        self.global_efficiency=None

    def compute_local_graph_measures(self): #(100,)
        # self.betweenness = brainconn.centrality.betweenness_wei(self.corr)
        # self.strength = brainconn.degree.strengths_und(self.corr)
        self.strengths_nodal_positive, self.strengths_nodal_negative,_,_ = brainconn.degree.strengths_und_sign(self.corr)
        # self.clustering_coefficient = brainconn.clustering.clustering_coef_wu(self.corr)
        #self.local_assortativity_positive, self.local_assortativity_negative = brainconn.core.local_assortativity_wu_sign(self.corr) #invalid value encountered in scalar divide, NAN
        #self.clustering_coefficient_positive, self.clustering_coefficient_negative = brainconn.clustering.clustering_coef_wu_sign(self.corr) 
        #self.local_efficiency = brainconn.distance.efficiency_wei(self.adj, local=True)

    def compute_global_graph_measures(self): #(1,)
        self.transitivity = brainconn.clustering.transitivity_wu(self.corr)
        self.assortativity = brainconn.core.assortativity_wei(self.corr)
        _,_,self.strengths_total_positive, self.strengths_total_negative = brainconn.degree.strengths_und_sign(self.corr)
        self.global_efficiency = brainconn.distance.efficiency_wei(self.adj, local=False)
        self.density, vertex_n, edge_n = brainconn.physical_connectivity.density_und(self.corr)
        
    def apped_global_measures_list(self,global_measure_dict):
        global_measure_dict['transitivity'].append(self.transitivity)
        global_measure_dict['assortativity'].append(self.assortativity)
        global_measure_dict['strengths_total_positive'].append(self.strengths_total_positive)
        global_measure_dict['strengths_total_negative'].append(self.strengths_total_negative)
        global_measure_dict['global_efficiency'].append(self.global_efficiency)
    
    def apped_local_measures_df(self, local_measure_dict):
        # local_measure_dict['betweenness'] = pd.concat([local_measure_dict['betweenness'], pd.DataFrame(self.betweenness.reshape(-1,100))], ignore_index=True)
        # local_measure_dict['strength'] = pd.concat([local_measure_dict['strength'], pd.DataFrame(self.strength.reshape(-1,100))], ignore_index=True)
        # local_measure_dict['strengths_nodal_positive'] = pd.concat([local_measure_dict['strengths_nodal_positive'], pd.DataFrame(self.strengths_nodal_positive.reshape(-1,100))], ignore_index=True)
        local_measure_dict['strengths_nodal_negative'] = pd.concat([local_measure_dict['strengths_nodal_negative'], pd.DataFrame(self.strengths_nodal_negative.reshape(-1,100))], ignore_index=True)
        # local_measure_dict['clustering_coefficient'] = pd.concat([local_measure_dict['clustering_coefficient'], pd.DataFrame(self.clustering_coefficient.reshape(-1,100))], ignore_index=True)
        # local_measure_dict['local_assortativity_positive'] = pd.concat([local_measure_dict['local_assortativity_positive'], pd.DataFrame(self.local_assortativity_positive.reshape(-1,100))], ignore_index=True)
        # local_measure_dict['clustering_coefficient_positive'] = pd.concat([local_measure_dict['clustering_coefficient_positive'], pd.DataFrame(self.clustering_coefficient_positive.reshape(-1,100))], ignore_index=True)
        # local_measure_dict['clustering_coefficient_negative'] = pd.concat([local_measure_dict['clustering_coefficient_negative'], pd.DataFrame(self.clustering_coefficient_negative.reshape(-1,100))], ignore_index=True)
        #local_measure_dict['local_efficiency'] = pd.concat([local_measure_dict['local_efficiency'], pd.DataFrame(self.local_efficiency.reshape(-1,100))], ignore_index=True)