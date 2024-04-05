# emacs: -*- mode: python-mode; py-indent-offset: 4; tab-width: 4; indent-tabs-mode: nil -*-
# ex: set sts=4 ts=4 sw=4 et:
"""
.. _ex1:

=========================================
 Calculate centrality measures
=========================================

Centrality is a thing with stuff and things.

"""
# sphinx_gallery_thumbnail_number = 3

###############################################################################
# Start with the necessary imports
# --------------------------------
import os.path as op

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import brainconn

###############################################################################
# Get some data
# --------------------------------
corr = np.loadtxt(op.join(brainconn.utils.get_resource_path(), 'example_corr.txt'))

# Zero diagonal
adj_wei = corr - np.eye(corr.shape[0])
adj_bin = brainconn.utils.binarize(brainconn.utils.threshold_proportional(adj_wei, 0.2))

###############################################################################
# Look at weighted adjacency matrix
# -------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(adj_wei)
fig.show()

###############################################################################
# Look at binary adjacency matrix
# -------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(adj_bin)
fig.show()

###############################################################################
# Compute stuff
# -------------------------------------------
betw_wei = brainconn.centrality.betweenness_wei(adj_wei)
betw_bin = brainconn.centrality.betweenness_bin(adj_bin)
edg_betw_wei = brainconn.centrality.edge_betweenness_wei(adj_wei)[0]
idx = np.triu_indices(edg_betw_wei.shape[0], k=1)
edg_betw_wei = edg_betw_wei[idx]
edg_betw_wei = edg_betw_wei[edg_betw_wei > 0]
edg_betw_bin = brainconn.centrality.edge_betweenness_bin(adj_bin)[0]
idx = np.triu_indices(edg_betw_bin.shape[0], k=1)
edg_betw_bin = edg_betw_bin[idx]
edg_betw_bin = edg_betw_bin[edg_betw_bin > 0]

vals = [betw_wei, betw_bin, edg_betw_wei, edg_betw_bin]
names = ['Weighted Node Betweenness Centrality',
         'Binary Node Betweenness Centrality',
         'Weighted Edge Betweenness Centrality',
         'Binary Edge Betweenness Centrality']
fig, axes = plt.subplots(nrows=4, figsize=(12, 7))
for i in range(4):
    sns.distplot(vals[i], ax=axes[i])
    axes[i].set_title(names[i])
fig.tight_layout()
fig.show()
