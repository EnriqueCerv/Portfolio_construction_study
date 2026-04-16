# %% 
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import pairwise_distances
from covariance import returns_corr_cov
# %%

def hrp_portfolio(
        returns: pd.DataFrame, 
        lw: bool = True, 
        plot: bool = False
    ) -> pd.Series:
    '''
    Input: DataFrame of returns for stocks
    Output: Weights obtained via hierarchical risk parity from Marcos Lopez de Prado
    '''
    
    # Get covariance and correlation
    covariance_matrix, corr_matrix = returns_corr_cov(returns, lw=lw, plot=plot)

    # Get correlation distance matrix and euclidean distance matrix
    corr_distance_matrix = corr_distance(corr_matrix, plot)
    euc_distance_matrix = euclidean_distance(corr_distance_matrix, plot)

    # Based on distances between correlation columns, get hierarchy of clustering
    linkage = tree_clustering(euc_distance_matrix)

    # Quasi-diagonalise the covariance via the clustering hierarchy
    quasi_diagonal_order, quasi_diagonal_cov = quasi_diagonalize(covariance_matrix, linkage, plot)

    # Get optimized weight trhough weighted inverse variance allocation of each cluster
    optimized_weights = recursive_bisection(quasi_diagonal_order, quasi_diagonal_cov)

    return optimized_weights

# %%
# # # # # # # # # # # # # #
# TREE CLUSTERING
# # # # # # # # # # # # #

def corr_distance(corr_matrix: pd.DataFrame, plot: bool) -> pd.DataFrame:
    '''
    Input: correlation matrix
    Output: matrix of correlation distances with d_ij = sqrt( (1 - rho_ij) / 2)
    '''
    corr_matrix = corr_matrix.clip(-1, 1)
    corr_distance_matrix = np.sqrt((1 - corr_matrix) / 2)

    if plot:
        plt.figure(figsize=(10,8))
        sns.heatmap(corr_distance_matrix, annot=True, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.title('Correlation distances')
        plt.show()
    
    return corr_distance_matrix

def euclidean_distance(corr_distance_matrix: pd.DataFrame, plot: bool) -> pd.DataFrame:
    '''
    Input: correlation distance matrix
    Output: matrix of eucledian distances with D_ij = || d_i - d_j ||_2, where d_i, d_j are ith, jth column of input
    '''
    euc_distance_matrix = pairwise_distances(corr_distance_matrix)
    euc_distance_matrix = pd.DataFrame(
        euc_distance_matrix, 
        index = corr_distance_matrix.index, 
        columns = corr_distance_matrix.index
    )      

    if plot:
        plt.figure(figsize=(10,8))
        sns.heatmap(euc_distance_matrix, annot=True, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.title('Eucledian correlation distances')
        plt.show()
    
    return euc_distance_matrix

def tree_clustering(euc_distance_matrix: pd.DataFrame) -> list:
    '''
    Input: matrix of eucledian distances
    Output: array of tuples (cluster1_i, cluster2_i, min_distance_i), where cluster1_i, cluster2_i are the clusters
            joined in the i iteratin, and min_distance_i is the min distance between the clusters 
            (also min distance between any cluster).
            This linkage specifies the clustering of the tree
    '''
    n = len(euc_distance_matrix)
    linkage = []

    for _ in range(n - 1):
        mask = np.eye(len(euc_distance_matrix), dtype = bool)
        masked = euc_distance_matrix.mask(mask)

        i, j = masked.stack().idxmin()
        min_dist = masked.loc[i, j]
        new_label = i + '|' + j

        new_euc_distance = euc_distance_matrix[[i, j]].min(axis = 1)
        new_euc_distance.name = new_label

        euc_distance_matrix = euc_distance_matrix.drop(index = [i, j], columns = [i, j])
        euc_distance_matrix[new_label] = new_euc_distance.drop([i, j])
        euc_distance_matrix.loc[new_label] = new_euc_distance.drop([i, j])
        euc_distance_matrix.loc[new_label, new_label] = 0

        linkage.append((i, j, min_dist))
    
    return linkage

# %%
# # # # # # # # # # # # # #
# QUASI DIAGONALIZATION
# # # # # # # # # # # # #

def quasi_diagonalize(covariance_matrix: pd.DataFrame, linkage: list, plot: bool) -> tuple:
    '''
    Input: covariance matrix, linkage specifying the clustering
    Output: the quasi diagonal ordering of the stocks, the quasi diagonal covariance matrix
    '''

    root = linkage[-1][0] + '|' + linkage[-1][1]
    merge_info = {i + '|' + j: (i, j, dist) for i, j, dist in linkage}

    def find_quasi_diagonal_order(label: str, merge_info: dict):
        if label not in merge_info:
            return [label]

        i, j, dist = merge_info[label]
        return find_quasi_diagonal_order(i, merge_info) + find_quasi_diagonal_order(j, merge_info)

    quasi_diagonal_order = find_quasi_diagonal_order(root, merge_info)
    quasi_diagonal_cov = covariance_matrix.loc[quasi_diagonal_order, quasi_diagonal_order]
    
    if plot:
        plt.figure(figsize=(10,8))
        sns.heatmap(quasi_diagonal_cov, annot=True, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.title('Quasi-diagonal covariance')
        plt.show()

    return quasi_diagonal_order, quasi_diagonal_cov

# %%
# # # # # # # # # # # # # #
# RECURSIVE BISECTION
# # # # # # # # # # # # #

def recursive_bisection(quasi_diagonal_order: list, quasi_diagonal_cov: pd.DataFrame) -> pd.Series:
    '''
    Input: quasi diagonal ordering and covariance matrix
    Output: optimal weight distribution vector
    '''

    # Initialise all weights to 1 in quasi-diagonal ordering
    w = pd.Series(1.0, index=quasi_diagonal_order)
    clusters = [quasi_diagonal_order]

    # Recursively bisect clusets
    while len(clusters) > 0:
        # bisect each cluster
        new_clusters = []
        for cluster in clusters:
            if len(cluster) > 1:
                split = len(cluster) // 2
                new_clusters.append(cluster[:split])
                new_clusters.append(cluster[split:])
        
        clusters = new_clusters

        # update weights
        for i in range(0, len(clusters), 2):
            c1 = clusters[i]
            c2 = clusters[i + 1]

            var1 = get_cluster_var(quasi_diagonal_cov, c1)
            var2 = get_cluster_var(quasi_diagonal_cov, c2)
            
            alpha = 1 - var1 / (var1 + var2)
            w[c1] *= alpha
            w[c2] *= 1 - alpha
    
    return w

def get_cluster_var(quasi_diagonal_covariance, cluster):    
    '''
    Input: quasi diagonal covariance, cluster of stocks
    Output: cluster's total variance, < w, Corr w>, where w is the inverse variance, Corr is the submatrix of cluster 
    '''
    cur_covariance = quasi_diagonal_covariance.loc[cluster, cluster]
    cur_weight = get_inverse_variance(cur_covariance).reshape(-1, 1)
    cur_var = float(cur_weight.T @ cur_covariance.values @ cur_weight)
    return cur_var

def get_inverse_variance(covariance):
    '''
    Input: covariance matrix
    Output: diagonal matrix where diag_i = (1/var[i]) / (1/sum_j var[j])
    
    This uses the approximation of no cross-covariance between stocks, in which case the allocation is just normalised inverse var
    '''
    inverse_variance = 1 / np.diag(covariance)
    inverse_variance /= inverse_variance.sum()
    return inverse_variance

# %%

if __name__ == '__main__':
    from src.data import returns
    optimized_weights = hrp_portfolio(returns)
    optimized_weights.sort_values().plot(kind='barh', title=f'HRP Portfolio Weights', figsize=(8,5))

    print(optimized_weights)
# %%
