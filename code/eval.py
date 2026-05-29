import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import orthogonal_procrustes


def procrustes(X,Y):
    #useful for aligning embeddings 
    X = X- np.mean(X,axis =0 )
    Y = Y - np.mean(Y, axis = 0)
    R, _ = orthogonal_procrustes(Y,X)
    Y = Y @ R
    return (Y,R)

def normalized_stress(orig_distances, emb_distances, clusters):
    
    #input format is a matrix of the distances 
    #returns the normalized stress within each cluster, between clusters, and over all points
    
    n_clusters = len(np.unique(clusters))
    total_stress= 0
    between_cluster_stress = np.zeros(int(n_clusters*(n_clusters-1)/2))
    within_cluster_stress = np.zeros(n_clusters)
    
    k = 0
    for i in range(n_clusters):
        idx_i = np.where(clusters == i)[0]
        
        if len(idx_i) == 1:
             within_cluster_stress[i] = np.nan

        else:
            within_cluster_orig = squareform(orig_distances[np.ix_(idx_i, idx_i)])
            within_cluster_emb  = squareform(emb_distances[np.ix_(idx_i, idx_i)])

            within_cluster_stress[i] = np.sum((within_cluster_orig  - within_cluster_emb)**2)
            total_stress += within_cluster_stress[i]
            within_cluster_stress[i] /= np.sum(within_cluster_orig**2)
        
        for j in range(i+1, n_clusters):
            idx_j = np.where(clusters == j)[0]
            
            if len(idx_i) ==1 and len(idx_j) == 1:
                between_cluster_stress[k] = np.nan
            else:
                between_cluster_orig = orig_distances[np.ix_(idx_i, idx_j)]
                between_cluster_emb  = emb_distances[np.ix_(idx_i, idx_j)]
                between_cluster_stress[k] = np.sum((between_cluster_orig -between_cluster_emb )**2)
                total_stress += between_cluster_stress[k] 
                between_cluster_stress[k]  /= np.sum(between_cluster_orig**2)
            k+=1
    total_stress /= np.sum(squareform(orig_distances)**2)
    return (np.nanmean(np.sqrt(within_cluster_stress)),  np.nanmean(np.sqrt(between_cluster_stress)), np.sqrt(total_stress))
        

def scale_normalized_stress(orig_distances, emb_distances, clusters):
    # scale noramlized stress, as described here: https://arxiv.org/abs/2408.07724
    stress = _scale_normalized_helper(squareform(orig_distances), squareform(emb_distances))
    
    n_clusters = len(np.unique(clusters))
    between_cluster_stress = np.zeros(int(n_clusters*(n_clusters-1)/2))
    within_cluster_stress = np.zeros(n_clusters)
    k = 0
    for i in range(n_clusters):
        idx_i = np.where(clusters == i)[0]
        
        if len(idx_i) == 1:
            within_cluster_stress[i] = np.nan
        else:
            within_cluster_orig = squareform(orig_distances[np.ix_(idx_i, idx_i)])
            within_cluster_emb  = squareform(emb_distances[np.ix_(idx_i, idx_i)])

            within_cluster_stress[i] =  _scale_normalized_helper(within_cluster_orig, within_cluster_emb )
        
        for j in range(i+1, n_clusters):
            idx_j = np.where(clusters == j)[0]
            
            if len(idx_i) ==1 and len(idx_j) == 1:
                between_cluster_stress[k] = np.nan
            else:
                between_cluster_orig = orig_distances[np.ix_(idx_i, idx_j)].flatten()
                between_cluster_emb  = emb_distances[np.ix_(idx_i, idx_j)].flatten()
                between_cluster_stress[k] = _scale_normalized_helper(between_cluster_orig, between_cluster_emb)
            k+=1
    
    return (np.nanmean(within_cluster_stress), np.nanmean(between_cluster_stress), stress)

def _scale_normalized_helper(orig_distances, emb_distances):
    #inputs should be flattened 
    alpha = np.sum(emb_distances*orig_distances)/np.sum(emb_distances**2)
    stress = np.sum((orig_distances - alpha*emb_distances)**2)/np.sum(orig_distances**2)
    return np.sqrt(stress)
    
    
def spearman(orig_distances, emb_distances, clusters):
    
    n_clusters = len(np.unique(clusters))
    total_spearman = spearmanr(squareform(orig_distances), squareform(emb_distances)).statistic
    between_cluster = np.zeros(int(n_clusters*(n_clusters-1)/2))
    within_cluster = np.zeros(n_clusters)
    
    k = 0
    for i in range(n_clusters):
        idx_i = np.where(clusters == i)[0]
        if len(idx_i) ==1:
            within_cluster[i] = np.nan
            
        else:
            within_cluster_orig = squareform(orig_distances[np.ix_(idx_i, idx_i)]).flatten()
            within_cluster_emb  = squareform(emb_distances[np.ix_(idx_i, idx_i)]).flatten()
            within_cluster[i] = spearmanr(within_cluster_orig,within_cluster_emb).statistic

       
        for j in range(i+1, n_clusters):
            idx_j = np.where(clusters == j)[0]
            
            if len(idx_i) ==1 and len(idx_j) == 1:
                between_cluster[k] = np.nan
            else:
                between_cluster_orig = orig_distances[np.ix_(idx_i, idx_j)].flatten()
                between_cluster_emb  = emb_distances[np.ix_(idx_i, idx_j)].flatten()
                between_cluster[k] = spearmanr(between_cluster_orig,between_cluster_emb).statistic
            k+=1
   
    return (np.nanmean(within_cluster), np.nanmean(between_cluster), total_spearman)


def class_preservation(orig_distances, emb_distances, classes):
    # take in a square matrix 

    unique_classes = np.unique(classes)
    m = len(unique_classes)

    avg_class_distances_orig = np.full((m, m), np.nan)
    avg_class_distances_emb = np.full((m, m), np.nan)

    for i in range(m - 1):
        idxs1 = np.where(classes == unique_classes[i])[0]
        for j in range(i + 1, m):
            idxs2 = np.where(classes == unique_classes[j])[0]

            avg_orig = np.mean(orig_distances[np.ix_(idxs1, idxs2)])
            avg_emb = np.mean(emb_distances[np.ix_(idxs1, idxs2)])

            avg_class_distances_orig[i, j] = avg_orig
            avg_class_distances_orig[j, i] = avg_orig

            avg_class_distances_emb[i, j] = avg_emb
            avg_class_distances_emb[j, i] = avg_emb

    # for each class calculate spearman correlation
    correlation_scores = []
    for i in range(m):
        orig_row = avg_class_distances_orig[i, :]
        emb_row = avg_class_distances_emb[i, :]

        mask = ~np.isnan(orig_row) & ~np.isnan(emb_row)
        orig = orig_row[mask]
        emb = emb_row[mask]

        correlation_scores += [spearmanr(orig, emb).statistic]
    return np.mean(correlation_scores)
    
def knn_recall(orig, emb, k_vals):
    max_k = max(k_vals)
    n = orig.shape[0]

    knn1 = NearestNeighbors(n_neighbors=max_k, metric="euclidean")
    knn1.fit(orig)
    idxs1 = knn1.kneighbors(return_distance=False)

    knn2 = NearestNeighbors(n_neighbors=max_k, metric="euclidean")
    knn2.fit(emb)
    idxs2 = knn2.kneighbors(return_distance=False)

    scores = []

    for k in k_vals:
        nbhd1 = idxs1[:, :k]  # (n, k)
        nbhd2 = idxs2[:, :k]  # (n, k)

        # Vectorized overlap: check membership
        # shape: (n, k, k) → compare each element in nbhd1 to nbhd2
        matches = (nbhd1[:, :, None] == nbhd2[:, None, :])

        # any match along last axis → whether each neighbor appears in other set
        overlap_counts = matches.any(axis=2).sum()

        scores.append(overlap_counts / (k * n))

    return scores