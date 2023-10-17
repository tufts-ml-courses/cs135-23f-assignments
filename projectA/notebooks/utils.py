import numpy as np


def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    ''' Compute and return k-nearest neighbors under Euclidean distance

        Args
        ----
        data_NF : 2D np.array, shape = (n_examples, n_feats) == (N, F)
            Each row is a feature vector for one example in dataset
        query_QF : 2D np.array, shape = (n_queries, n_feats) == (Q, F)
            Each row is a feature vector whose neighbors we want to find
        K : int, must satisfy K >= 1 and K <= n_examples aka N
            Number of neighbors to find per query vector

        Returns
        -------
        neighb_QKF : 3D np.array, (n_queries, n_neighbors, n_feats) == (Q, K, F)
            Entry q,k is feature vector of k-th nearest neighbor of the q-th query
            If two vectors are equally close, then we break ties by taking the one
            appearing first in row order in the original data_NF array
    '''
    # Unpack to get number of examples (N), features (F), and queries (Q)
    N, F = data_NF.shape
    Q, F2 = query_QF.shape
    assert F == F2

    K = int(K)
    if K < 1:
        raise ValueError("Invalid number of neighbors (K). Too small.")
    if K > N:
        raise ValueError("Invalid number of neighbors (K). Too large.")

    # We'll fill the neighbors array one query at a time
    neighbors_QKF = np.zeros((Q, K, F))
    closest_ids_per_query = []
    for q in range(Q):
        # Compute squared euclidean distance from query to N data vectors
        # Only relative distance needed, so squared euclidean works
        # as well as euclidean
        dist_N = np.sum(np.square(data_NF - query_QF[q:q+1]), axis=1)

        # Get the K example ids that have the smallest distance
        # Remember, argsort defaults to ascending order (lowest to highest)
        # Stable sort is the way to make sure the order of neighbors
        # in data breaks is preserved in the result
        closest_ids_by_distance_K = np.argsort(dist_N, kind='stable')[:K]

        # Fill the neighbors array with the K closest data vectors
        neighbors_QKF[q, :, :] = data_NF[closest_ids_by_distance_K]
        closest_ids_per_query.append(closest_ids_by_distance_K)
        
    
    return neighbors_QKF, closest_ids_per_query