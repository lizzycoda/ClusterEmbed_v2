import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.manifold import ClassicalMDS
from scipy.optimize import minimize_scalar, minimize

class Embedding:
    def __init__(self,embeddings, clusters, distances, alpha = 1):
        """
        embeddings: (n,2) array of embeddings.
        clusters: length n array with cluster labels 0 through k.
        distances: dictionary where the value of (i,j) is the pairwise distances between cluster i and cluster j. Convention is i<j.
        alpha: scalar to scale between cluster distances for in the aligned embedding 
        """

        self._cluster_preprocess(clusters)
        self.embeddings = self._embedding_preprocess(embeddings, clusters)
        self.n = len(embeddings)

        if type(distances) is dict:
            self.distances = distances
        else:
            self.distances = self._distances_preprocess(distances) #TO DO
        self.alpha = alpha
        if self.alpha != 1:
            self._scale_distances()

        self.translations = np.zeros((self.n_clusters,2))
        self.rotations =np.zeros(self.n_clusters) 
        self.reflections = np.zeros(self.n_clusters) 
        
    ##################################################### 
    ### Preprocessing functions #########################
    ##################################################### 
    def _cluster_preprocess(self,clusters):
        self.clusters = clusters
        unique, counts = np.unique(clusters, return_counts = True)
        self.n_clusters = len(unique)
        self.unique_clusters = unique
        self.cluster_counts = {i:count for i,count in zip(unique,counts)}
        
        
    def _embedding_preprocess(self, embeddings, clusters):
        embedding_dict = {}
        for i in self.unique_clusters:
            emb_i = embeddings[clusters ==i]
            embedding_dict[i] = emb_i 
        return embedding_dict
        
    def _scale_distances(self):
        #make a new copy so that original input is left unchanged 
        self.distances = {key:self.alpha*val for key,val in self.distances.items()}
        
    def get_default_alpha(self, m = 1000,q = .95):
        cluster_diams = []
        for i in range(self.n_clusters):
            if self.cluster_counts[i] > m:
                idxs = np.random.choice(self.cluster_counts[i], m, replace = False)
                emb_cluster = self.embeddings[i][idxs]
                cluster_diams += [np.quantile(pdist(emb_cluster), q)]
            else:
                emb_cluster = self.embeddings[i]
                cluster_diams += [np.max(pdist(emb_cluster))]
        tau  = np.mean(cluster_diams)
    
        
        # get average distance between clusters
        deltas = []
        for i in range(self.n_clusters):
            for j in range(i+1, self.n_clusters):
                deltas += [np.mean(self.distances[i,j])]
        Delta = np.mean(deltas)
        alpha = np.max([1,self.n_clusters*tau/(2*np.pi*Delta)])
        return alpha
            

                
        
    ##################################################### 
    ### Embedding functions #############################
    ##################################################### 
    def get_global_alignment(self):
        final_emb = np.zeros((self.n,2))
        for i in self.unique_clusters:
            cluster_emb = self.get_cluster_embedding(i)
            final_emb[self.clusters == i] = cluster_emb
        final_emb = final_emb - np.mean(final_emb, axis = 0) #center embedding
        return final_emb
               
        
    def embed_points(self,pts, clusters):
        # given an existing alignment, globally align additional points given their cluster level embeddings

        final_emb = np.zeros(pts.shape)
        for i in range(self.n_clusters):
            idxs = np.where(clusters == i)[0]
            
            Z = pts[idxs,:]
            rotation = self.rotations[i]
            reflection = self.reflections[i]
            translation =  self.translations[i]
            
            final_emb[idxs,:] =  self.rigid_rotation(Z,rotation, reflection)
            final_emb[idxs,0] += translation[0]
            final_emb[idxs,1] += translation[1]
        final_emb = final_emb - np.mean(final_emb, axis = 0) #center embedding   
        return final_emb
        
    def get_cluster_embedding(self, cluster_idx, subsampled_idxs = None, rotation = None, reflection = None, translation = None):
        
        if rotation is None:
            rotation = self.rotations[cluster_idx]
        
        if reflection is None:
            reflection = self.reflections[cluster_idx]
            
        if translation is None:
            translation = self.translations[cluster_idx,:]
        
        if subsampled_idxs is None:
            Z = self.embeddings[cluster_idx]
        else:
            Z = self.embeddings[cluster_idx][subsampled_idxs]
        out = self.rigid_rotation(Z,rotation, reflection)
        out[:,0] += translation[0]
        out[:,1] += translation[1]
        return out
        
    def orthogonal_matrix(self, theta, reflection = 0):
        R = np.array([ [np.cos(theta), np.sin(theta)],
                        [-np.sin(theta), np.cos(theta)]])
    
        if reflection == 1:
            reflection_matrix = np.array([[1, 0],
                                          [0, -1]])
            R = reflection_matrix @ R

        return R
    
    def rigid_rotation(self, Z, rotation, reflection):
        R = self.orthogonal_matrix(rotation, reflection)
        Z_transformed = Z @ R.T # Z is nx2

        return Z_transformed 

    ##################################################### 
    ### Alignment functions ############################
    #####################################################  
    
    def align_embedding(self, m = None, max_init_iter = 10, max_iter = 10):
        # if beta = 1 means use all the available points in the alignmetn
        # otherwise, m controls the number of points sampled per cluster 
        #(future implementaiton could be to change the default of this to do the sampling some other way...) 
        
        self.translations = self.initialize_translations()
        
        #stage 1 of alignment: align every cluster with respect to an anchor cluster 
        fixed_cluster = max(self.cluster_counts, key=self.cluster_counts.get)
        moveable_clusters = self.unique_clusters[ self.unique_clusters != fixed_cluster]

        thresh = .01
        for j in moveable_clusters:
            for iter_ in range(max_init_iter):
                    loss = self.align_cluster(j, [fixed_cluster], m)
                    curr_loss = loss

                    if iter_ > 1:
                        if(abs(curr_loss - prev_loss)<thresh):
                            break
                    prev_loss = curr_loss

        #stage 2 of alignment: anchor cluster is still fixed
        for iter_ in range(max_iter):
            np.random.shuffle(moveable_clusters)
            curr_loss = 0
            for j in moveable_clusters:
                loss = self.align_cluster(j,  self.unique_clusters[ self.unique_clusters != j], m)
                curr_loss += loss
            if iter_ > 1:
                if(abs(curr_loss - prev_loss)<thresh):
                    break
            prev_loss = curr_loss
            

        #stage 3 of alignment: align every cluster (including the anchor cluster)
        for iter_ in range(max_iter):
            np.random.shuffle(self.unique_clusters)
            curr_loss = 0
            for j in moveable_clusters:
                loss = self.align_cluster(j,  self.unique_clusters[ self.unique_clusters != j], m)
                curr_loss += loss
            if iter_ > 1:
                if(abs(curr_loss - prev_loss)<thresh):
                    break
            prev_loss = curr_loss
            
        return self.get_global_alignment()

    
    def align_cluster(self, cluster_idx, anchors, m):
        if m is None:
            iter_idxs = None 
        else:
            iter_idxs = self.subsample_idxs(cluster_idx, anchors, m)
            
        #first step optimize the orthogonal transformation
        translation = self.translations[cluster_idx]        
        rot_args = (cluster_idx, 0, translation, anchors, iter_idxs)         
        rot_out = minimize_scalar(self.rotation_stress, args = rot_args, bounds=(0, 2*np.pi), method='bounded')
    
        ref_args =  (cluster_idx, 1, translation, anchors, iter_idxs)      
        ref_out = minimize_scalar(self.rotation_stress, args = ref_args, bounds=(0, 2*np.pi), method='bounded')        

        if ref_out.fun < rot_out.fun:
            reflection = 1
            rotation = ref_out.x
        else:
            reflection = 0
            rotation = rot_out.x
        self.reflections[cluster_idx] = reflection
        self.rotations[cluster_idx] = rotation       
        
        #second step is optimize the translation 
        translation_args = (cluster_idx, rotation, reflection, anchors, iter_idxs)
        translation_out = minimize(self.translation_stress_with_grad, self.translations[cluster_idx], 
                                   args = translation_args,
                                   method='BFGS',
                                  jac = True)

        loss = translation_out.fun
        self.translations[cluster_idx] = translation_out.x
        return loss
        
            
    def subsample_idxs(self,cluster_idx, anchors, m):
        #sample m idxs from each cluster 
        #if less than m idxs just return every idx in the cluster 
        anchors = list(anchors)
        iter_idxs = {i: np.random.choice(self.cluster_counts[i], 
                                                   size=min(m, self.cluster_counts[i]),
                                                   replace=False) for i in [cluster_idx] + anchors}
        
        return iter_idxs
        
        
     
    def initialize_translations(self):
        n_clusters = self.n_clusters

        mean_cluster_distances = np.zeros((n_clusters,n_clusters ))
        for i in range(n_clusters):
            for j in range(i+1, n_clusters):
                batch_distances = self.distances[i,j]
                mean_cluster_distances[i,j] = np.mean(batch_distances)
                mean_cluster_distances[j,i] = mean_cluster_distances[i,j]
                
        mds = ClassicalMDS(n_components=2, metric='precomputed')
        translations = mds.fit_transform(mean_cluster_distances)
        return translations
    
    ##################################################### 
    ### Stress functions ################################
    #####################################################  
    
    def cluster_stress(self, cluster_idx, rotation, reflection, translation,anchors,subsampled_idxs):
        #compute the pairwise stress between the indicated cluster with the specified transformations and the fixed anchor clusters
        #if applicable, use only the subsampled idxs
        stress  = 0
        if subsampled_idxs is not None:
            
            cluster1_embeddings = self.get_cluster_embedding(cluster_idx, 
                                                            subsampled_idxs[cluster_idx],
                                                            rotation = rotation,
                                                            reflection = reflection,
                                                            translation = translation)
            idx1 = subsampled_idxs[cluster_idx]
            
            for j in anchors:
                cluster2_embeddings = self.get_cluster_embedding(j, subsampled_idxs[j])
                idx2 = subsampled_idxs[j]
                
                if cluster_idx < j:
                    emb_distances = cdist(cluster1_embeddings,cluster2_embeddings)
                    orig_distances = self.distances[cluster_idx, j][np.ix_(idx1,idx2)]
                else:
                    emb_distances = cdist(cluster2_embeddings,cluster1_embeddings)
                    orig_distances = self.distances[j,cluster_idx][np.ix_(idx2,idx1)]
            
                stress += np.sum((orig_distances - emb_distances)**2)
            
        else:
            cluster1_embeddings = self.get_cluster_embedding(cluster_idx,
                                                            rotation = rotation,
                                                            reflection = reflection,
                                                            translation = translation)
            for j in anchors:
                cluster2_embeddings = self.get_cluster_embedding(j)            
    
                if cluster_idx < j:
                    emb_distances = cdist(cluster1_embeddings,cluster2_embeddings)
                    orig_distances = self.distances[cluster_idx, j]
                else:
                    emb_distances = cdist(cluster2_embeddings,cluster1_embeddings)
                    orig_distances = self.distances[j,cluster_idx]
            
                stress += np.sum((orig_distances - emb_distances)**2)



        return stress   

    
    def rotation_stress(self, rotation, cluster_idx, reflection, translation,anchors,subsampled_idxs):
        return self.cluster_stress(cluster_idx, rotation, reflection, translation,anchors,subsampled_idxs)

    def translation_stress_with_grad(self, translation, cluster_idx, rotation, reflection, anchors,subsampled_idxs):
        #computes stress with indicated translation and also the gradient simultaneously 
        stress = 0
        grad =  np.array([0.,0.])

        if subsampled_idxs is not None:
            cluster1_embeddings = self.get_cluster_embedding(cluster_idx, 
                                                            subsampled_idxs[cluster_idx],
                                                            rotation = rotation,
                                                            reflection = reflection,
                                                            translation = translation)
            idx1 = subsampled_idxs[cluster_idx]
        
            for j in anchors:
                cluster2_embeddings = self.get_cluster_embedding(j, subsampled_idxs[j])
                idx2 = subsampled_idxs[j] 
                
                if cluster_idx < j:
                    emb_distances = cdist(cluster1_embeddings,cluster2_embeddings)
                    orig_distances = self.distances[cluster_idx, j][np.ix_(idx1,idx2)]
                    grad += self._pairwise_grad(cluster1_embeddings, cluster2_embeddings, orig_distances, emb_distances)
                    
                else:
                    emb_distances = cdist(cluster2_embeddings,cluster1_embeddings)
                    orig_distances = self.distances[j,cluster_idx][np.ix_(idx2,idx1)]
                    grad += self._pairwise_grad(cluster1_embeddings, cluster2_embeddings, orig_distances.T, emb_distances.T)
                
            
                stress += np.sum((orig_distances - emb_distances)**2)

        else:
            cluster1_embeddings = self.get_cluster_embedding(cluster_idx, 
                                                            rotation = rotation,
                                                            reflection = reflection,
                                                            translation = translation)

        
            for j in anchors:
                cluster2_embeddings = self.get_cluster_embedding(j)
                
                if cluster_idx < j:
                    emb_distances = cdist(cluster1_embeddings,cluster2_embeddings)
                    orig_distances = self.distances[cluster_idx, j]
                    grad += self._pairwise_grad(cluster1_embeddings, cluster2_embeddings, orig_distances, emb_distances)
                    
                else:
                    emb_distances = cdist(cluster2_embeddings,cluster1_embeddings)
                    orig_distances = self.distances[j,cluster_idx]
                    grad += self._pairwise_grad(cluster1_embeddings, cluster2_embeddings, orig_distances.T, emb_distances.T)
                
            
                stress += np.sum((orig_distances - emb_distances)**2)
    
    
        return stress, grad

    def _pairwise_grad(self,embeddings1, embeddings2, orig_distances, emb_distances):
        #calculate gradient of stress with respect to translation vector applied to cluster 1, everything else held fixed.
        
        # Compute pairwise differences for x and y coordinates
        embedding_x_diff = embeddings1[:, 0][:, None] - embeddings2[:, 0][None, :]
        embedding_y_diff = embeddings1[:, 1][:, None] - embeddings2[:, 1][None, :]

        # Compute gradient components
        dx = np.sum((emb_distances - orig_distances) * embedding_x_diff / emb_distances)
        dy = np.sum((emb_distances - orig_distances) * embedding_y_diff / emb_distances)

        return np.array([dx, dy])
    
        
        