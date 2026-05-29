# Cluster + Embed 

This repository contains the code to reproduce the experiments and figures in the paper [Cluster and then Embed: A Modular Approach for Visualization](https://arxiv.org/abs/2509.03373). 

Note: [Here](https://github.com/lizzycoda/cluster_embed) is an older version of this repo in R. The code here is in Python and has been updated to better handle larger datasets using subsampling in the alignment step. 

<p align="center"><img width="800" alt="Schematic of the cluster + embed approach" src="/figures/ce_schematic.png">

We propose a modular approach for visualization of clustered data that consists of first clustering the data, then embedding each cluster, and finally aligning the clusters to obtain a global embedding that preserves global distances. Any method can be chosen to cluster the data, and likewise, any method can be chosen to embed each cluster, based on what the user hopes to gain from the visualization. 

By embedding each cluster individually, we aim to produce an embedding that suffers from less distortion than if we were to embed all of the data together.  Additionally, the method includes a tuning parameter $\alpha$ to allow for separation between clusters. The method is competitive with existing methods, while offering much more flexibility and transparency. 

Below is an example of the C+E approach on the MNIST dataset,
    
<p align="center"><img width="800" alt="Example of the cluster + embed approach on MNIST" src="/figures/mnist_example.png">
    
For example usage, see this [tutorial](https://github.com/lizzycoda/ClusterEmbed_v2/blob/main/tutorial.ipynb). 

## Organization

All experiments in the paper and its appendix are in the examples folder, with a separate notebook for each dataset.

Each file contains code to obtain the C+E embedding, the other embeddings we compared against, and to evaluate the embeddins on a variety of different metrics. 

The repository contains the following files:

- `code/`
  - `align.py` contains the main code for the alignment step of method.
  - `embed.py` contains the code for Landmark Isomap. For all other embedding methods, we used existing implementations. 
  - `eval.py` contains the code to evaluate embeddings on the metrics used in the paper, as well as produce plots

- `examples/` contains the examples in the paper, listed below in the order they appear in the paper.
  - `gmm.ipynb`
  - `shapes.ipynb`
  - `mnist.ipynb`
  - `human_brain.ipynb`
  - `fmnist.ipynb`
  - `mouse_cortex.ipynb`
  
- `additional_experiments/` contains the appendix experiments that evaluate the sensitivity of C+E.
    - `mnist_clustering_effect.ipynb`
    - `human_brain_clustering_effect.ipynb`
    - `mnist_alpha_tuning.ipynb`
    - `human_brain_alpha_tuning.ipynb`
  
## Data

The synthetic datasets used in the paper are available in the `data\` folder.

The MNIST dataset was downloaded using the [torchvision API](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) and the Fashion MNIST dataset was downloaded from [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist).
              
The human brain organoid data is from [Kanton et al. (2019)](https://www.nature.com/articles/s41586-019-1654-9). We downloaded the [metadata](https://www.ebi.ac.uk/biostudies/files/E-MTAB-7552/metadata_human_cells.tsv) and [counts data](https://www.ebi.ac.uk/biostudies/files/E-MTAB-7552/human_cell_counts_consensus.mtx), and used the preprocessing of [Damrich et al.](https://www.biorxiv.org/content/10.1101/2024.04.26.590867v1.abstract), using [this code](https://github.com/berenslab/ne_spectrum_scRNAseq/tree/main).

The mouse cortex data is from [Tasic et al. (2018)](https://www.nature.com/articles/s41586-018-0654-5) downloaded from [here](http://celltypes.brain-map.org/api/v2/well_known_file_download/694413985). We then used the preprocessing in [Kobak & Berens, 2019](https://www.nature.com/articles/s41467-019-13056-x) with code available [here](https://github.com/berenslab/rna-seq-tsne/blob/master/tasic-et-al.ipynb). 
          
## Citation 

```
@article{cluster_embed26,
      title={Cluster and then Embed: A Modular Approach for Visualization}, 
      author={Elizabeth Coda and Ery Arias-Castro and Gal Mishne},
      journal={arXiv preprint arXiv: 2509.03373}
      year={2026}
}
```