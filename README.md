# Dimension Reduction of Single-cell RNA Data: Comparison of Autoencoder and PCA

### Code
`sma_single.py` contains code for autoencoder architecture used to reduce dimension of the gene expression data with Structured Masked Autoencoder.
`embeddings.ipynb` contains the code for UMAP dimensionality reduction of the embeddings learned by the autoencoder, as well as its visualizations with different types of coloring for the data points. 
`pca.ipynb` contains code for applying PCA on the expression data using different numbers of components, as well as

### Data
Data and metadata located in onedrive at the following link:
`anndata_human.h5ad` is the AnnData file for human gene expression data.
`Trevino.h5` is the complete HDF5 data file for the human gene expression data. This was used only for conversion into AnnData format, not for actual analysis. 
`Trevino_meta.csv` is the metadata containing cell types corresponding to the human gene expression data.

Onedrive Link: [CGDA_data](https://livejohnshopkins-my.sharepoint.com/:f:/r/personal/gli44_jh_edu/Documents/CGDA_data?csf=1&web=1&e=hhNCVJ)

### Instructions for Running
Run the autoencoder using the command `python sma_single.py`. The file saves the embedding list in the specified directory at the end of the main function (change accordingly). The Python notebooks for plotting data can be run by changing the appropriate file paths and hyperparameters in the notebooks.

### Conclusions
PCA and autoencoders are very distinct methods of dimensionality reduction for gene expression data. Applying UMAP to each method’s embeddings produced results that cluster the data ways that appear visually different and require slightly different biological interpretations. When its UMAP results were plotted, PCA clustered each cell type into distinct clusters that followed the expected differentiation trajectory. The components learned by the autoencoder had less distinct clusters, as different cell types were more mixed when plotted. Additionally, autoencoders are black boxes and are thus less interpretable than PCA. Therefore, when training the autoencoder, it was difficult to relate the different choices of hyperparameters to its output. 

The results thus favor PCA as a dimensionality reduction method for clustering stem cell type from gene expression data. This is somewhat surprising, as autoencoders have greater flexibility and can learn nonlinear relationships in input data. One possibility is that the amount of nonlinearities in this dataset were limited, such that PCA was still able to cluster its underlying cell types. However, this project only tested one type of autoencoder architecture, a structured autoencoder. Current work done by a colleague at Johns Hopkins showed that a variational autoencoder was able to cluster a different single-cell RNA gene expression data into more distinct clusters that better represent the cell differentiation pathway. Future work could involve testing the performance of a variational autoencoder on dimensionality reduction of this dataset.
