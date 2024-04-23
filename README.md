# Dimension Reduction of Single-cell RNA Data: Comparison of Autoencoder and PCA

### Code
‘sma_single.py’ contains code for autoencoder architecture used to reduce dimension of the gene expression data with Structured Masked Autoencoder.
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