# PCA_Image_Reconstruction

### Learn about,
[Principal Component Analysis](https://github.com/rjnp2/Data-Science/blob/main/tutorial/6.%20Machine%20Learning/7.%20Dimensionality%20reduction/PCA/readme.md)

## PCA for Compression
Obviously after dimensionality reduction, the training set takes up much less space. For example, try applying PCA to the MNIST dataset while preserving 95% of its variance. You should find that each instance will have just over 150 features, instead of the original 784 features. \
It is also possible to decompress the reduced dataset back to 784 dimensions by applying the inverse transformation of the PCA projection. Of course this won’t give you back the original data, since the projection lost a bit of information (within the 5% variance that was dropped), but it will likely be quite close to the original data. The mean squared distance between the original data and the reconstructed data (compressed and then decompressed) is called the reconstruction error. For example, the following code compresses the MNIST dataset down to 154 dimensions, then uses the inverse_transform() method to decompress it back to 784 dimensions.

  ```python
    pca = PCA(n_components = 154)
    X_mnist_reduced = pca.fit_transform(X_mnist)
    X_mnist_recovered = pca.inverse_transform(X_mnist_reduced)
 ```
  ![image](https://user-images.githubusercontent.com/58425689/108389569-e6730f00-7237-11eb-83cd-c60dea85aab8.png)
