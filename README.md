# Title :
Class-Informed Variational AutoEncoders (CI-VAEs) for Enhanced Within Class Data Interpolation


## Abstract:
We proposed Class-Informed Variational Autoencoder (CI-VAE) to enable interpolation between arbitrary pairs of observations of the same class. CI-VAE combines the general VAE architecture with a linear discriminator layer on the latent space to enforce the construction of a latent space such that observations from different classes are linearly separable. In conventional VAEs, class overlapping on the latent space usually occur. However, in CI-VAE, the enforced linear separability of classes on the latent space allows for robust latent-space linear traversal and data generation between two arbitrary observations of the same class. Class-specific data interpolation has extensive potential applications in science and particularly in biology such as uncovering the biological trajectory of diseases or cancer. We used the MNIST dataset of handwritten digits as a case study to compare the performance of CI-VAE and VAE in class-specific data augmentation. We showed that CI-VAE significantly improved class-specific linear traversal and data augmentation compared with VAE while maintaining comparable reconstruction error.
<img width="750" alt="image" src="https://user-images.githubusercontent.com/11249004/209891763-df430c00-9d67-485a-9b95-521cb0839640.png">

