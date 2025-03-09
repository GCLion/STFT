## Spatial-Temporal Forgery Trace based Forgery Image Identification

![STFT Framework](pics/img1.png)

As illustrated in the figure, STFT utilizes the diffusion process to map the image into the latent distribution space and extracts forgery traces from this latent distribution through comprehensive spatial-temporal analysis. The overall framework consists of the following three main modules:

- **Temporal Prior Correlation Modeling**: Analyzes the latent space representations at different time steps of the diffusion process to capture their temporal variation characteristics.
- **Spatial Correlation Modeling**: Extracts features from different latent dimensions and computes spatial correlations using a self-attention mechanism to model spatial dependencies.
- **Frequency-Enhanced Attention Mechanism**: Leverages frequency domain information to guide temporal prior correlation computation and spatial correlation analysis, thereby accelerating forgery trace localization and improving model generalization by disregarding the interference of irrelevant features.