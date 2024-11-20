# Denoising Diffusion Probabilistic Models (DDPM) Implementation

This repository provides a comprehensive implementation of Denoising Diffusion Probabilistic Models (DDPM) for generating images, with a particular emphasis on the inference process. The model is specifically trained on RGB images with dimensions of 128x128 pixels.

## Model Architecture

The model utilizes a U-Net architecture characterized by:
- Skip connections that link encoder and decoder blocks
- Time embeddings incorporated at each block
- Residual connections to enhance learning
- Batch normalization for stable training
- Optimization for processing 128x128 RGB images

## Inference Process

The inference, or sampling process, involves reversing the diffusion process to synthesize new images:

1. Begin with random noise \( x_T \) drawn from a standard normal distribution \( \mathcal{N}(0, I) \).
2. Iteratively denoise the image for each timestep \( t \) from \( T-1 \) to \( 0 \) using the formula:
   ```python
   x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t \cdot z
   ```
   where:
   - \( x_t \) is the noisy image at timestep \( t \)
   - \( \alpha_t \) and \( \bar{\alpha}_t \) are diffusion parameters
   - \( \epsilon_\theta \) represents the noise prediction network
   - \( z \) is random Gaussian noise
   - \( \sigma_t \) is the noise scale

3. The final denoised image \( x_0 \) is obtained after completing all \( T \) steps.

### Key Components

- `get_beta_schedule()`: Function to generate a linear noise schedule.
- `generate_sample()`: Core sampling function that:
  - Initiates from random noise 
  - Executes reverse diffusion steps
  - Outputs the generated RGB image

@Inference of DDPM.ipynb
