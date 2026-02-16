---

# Diffusion Transformer in Latent Space

## A From-Scratch Implementation with Rectified Flow and Distributed Streaming Training

---

## Abstract

This repository presents a from-scratch implementation of a latent diffusion model parameterized by a Transformer backbone with adaptive normalization conditioning.

The model operates in the latent space of a pretrained Variational Autoencoder (VAE) and is trained using a rectified-flow formulation of diffusion, where the objective reduces to direct velocity prediction.

The architecture integrates:

* Patch-based Vision Transformer tokenization
* Time-conditioned Adaptive LayerNorm (AdaLN-Zero)
* Symmetric cross-attention between image and text tokens
* Distributed streaming training over LAION metadata shards
* Exponential Moving Average (EMA) stabilization

This document describes the mathematical formulation, architectural components, and systems-level design decisions.

---

# 1. Problem Formulation

Let:

* x in R^(3 x H x W) be an image
* E_VAE encode images into latents
* z0 = E_VAE(x)
* c be conditioning tokens (e.g., caption embeddings)

We seek to learn a generative model over latent space z conditioned on c.

---

# 2. Diffusion as Rectified Flow

Instead of using a variance schedule beta_t, we define a linear interpolation process:

```
z_t = z0 + t * (z1 - z0)
```

where:

* t ~ Uniform(0, 1)
* z1 ~ Normal(0, I)

Define the target velocity:

```
v* = z1 - z0
```

The model learns:

```
v_theta(z_t, c, t) ~= v*
```

Training loss:

```
L = E_{z0, z1, t} [ || v_theta(z_t, c, t) - (z1 - z0) ||^2 ]
```

This reduces diffusion training to a regression problem over the velocity field.

---

# 3. Latent Diffusion

We operate in the latent space of a pretrained Stable Diffusion VAE.

Encoding:

```
z = mean( E_VAE(x) ) * s
```

where s is a scaling factor from the VAE configuration.

Advantages:

* ~16x spatial compression
* Reduced memory footprint
* Faster training
* Improved inductive bias

---

# 4. Model Architecture

The backbone is a Diffusion Transformer (DiT)-style encoder.

---

## 4.1 Patch Embedding

Latent tensors:

```
z in R^(B x C x H x W)
```

are converted into tokens via:

```
Conv2d(C -> d_model, stride = P)
```

Flattened into:

```
X in R^(B x N x d_model)
```

where:

```
N = (H / P)^2
```

Learned positional embeddings are added to token representations.

---

## 4.2 Time Embedding

We use sinusoidal timestep embeddings:

```
Emb(t) = [ sin(w_i * t), cos(w_i * t) ]
```

These are projected via an MLP and fused with caption embeddings:

```
t_tilde = W_t * Emb(t) + W_c * mean(c)
```

This fused time representation conditions all transformer layers.

---

## 4.3 Adaptive LayerNorm Zero (AdaLN-Zero)

For each transformer block:

```
x_norm = LayerNorm(x)

x = x_norm * (1 + gamma(t)) + beta(t)
```

Parameters gamma and beta are initialized to zero.

Benefits:

* Stable conditioning
* Reduced gradient explosion
* Smooth time-dependent modulation

---

## 4.4 Cross-Attention Between Image and Text

Each DiT layer processes:

* Image tokens x
* Caption tokens c

Procedure:

1. Pre-normalize both streams
2. Compute Q, K, V for both
3. Concatenate token sequences
4. Perform multi-head attention
5. Split outputs
6. Apply residual connections + MLP

This yields symmetric conditioning:

```
(x, c) -> (x', c')
```

Unlike U-Net diffusion architectures that use asymmetric cross-attention, this design enables full token-level interaction.

---

## 4.5 Multi-Head Self Attention

Attention is computed as:

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
```

This is implemented manually for architectural transparency and experimental flexibility.

---

# 5. Training Objective

Full training step:

1. Sample t
2. Compute z_t
3. Predict velocity
4. Compute MSE loss
5. Update model
6. Update EMA

Objective:

```
L = || v_theta(z_t, c, t) - (z1 - z0) ||^2
```

EMA stabilizes sampling trajectories and improves convergence.

---

# 6. Distributed Streaming Training

Training is performed over streamed LAION metadata shards using DistributedDataParallel (DDP).

Key characteristics:

* Rank-based sharding
* RAM-backed local image cache
* Infinite-step training loop (no fixed epochs)

Advantages:

* No full dataset download
* Scales linearly with GPU count
* Robust to failed URL fetches
* Enables web-scale training

---

# 7. Comparison to Standard DDPM

| Feature        | Traditional DDPM    | This Implementation    |
| -------------- | ------------------- | ---------------------- |
| Noise Schedule | beta-schedule       | Linear interpolation   |
| Prediction     | epsilon or x0       | Velocity               |
| Backbone       | U-Net               | Transformer            |
| Space          | Pixel               | Latent                 |
| Conditioning   | Cross-attn in U-Net | Symmetric token fusion |

---

# 8. Limitations

* Sampling process not yet implemented
* No classifier-free guidance
* No FlashAttention
* No FSDP
* No gradient checkpointing
* Caption encoder not integrated (assumes embeddings provided)

---

# 9. Conclusion

We presented a modular, from-scratch implementation of a Diffusion Transformer trained in latent space using a rectified-flow formulation.

The system emphasizes:

* Architectural clarity
* Scalability
* Research extensibility

This implementation bridges:

* Diffusion modeling theory
* Transformer conditioning
* Large-scale training systems
* Latent generative modeling

---
