
# Diffusion Transformer in Latent Space

## A From-Scratch Implementation with Rectified Flow and Distributed Streaming Training

---

## Abstract

We present a from-scratch implementation of a latent diffusion model parameterized by a Transformer backbone with adaptive normalization conditioning. The model operates in the latent space of a pretrained Variational Autoencoder (VAE) and is trained using a rectified-flow formulation of diffusion, where the objective reduces to direct velocity prediction.

The architecture integrates:

* Patch-based Vision Transformer tokenization
* Time-conditioned Adaptive LayerNorm (AdaLN-Zero)
* Symmetric cross-attention between image and text tokens
* Distributed streaming training over LAION metadata shards
* Exponential Moving Average (EMA) stabilization

This report describes the mathematical formulation, architectural components, and systems-level design decisions.

---

# 1. Problem Formulation

Let:

* $x \in \mathbb{R}^{3 \times H \times W}$ be an image
* $\mathcal{E}_{VAE}$ encode images into latents
* $z_0 = \mathcal{E}_{VAE}(x)$
* $c$ be conditioning tokens (e.g., caption embeddings)

We seek to learn a generative model over latent space $z$ conditioned on $c$.

---

# 2. Diffusion as Rectified Flow

Instead of a variance schedule $\beta_t$, we define a linear interpolation process:

$$
z_t = z_0 + t (z_1 - z_0)
$$

where:

* $t \sim \mathcal{U}(0,1)$
* $z_1 \sim \mathcal{N}(0, I)$

Define the target velocity:

$$
v^* = z_1 - z_0
$$

The model learns:

$$
v_\theta(z_t, c, t) \approx v^*
$$

Training loss:

$$
\mathcal{L} = \mathbb{E}*{z_0, z_1, t}
\left[
\left| v*\theta(z_t, c, t) - (z_1 - z_0) \right|_2^2
\right]
$$

This simplifies training to a regression problem over the velocity field.

---

# 3. Latent Diffusion

We operate in the latent space of a pretrained Stable Diffusion VAE.

Encoding:

$$
z = \text{mean}(\mathcal{E}_{VAE}(x)) \cdot s
$$

where $s$ is a scaling factor from the VAE configuration.

Advantages:

* $\sim 16\times$ spatial compression
* Reduced memory footprint
* Faster training
* Improved inductive bias

---

# 4. Model Architecture

The backbone is a Diffusion Transformer (DiT)-style encoder.

---

## 4.1 Patch Embedding

Latent tensors
$$
z \in \mathbb{R}^{B \times C \times H \times W}
$$

are converted to tokens via:

$$
\text{Conv2d}(C \rightarrow d_{model}, \text{stride} = P)
$$

Flattened into:

$$
X \in \mathbb{R}^{B \times N \times d_{model}}
$$

where

$$
N = \left(\frac{H}{P}\right)^2
$$

Learned positional embeddings are added to token representations.

---

## 4.2 Time Embedding

We use sinusoidal timestep embeddings:

$$
\text{Emb}(t) =
\left[
\sin(\omega_i t), \cos(\omega_i t)
\right]
$$

These are projected via MLP and fused with caption embeddings:

$$
\tilde{t} = W_t \text{Emb}(t) + W_c ,\text{mean}(c)
$$

This fused time representation conditions all transformer layers.

---

## 4.3 Adaptive LayerNorm Zero (AdaLN-Zero)

For each transformer block:

$$
x_{norm} = \text{LayerNorm}(x)
$$

$$
x = x_{norm} \cdot (1 + \gamma(t)) + \beta(t)
$$

Parameters $\gamma$ and $\beta$ are initialized to zero.

Benefits:

* Stable conditioning
* Reduced gradient explosion
* Smooth time-dependent modulation

---

## 4.4 Cross-Attention Between Image and Text

Each DiT layer processes:

* Image tokens $x$
* Caption tokens $c$

Procedure:

1. Pre-normalize both streams
2. Compute $Q$, $K$, $V$ for both
3. Concatenate token sequences
4. Perform multi-head attention
5. Split outputs
6. Apply residual + MLP

This yields symmetric conditioning:

$$
(x, c) \rightarrow (x', c')
$$

Unlike U-Net diffusion architectures that use asymmetric cross-attention, this design enables full token-level interaction.

---

## 4.5 Multi-Head Self Attention

Attention is computed as:

$$
\text{Attention}(Q,K,V)
=

\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

This is implemented manually for architectural transparency and experimental flexibility.

---

# 5. Training Objective

Full training step:

1. Sample $t$
2. Compute $z_t$
3. Predict velocity
4. Compute MSE loss
5. Update model
6. Update EMA

Objective:

$$
\mathcal{L}
=

\left|
v_\theta(z_t, c, t)
-

(z_1 - z_0)
\right|_2^2
$$

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
| Noise Schedule | $\beta$-schedule    | Linear interpolation   |
| Prediction     | $\epsilon$ or $x_0$ | Velocity               |
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

We presented a modular, from-scratch implementation of a Diffusion Transformer trained in latent space using a rectified-flow formulation. The system emphasizes architectural clarity, scalability, and research extensibility.

This implementation bridges:

* Diffusion modeling theory
* Transformer conditioning
* Large-scale training systems
* Latent generative modeling

---
