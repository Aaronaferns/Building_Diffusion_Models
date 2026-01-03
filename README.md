

# Diffusion to Consistency

### DDPM baseline -> iterative modern upgrades -> Consistency Models (Maybe)

A small, readable diffusion repository that starts with a **vanilla DDPM-style ε-prediction model** and gradually moves toward more modern diffusion systems (score-based modeling, improved samplers, stability tricks, and eventually Stable-Diffusion-like components) **in small additive increments with minimal architectural disruption**.

This repo currently trains a **U-Net with timestep embeddings and selective self-attention** on **CIFAR-10 (32×32)** and performs ancestral sampling from pure Gaussian noise.

![Forward diffusion](images/forward_noising.png)

Core components include:

* CIFAR-10 dataloader with normalization to **[-1, 1]**
* Forward diffusion, training step, and ancestral sampling loop
* U-Net backbone with ResBlocks, GroupNorm, timestep embeddings, and attention
* Training script with checkpointing and loss visualization
* Sinusoidal timestep embedding utilities

---

## Why this repo exists

Most diffusion repositories jump directly into large frameworks, heavy abstractions, or latent diffusion pipelines. This repo is intentionally different:

* **Start minimal** (DDPM baseline)
* **Instrument heavily** (plots, sample grids, trajectories)
* **Add improvements iteratively** (one concept per PR)
* **Avoid architectural churn** (keep the U-Net interface stable)

If you want a *from-scratch but not toy* stepping stone toward modern diffusion systems, this repo is designed for that purpose.

---

## Current Implementation (Baseline)

### Data

* CIFAR-10 training set
* Random horizontal flip
* `ToTensor()` followed by `Normalize((0.5,…),(0.5,…))` → **[-1, 1]**

---

### Diffusion Process

* Linear beta schedule from `1e-4` to `0.02`
* Total timesteps: `T = 1000`
* Precomputed quantities:

  * `α_t = 1 − β_t`
  * `\bar{α}_t = ∏ α_t`

Training procedure:

* Sample `t ~ Uniform({0 … T−1})`
* Generate
  [
  x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon
  ]
* Predict ε with the U-Net
* Optimize mean-squared error loss

---

### Forward Diffusion Behavior

The figure below visualizes the forward diffusion process applied to the same CIFAR-10 images at increasing timesteps. As expected, structure is gradually destroyed as noise variance increases.

![Forward diffusion behavior](images/forward_noising.png)

---

### Model: U-Net

* Encoder–decoder U-Net operating directly in pixel space
* ResBlocks consist of:

  * GroupNorm → SiLU → Conv
  * timestep embedding projection added to hidden activations
  * dropout + second conv with **zero initialization**
* Selective single-head self-attention at chosen resolutions (`attn_res`)
* Sinusoidal timestep embeddings followed by a 2-layer MLP
* Residual connections throughout (ResBlocks and Attention blocks)

The model interface is intentionally kept simple:

```
model(x_t, t) → ε̂
```

This allows objective and sampler upgrades without redesigning the backbone.

---

### Sampling

Sampling follows standard ancestral DDPM reverse diffusion:

* Initialize `x_T ~ N(0, I)`
* Iterate `t = T−1 … 0`
* Compute DDPM posterior mean from ε-prediction
* Add noise at all steps except `t = 0`

---

### Generated Samples

Below are samples generated via ancestral DDPM sampling from pure noise using the current baseline configuration.

#### warm-up training (~5k steps)

![Samples at 5k steps](images/samples_step_5k.png)

---

### Training Dynamics

![Training loss curve](images/loss_curve.png)

The training loss decreases steadily, indicating stable ε-prediction optimization under the linear noise schedule.

If available, loss can also be analyzed by timestep bucket to assess whether learning is balanced across the diffusion horizon.

---

## Repository Structure

* `datasetLoaders.py` — CIFAR-10 dataloader and preprocessing
* `diffusion.py` — schedules, forward diffusion, training step, samplers
* `models.py` — U-Net, ResBlocks, attention, up/downsampling blocks
* `utils.py` — timestep embeddings, sample saving, visualization helpers
* `scripts.py` — training entry point, checkpointing, loss plotting

---

## Quickstart

### 1) Install dependencies

```bash
pip install torch torchvision einops tqdm matplotlib
```

---

### 2) Train

```bash
python scripts.py
```

This will:

* download CIFAR-10 into `./data`
* train for `NUM_TRAIN_STEPS` (default: 1000)
* save checkpoints to `saves/<exp_no>/`
* write a training loss plot `<exp_no>_loss_curve.png`

---

### Checkpoints

Saved at:

```
saves/<exp_no>/<step>_checkpoint.pt
```

Each checkpoint contains:

* `step`
* model `state_dict`
* optimizer `state_dict`
* loss history

---

## Iterative Upgrade Path

The guiding principle is to keep `model(x_t, t)` stable and make most upgrades modular.

### Phase 1: DDPM baseline hardening

* Exponential moving average (EMA) of model weights
* Improved logging (CSV / JSON)
* Deterministic seeding and reproducibility
* Periodic sample grids during training

### Phase 2: Objective variants

* v-prediction (Stable Diffusion style)
* x₀-prediction
* SNR-weighted losses

All introduced without redesigning the U-Net.

### Phase 3: Score-based modeling

* Interpret outputs as score estimates
* Introduce continuous-time (VE / VP SDE) formulations incrementally

### Phase 4: Better samplers

* DDIM
* Predictor–corrector methods
* DPM-Solver-style samplers

### Phase 5: Toward Stable Diffusion

* Classifier-free guidance
* Conditioning pathways (class → later text)
* Latent diffusion via an auxiliary autoencoder

---

## Contributing / Collaboration

Contributions are welcome.

### What fits well

* Small, focused PRs (one concept at a time)
* Clear validation plots or metrics
* Minimal changes to `models.py` unless necessary

### Good starter contributions

* EMA weights + EMA sampling
* Sample grid saving during training
* Resume-from-checkpoint support
* DDIM sampler
* Metrics logging utilities

Open an issue first if you’re unsure — happy to discuss direction.

---

## Notes

* Default training runs are intentionally short; do not expect high-quality samples yet.
* No EMA or DDIM enabled by default.
* FID/KID not yet integrated.

---

## Acknowledgements

Design choices follow common DDPM and U-Net best practices: timestep embeddings, residual blocks with GroupNorm, selective attention, and ancestral sampling.

The goal is not novelty, but **clarity, correctness, and extensibility**.

