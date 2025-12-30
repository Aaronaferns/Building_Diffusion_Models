# Diffusion From Scratch (CIFAR-10) — DDPM baseline → iterative modern upgrades

A small, readable diffusion repo that starts with a **vanilla DDPM-style ε-prediction model** and gradually moves toward more modern diffusion systems (score-based modeling, improved samplers, stability tricks, and eventually Stable-Diffusion-like components) **in small additive increments with minimal architectural disruption**.

This repo currently trains a **U-Net with timestep embeddings + selective attention** on **CIFAR-10 (32×32)** and performs ancestral sampling from pure noise. Core files include:

* CIFAR-10 dataloader with normalization to **[-1, 1]** 
* Forward diffusion + training step + ancestral sampling loop 
* U-Net backbone with ResBlocks, GroupNorm, attention at chosen resolutions, and timestep embeddings 
* Training script, checkpointing, and a basic loss curve plot 
* Sinusoidal timestep embedding utility 

---

## Why this repo exists

Most diffusion repos jump straight into big frameworks, large abstractions, or latent diffusion pipelines. This one is the opposite:

* **Start minimal** (DDPM baseline).
* **Instrument heavily** (metrics + plots + sample grids).
* **Add improvements iteratively** (each PR = one concept).
* **Avoid architectural churn** (keep the U-Net interface stable as long as possible).

If you want a “from scratch but not toy” stepping stone toward modern diffusion, this is it.

---

## Current Implementation (Baseline)

### Data

* CIFAR-10 training loader with:

  * random horizontal flip
  * `ToTensor()` then `Normalize((0.5,…),(0.5,…))` → **[-1,1]** 

### Diffusion process

* Linear beta schedule from `1e-4` to `0.02` over `T=1000` 
* Precomputed:

  * `alpha_t = 1 - beta_t`
  * `alpha_bar_t = ∏ alpha_t` 
* Training:

  * sample `t ~ Uniform({0..T-1})`
  * generate `x_t = sqrt(ᾱ_t)x_0 + sqrt(1-ᾱ_t) ε`
  * predict ε with U-Net and use MSE loss 

### Model: U-Net

* ResBlocks use:

  * GroupNorm → SiLU → Conv
  * timestep embedding projection added to hidden activations
  * dropout + second conv with **zero-init** (residual block starts near identity) 
* Attention blocks (single-head) at selected resolutions (`attn_res`) 
* Timestep embeddings: sinusoidal + 2-layer MLP

### Sampling

* Ancestral reverse diffusion:

  * `x_T ~ N(0,I)`
  * loop `t = T-1 … 0`
  * compute DDPM mean from ε-pred and add noise except at `t=0` 

---

## Repository Structure

* `datasetLoaders.py` — CIFAR-10 dataloader + preprocessing 
* `diffusion.py` — schedules, forward diffusion, train step, sampling 
* `models.py` — U-Net + ResBlock + Attention + Up/Down blocks 
* `utils.py` — sinusoidal timestep embeddings 
* `scripts.py` — training entry point, checkpointing, basic loss plotting 

---

## Quickstart

### 1) Install

```bash
pip install torch torchvision einops tqdm matplotlib
```

### 2) Train

```bash
python scripts.py
```

What happens:

* downloads CIFAR-10 into `./data` 
* trains for `NUM_TRAIN_STEPS` (currently set to 1000) 
* saves checkpoints into `saves/<exp_no>/` every `save_interval` steps 
* writes a loss curve plot `<exp_no>_loss_curve.png` 

### 3) Resume training (planned)

Not wired yet (but checkpoints already contain `step`, `model`, `optimizer`, and `losses`). 
A small upcoming improvement will add a `load_checkpoint()` helper and resume logic.

---

## Outputs and Artifacts

### Checkpoints

Saved at:

```
saves/<exp_no>/<step>_checkpoint.pt
```

Contains:

* `step`
* `model` state_dict
* `optimizer` state_dict
* `losses` list 

### Plots

* Training loss curve saved as `<exp_no>_loss_curve.png` 

---

## Metrics & Graphs Roadmap (what will be displayed)


### Training-time metrics

* **Loss vs step** 
* **EMA-smoothed loss**
* **Grad norm** 
* **Parameter norm / weight statistics**
* **Learning rate**
* **Noise-pred error by timestep bucket**
  e.g. average MSE at t∈[0..99], [100..199], … helps detect whether the model only learns early timesteps

### Sampling-time metrics

* **Sample grid every N steps**
* **Diversity metrics** (simple: pixel variance across batch)
* **FID/KID** (optional; more work, but common)
* **Inference speed** (ms per sample) & memory stats
* **Trajectory snapshots** (store `x_t` at a few t values to visualize denoising dynamics)

### Graphs

Planned figures:

* `loss_curve.png`
* `loss_by_timestep_bucket.png`
* `sample_grid_step_<k>.png`
* `denoise_trajectory.png` (show x_T → x_0 at selected t’s)

---

## Iterative Upgrade Path (minimal architectural changes)

The philosophy: keep `model(x_t, t)` stable as long as possible, and make diffusion upgrades mostly in `diffusion.py` / sampling utilities. 

### Phase 1: DDPM baseline hardening (small PRs)

1. **EMA of model weights** (often improves sample quality a lot)
2. **Better logging** (CSV/JSON + plots)
3. **Deterministic seeding** + reproducible runs
4. **Sample grids during training** (qualitative feedback loop)

### Phase 2: Objective variants (no U-Net redesign)

1. **v-prediction** (stable-diffusion style training target)
2. **x0-prediction** (predict x0 directly; similar interface)
3. **SNR weighting / loss reweighting**
   All of these can be introduced while preserving the same U-Net structure and only changing targets in `diffusionTrainStep`. 

### Phase 3: Score-based modeling (additive)

* Interpret outputs as **score** ∇x log p(x_t) or map from ε-pred to score depending on parameterization.
* Introduce continuous-time formulations incrementally (VE/VP SDE style), keeping U-Net mostly the same but changing time embedding scaling and sampler logic.

### Phase 4: Better samplers (mostly sampling code)

* DDIM sampling (fast, deterministic option)
* Predictor-corrector (Langevin steps)
* DPM-Solver-ish upgrades (later; more math but still modular)

All of this lives mostly next to `sample()` in `diffusion.py`. 

### Phase 5: “Stable diffusion direction” without big rewrites

We’ll do this carefully:

* keep U-Net API stable
* add features behind flags:

  * classifier-free guidance (CFG)
  * conditioning pathways (start with simple class conditioning for CIFAR-10, then text later)
  * eventually: latent diffusion (requires an autoencoder, but can still be isolated as a new module)

---

## Contributing / Collaboration

If you want to collaborate, you’re very welcome.

### What I’m looking for

* People who enjoy **clean incremental engineering** (one concept per PR).
* Folks comfortable with diffusion math OR willing to learn while implementing.

### Suggested first contributions (good starter PRs)

* Add **sample grid** saving during training (every `save_interval`)
* Add **EMA weights** + sample with EMA model
* Add a basic **metrics logger** (CSV + matplotlib plots)
* Add **DDIM sampler** option
* Add **resume-from-checkpoint** support

### How to collaborate

* Open an issue describing what you want to add and the smallest test/validation you’ll include.
* Keep PRs small and self-contained (one feature, one plot/test).
* Prefer minimal changes to `models.py` unless the feature truly needs it. 

---

## Notes / Known limitations (current state)

* Training steps are very low by default (`NUM_TRAIN_STEPS=1000`), so don’t expect great samples yet. 
* No EMA, no DDIM, no periodic sample saving (yet).
* No formal evaluation (FID/KID not integrated).

---

## Citation / Inspiration

Baseline design choices reflect common DDPM/U-Net best practices: timestep embeddings, ResBlocks, GroupNorm, selective attention at lower resolutions, and ancestral sampling.

