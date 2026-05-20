# GoodLiar 🤥

## Description

Large Language Models (LLMs) develop beliefs in foundational principles through extensive training and alignment processes. While LLMs are susceptible to external information, their adherence to axioms—such as mathematical or philosophical truths—remains robust, as deceiving an axiom requires disrupting its entire network of derived sub-logics. 

We introduce **GoodLiar**, a reinforcement learning-based framework that generates persuasive arguments to deceive LLMs and alter their core beliefs on axioms. It consists of two modules:

1. **Liar Agent**: Generates arguments to change an LLM’s belief.
2. **Reward Module**: Incentivizes successful deception.

Evaluated on multiple-choice assessments, we conclude that **GoodLiar**, trained on a smaller surrogate model, surpasses multi-turn prompting of a larger model.

## Schematic Overview

We formulate the task of generating effective deceptive contexts as a reinforcement learning (RL) problem, where an RL agent crafts arguments to alter an LLM’s belief. **GoodLiar** consists of two components: (1) **Liar Agent**, which generates deceptive contexts targeting specific axioms, and (2) **Reward Module**, which evaluates deception success and provides scalar rewards. While both modules use LLMs, only the Liar Agent is optimized through RL, while the Reward Module remains fixed.


<div align="center">
    <img src="diagram.jpg" alt="GoodLiar Diagram" width="70%">
</div>



## Installation

To set up and run **GoodLiar**, follow these steps:

1. **Create a Conda environment**  
   ```bash
   conda env create -f environment.yml
   ```

2. **Install `trlx`**  
   Follow the installation instructions from the official repository:  
   [https://github.com/CarperAI/trlx](https://github.com/CarperAI/trlx)

3. **Download pre-recorded datasets**  
   Download datasets containing lies on five axioms from [Google Drive](https://drive.google.com/drive/folders/1-2uvwAdrRVNAiNPP06p1M6nLBeNxhewr?usp=sharing) and save them in the `lie_dataset` folder.

## Training

Run the following command to train **GoodLiar**:

```bash
python main_liar.py <axiom_number> <gpu_id>
```

### Axiom Numbers

| Axiom Number | Axiom Description |
|-------------|------------------|
| 1 | **[Mathematical Axiom]** If A = B and B = C, then A = C |
| 2 | **[Mathematical Axiom]** For any sets A and B, there exists a set C that contains A and B |
| 3 | **[Mathematical Axiom]** If A < B and B < C, then A < C |
| 4 | **[Mathematical Axiom]** A + B = A + B and A × B = B × A |
| 5 | **[Philosophical Axiom]** Everything is identical to itself |

Happy experimenting with **GoodLiar** 🤥! 😃 🎉

## Mac Setup + Modal Reproduction Notes

### Goal

This repo was originally written to run GoodLiar training in a GPU/Linux environment. Since local Mac machines do not have CUDA GPUs, the current setup uses a Mac only as the control machine and runs the actual training on Modal with an A100 GPU.

---

## 1. Local Mac setup

### Create and activate a conda environment

```bash
conda create -n goodliar python=3.11 -y
conda activate goodliar
```

### Install local dependencies

```bash
pip install modal wandb huggingface_hub
```

You do not need to install the full CUDA/PyTorch training stack locally if you are using Modal. Modal builds the GPU image remotely.

---

## 2. Authenticate services

### Modal

```bash
modal token new
```

### Weights & Biases

```bash
wandb login
```

### Hugging Face

```bash
huggingface-cli login
```

---

## 3. Modal secrets

Create these Modal secrets before running:

```bash
modal secret create wandb-secret WANDB_API_KEY=your_wandb_key
modal secret create huggingface-secret HF_TOKEN=your_huggingface_token
```

The Modal app expects these names:

```python
modal.Secret.from_name("wandb-secret")
modal.Secret.from_name("huggingface-secret")
```

---

## 4. Modal volume

Create or reuse the checkpoint volume:

```bash
modal volume create goodliar-checkpoints
```

The code saves checkpoints to:

```bash
/checkpoints
```

This keeps model outputs after the Modal container exits.

---

## 5. Running training on Modal

From the repo root:

```bash
modal run modal_app.py
```

To run a specific axiom, call the function with an argument if supported by the Modal entrypoint:

```bash
modal run modal_app.py::train --axiom-idx 1
```

Axiom IDs:

```text
1 = If A=B and B=C then A=C
2 = For any sets A and B, there exists a set C that contains A and B
3 = If A<B and B<C then A<C
4 = A+B = A+B and AxB = BxA
5 = Everything is identical to itself
```

---

## What has been done so far

### Modal compatibility

We added a `modal_app.py` entrypoint that builds a CUDA image remotely, installs the required packages, mounts the repo into `/root/goodliar`, attaches an A100 GPU, and runs:

```bash
python main_liar.py <axiom_idx>
```

The Mac is only used to launch the Modal job.

### Argument handling

The original script expected both an axiom index and a GPU number. For Modal, GPU assignment is handled by Modal, so `main_liar.py` now defaults to axiom `1` if no argument is provided and no longer needs a local GPU index.

```python
arg1 = sys.argv[1] if len(sys.argv) > 1 else "1"
```

### Checkpoint persistence

Checkpoints now save to the Modal volume:

```python
save_path = f"/checkpoints/ckpts_liar_case_{ave_reward}_axiom_{arg1}"
```

This prevents outputs from disappearing when the Modal container shuts down.

### Logging

Extra `print(..., flush=True)` statements were added so Modal logs show progress during:

* epoch start
* reward scoring
* generation
* TRLX training
* checkpoint saving

This is important because generation and reward scoring can look stalled without frequent logs.

### Debugging changes currently active

These settings were added to make training more stable while debugging:

```python
# Smaller training batch size.
# This means TRLX processes fewer samples at once, which uses less GPU memory
# and makes optimization less likely to diverge while debugging.
# Original value was 20.
default_config['train']['batch_size'] = 4

# Lower learning rate.
# This controls how large each optimizer update is during training.
# A smaller value makes ILQL training less aggressive and helps avoid NaN losses.
default_config['optimizer']['kwargs']['lr'] = 1e-6

# Shorter eval generation length.
# This limits how many new tokens TRLX generates during evaluation.
# It makes eval faster and reduces the chance of generation-time instability.
default_config['method']['gen_kwargs']['max_new_tokens'] = 128
```

### Bug fixes found during Modal testing

The original training flow uses TRLX, which returns an ILQL wrapper model. That wrapper cannot be passed directly into `transformers.pipeline()` because it does not expose the same `.config` interface as a normal Hugging Face causal LM.

So the code keeps the original behavior:

```python
liar = trlx.train(...).model
model_liar = model
```

But saves the trained TRLX output:

```python
liar.save_pretrained(save_path)
```

This preserves the original epoch flow while still saving the trained checkpoint.

---

## Current debugging status

The code has successfully reached later epochs with reduced sample sizes.

The current debug mode uses smaller values such as:

```python
data = result_all["argu"][:20]
num_sample = 20
```

This proves the code path works, but it does not yet prove the full training scale is stable.

---

## What still needs to be done

### 1. Finish controlled scaling

Recommended scaling path:

```python
data = result_all["argu"][:100]
num_sample = 100
for epoch in range(3)
```

Then:

```python
data = result_all["argu"][:500]
num_sample = 500
for epoch in range(3)
```

Then full scale:

```python
data = result_all["argu"]
num_sample = 1000
for epoch in range(31)
```

### 2. Watch for NaNs

If logs show:

```text
losses/loss: nan
```

stop the run. That means training diverged.

The earlier crash happened because NaN losses caused invalid generation probabilities during evaluation.

### 3. Restore original training settings gradually

After medium-scale runs work, restore one thing at a time.

First:

```python
default_config['method']['gen_kwargs']['max_new_tokens'] = 1024
```

Then:

```python
default_config['train']['batch_size'] = 20
```

Finally, decide whether to remove or increase:

```python
default_config['optimizer']['kwargs']['lr'] = 1e-6
```

Do not restore all settings at once.

### 4. Save generated result files to the Modal volume

Generated `.pkl` result files should also save to `/checkpoints`:

```python
with open(f"/checkpoints/liar_result_case{arg1}_{epoch}.pkl", "wb") as f:
    pickle.dump(result_new, f)
```

Otherwise they may be lost when the Modal container exits.

---

## Reproducing GoodLiar locally

On a Mac, “local reproduction” means:

1. clone the repo
2. set up the conda environment
3. authenticate Modal, W&B, and Hugging Face
4. run the training job remotely through Modal

The full training itself should not be expected to run locally on a Mac because it requires CUDA GPU support.

`main_liar.py` includes the Modal-compatible argument handling, checkpoint path, debug config, and TRLX save/reset flow.
