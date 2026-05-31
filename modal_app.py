import modal

app = modal.App("goodliar-train")

volume = modal.Volume.from_name("goodliar-checkpoints", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "DS_BUILD_OPS": "0",
        "CUDA_LAUNCH_BLOCKING": "1",
    })
    .pip_install(
        "torch==2.5.1",
        "transformers==4.34.0",
        "datasets==2.14.0",
        "accelerate==0.24.0",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "matplotlib==3.9.2",
        "pyyaml==6.0.2",
        "tqdm==4.66.6",
        "einops==0.8.0",
        "openai==0.28.0",
        "wandb==0.18.5",
        "huggingface_hub==0.17.3",
        "deepspeed==0.15.3",
    )
    .pip_install("git+https://github.com/CarperAI/trlx.git")
    .add_local_dir(".", remote_path="/root/goodliar")
)

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 6,
    volumes={"/checkpoints": volume},
    secrets=[
        modal.Secret.from_name("wandb"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train(
    axiom_idx: str = "1",
    max_turns: int = 5,
    aggregation: str = "max",
    num_samples: int = 20,
    epochs: int = 31,
    rollout_only: bool = False,
    no_early_stop: bool = False,
):
    import subprocess

    cmd = [
        "python", "main_liar.py", axiom_idx,
        "--max-turns", str(max_turns),
        "--aggregation", aggregation,
        "--num-samples", str(num_samples),
        "--epochs", str(epochs),
    ]
    if rollout_only:
        cmd.append("--rollout-only")
    if no_early_stop:
        cmd.append("--no-early-stop")

    subprocess.run(cmd, cwd="/root/goodliar", check=True)