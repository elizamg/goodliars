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
        "transformers==4.46.1",
        "datasets==3.1.0",
        "accelerate==1.1.0",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "matplotlib==3.9.2",
        "pyyaml==6.0.2",
        "tqdm==4.66.6",
        "einops==0.8.0",
        "openai==0.28.0",
        "wandb==0.18.5",
        "huggingface_hub==0.26.2",
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
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train(axiom_idx: str = "1"):
    import subprocess

    subprocess.run(
        ["python", "main_liar.py", axiom_idx],
        cwd="/root/goodliar",
        check=True,
    )