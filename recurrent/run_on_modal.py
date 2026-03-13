import subprocess
from pathlib import Path
import modal

APP_NAME = "deltanet-recurrent-b200"
PROJECT_ROOT = Path(__file__).resolve().parent

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("gcc", "g++")
    .run_commands(
        "pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128",
    )
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements.txt"))
    .add_local_dir(
        str(PROJECT_ROOT),
        remote_path="/root/recurrent",
        copy=True,
    )
    .workdir("/root/recurrent")
)


def _run_cmd(cmd: list[str]):
    print(f"\n[Modal] Running: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


@app.function(image=image, gpu="B200", timeout=60 * 60)
def run_fla_recurrent():
    """FLA gated delta-rule recurrent kernel benchmark."""
    _run_cmd(["python", "FLA_recurrent.py"])


@app.function(image=image, gpu="B200", timeout=60 * 60)
def run_recurrent():
    """Custom Triton DeltaNet recurrent kernel (tiled v2) validation."""
    _run_cmd(["python", "recurrent.py"])


@app.function(image=image, gpu="B200", timeout=60 * 60)
def run_cuda():
    """CUDA kernel vs Triton kernel comparison."""
    _run_cmd(["python", "deltanet_recurrent_cuda.py"])


@app.function(image=image, gpu="B200", timeout=60 * 60)
def run_all():
    """Run all three kernels sequentially."""
    _run_cmd(["python", "FLA_recurrent.py"])
    _run_cmd(["python", "recurrent.py"])
    _run_cmd(["python", "deltanet_recurrent_cuda.py"])


@app.local_entrypoint()
def main(which: str = "all"):
    dispatch = {
        "fla": run_fla_recurrent,
        "recurrent": run_recurrent,
        "cuda": run_cuda,
        "all": run_all,
    }
    fn = dispatch.get(which)
    if fn is None:
        raise ValueError(
            f"Unknown option: {which}. Choose from {' / '.join(dispatch)}"
        )
    fn.remote()
