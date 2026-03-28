import subprocess
from pathlib import Path
import modal

APP_NAME = "deltanet-chunked-b200"
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
        remote_path="/root/chunked",
        copy=True,
    )
    .workdir("/root/chunked")
)


def _run_cmd(cmd: list[str]):
    print(f"\n[Modal] Running: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


@app.function(image=image, gpu="B200", timeout=60 * 60)
def run_bench():
    _run_cmd(["rm", "-rf", "/root/.triton/cache", "/tmp/triton_cache"])
    _run_cmd(["python", "bench.py"])


@app.function(image=image, gpu="B200", timeout=60 * 60)
def run_all():
    """Run all kernels sequentially."""
    _run_cmd(["python", "bench.py"])


@app.function(image=image, gpu="B200", timeout=60 * 60)
def run_ncu(
    kernel_filter: str = "",
    ncu_set: str = "detailed",
    num_seqs: int = 4,
    seq_len: int = 8192,
):
    """
    Run ncu profiling on chunk_gated_delta_rule.

    Args:
        kernel_filter: regex to filter kernel names (e.g. "chunk_gated_delta_rule_fwd_kernel_h.*").
                       Empty string = profile all kernels.
        ncu_set:       ncu metric set — "basic", "detailed", or "full".
        num_seqs:      number of sequences.
        seq_len:       sequence length per sequence.
    """
    cmd = [
        "ncu",
        "--set", ncu_set,
        "--profile-from-start", "off",   # skip autotune + warmup
        "--target-processes", "all",
        "--replay-mode", "kernel",
        "--page", "details",
        "--csv",
    ]
    if kernel_filter:
        cmd += ["--kernel-name", kernel_filter]

    cmd += [
        "python", "ncu_profile.py",
        str(num_seqs), str(seq_len),
    ]
    _run_cmd(cmd)


@app.local_entrypoint()
def main(which: str = "all"):
    dispatch = {
        "bench": run_bench,
        "all": run_all,
        "ncu": run_ncu,
    }
    fn = dispatch.get(which)
    if fn is None:
        raise ValueError(
            f"Unknown option: {which}. Choose from {' / '.join(dispatch)}"
        )
    fn.remote()
