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


def _run_cmd_capture(cmd: list[str]) -> str:
    """Run command and return stdout+stderr as string."""
    print(f"\n[Modal] Running: {' '.join(cmd)}\n", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    print(output, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}")
    return output


@app.function(image=image, gpu="B200", timeout=60 * 60)
def run_bench():
    _run_cmd(["python", "bench.py"])


@app.function(image=image, gpu="B200", timeout=60 * 60)
def run_ncu(batch_size: int = 128):
    """Profile CUDA V2 kernel with NCU and print detailed analysis."""
    bs = str(batch_size)

    # --- Collect key metrics via ncu CSV mode ---
    metrics = ",".join([
        # Memory throughput
        "dram__bytes.sum",                            # Total DRAM bytes
        "dram__bytes_read.sum",                       # DRAM bytes read
        "dram__bytes_write.sum",                      # DRAM bytes written
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",  # DRAM BW utilization %
        # L2 cache
        "lts__t_bytes.sum",                           # L2 total bytes
        "lts__t_bytes_lookup_hit.sum",                # L2 hit bytes
        "lts__t_bytes_lookup_miss.sum",               # L2 miss bytes
        # L1/shared memory
        "l1tex__t_bytes.sum",                         # L1 total bytes
        # Compute
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",  # SM utilization %
        "sm__warps_active.avg.pct_of_peak_sustained_elapsed",  # Warp occupancy %
        # Instruction
        "smsp__inst_executed.sum",                    # Instructions executed
        "smsp__cycles_active.avg",                    # Active cycles
        # Launch info
        "launch__grid_size",                          # Grid size
        "launch__block_size",                         # Block size
        "launch__registers_per_thread",               # Registers per thread
        "launch__shared_mem_per_block_allocated",     # Shared mem per block
        # Duration
        "gpu__time_duration.sum",                     # Kernel duration (ns)
    ])

    ncu_cmd = [
        "ncu",
        "--kernel-name", "deltanet_recurrent_v3_kernel",
        "--launch-skip", "20",
        "--launch-count", "1",
        "--metrics", metrics,
        "--csv",
        "python", "ncu_profile.py", bs,
    ]

    csv_output = _run_cmd_capture(ncu_cmd)

    # --- Parse CSV and print formatted results ---
    print("\n" + "=" * 72)
    print(f"  NCU Profiling Report: CUDA V3 DeltaNet Kernel (B={batch_size})")
    print("=" * 72)

    import csv
    import io

    # ncu CSV output may have header lines starting with "=="
    csv_lines = [
        line for line in csv_output.splitlines()
        if line and not line.startswith("==") and not line.startswith("[ncu_profile]")
    ]
    csv_text = "\n".join(csv_lines)

    if csv_text.strip():
        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)

        # Group metrics for display
        sections = {
            "Memory Throughput": [
                ("dram__bytes.sum", "Total DRAM Bytes"),
                ("dram__bytes_read.sum", "DRAM Bytes Read"),
                ("dram__bytes_write.sum", "DRAM Bytes Written"),
                ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM BW Utilization"),
            ],
            "Cache": [
                ("lts__t_bytes.sum", "L2 Total Bytes"),
                ("lts__t_bytes_lookup_hit.sum", "L2 Hit Bytes"),
                ("lts__t_bytes_lookup_miss.sum", "L2 Miss Bytes"),
                ("l1tex__t_bytes.sum", "L1 Total Bytes"),
            ],
            "Compute": [
                ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "SM Utilization"),
                ("sm__warps_active.avg.pct_of_peak_sustained_elapsed", "Warp Occupancy"),
                ("smsp__inst_executed.sum", "Instructions Executed"),
                ("smsp__cycles_active.avg", "Active Cycles"),
            ],
            "Launch Config": [
                ("launch__grid_size", "Grid Size"),
                ("launch__block_size", "Block Size"),
                ("launch__registers_per_thread", "Registers/Thread"),
                ("launch__shared_mem_per_block_allocated", "Shared Mem/Block"),
            ],
            "Timing": [
                ("gpu__time_duration.sum", "Kernel Duration"),
            ],
        }

        # ncu CSV: each row is one metric. Key columns: "Metric Name", "Metric Value", "Metric Unit"
        metric_map = {}
        for row in rows:
            name = row.get("Metric Name", "")
            value = row.get("Metric Value", "")
            unit = row.get("Metric Unit", "")
            if name:
                metric_map[name] = (value, unit)

        for section_name, metric_list in sections.items():
            print(f"\n--- {section_name} ---")
            for metric_key, display_name in metric_list:
                if metric_key in metric_map:
                    val, unit = metric_map[metric_key]
                    # Format large numbers
                    try:
                        fval = float(val.replace(",", ""))
                        if unit in ("byte", "bytes") and fval > 1024 * 1024:
                            print(f"  {display_name:>30}: {fval / (1024*1024):>12.2f} MB")
                        elif unit in ("byte", "bytes") and fval > 1024:
                            print(f"  {display_name:>30}: {fval / 1024:>12.2f} KB")
                        elif unit == "nsecond" or unit == "ns":
                            print(f"  {display_name:>30}: {fval / 1000:>12.2f} us")
                        elif unit == "%":
                            print(f"  {display_name:>30}: {fval:>12.2f} %")
                        else:
                            print(f"  {display_name:>30}: {val:>12} {unit}")
                    except ValueError:
                        print(f"  {display_name:>30}: {val:>12} {unit}")
                else:
                    print(f"  {display_name:>30}: (not available)")

        # --- Derived analysis ---
        print(f"\n--- Derived Analysis ---")
        try:
            dram_bytes = float(metric_map.get("dram__bytes.sum", ("0",""))[0].replace(",", ""))
            duration_ns = float(metric_map.get("gpu__time_duration.sum", ("1",""))[0].replace(",", ""))
            dram_read = float(metric_map.get("dram__bytes_read.sum", ("0",""))[0].replace(",", ""))
            dram_write = float(metric_map.get("dram__bytes_write.sum", ("0",""))[0].replace(",", ""))

            duration_us = duration_ns / 1000
            bw_tb_s = dram_bytes / duration_ns  # bytes/ns = GB/s, * 1000 = TB/s... no.
            # bytes / ns = GB/s
            bw_gb_s = dram_bytes / duration_ns
            bw_tb_s = bw_gb_s / 1000

            # Theoretical minimum bytes: state read + write = B * HV * V * K * 4 * 2
            # Plus q, k, v, gates (tiny)
            state_bytes = batch_size * 8 * 128 * 128 * 4 * 2
            overhead_ratio = dram_bytes / state_bytes if state_bytes > 0 else 0

            print(f"  {'Achieved BW':>30}: {bw_gb_s:>12.2f} GB/s ({bw_tb_s:.2f} TB/s)")
            print(f"  {'Kernel Duration':>30}: {duration_us:>12.2f} us")
            print(f"  {'DRAM Read':>30}: {dram_read / (1024*1024):>12.2f} MB")
            print(f"  {'DRAM Write':>30}: {dram_write / (1024*1024):>12.2f} MB")
            print(f"  {'Theoretical Min State Bytes':>30}: {state_bytes / (1024*1024):>12.2f} MB")
            print(f"  {'Actual / Theoretical Ratio':>30}: {overhead_ratio:>12.2f}x")
            print(f"  {'Read/Write Ratio':>30}: {dram_read/dram_write if dram_write > 0 else 0:>12.2f}")
        except Exception as e:
            print(f"  (Could not compute derived metrics: {e})")

    else:
        print("  (No CSV metrics captured — check ncu output above)")

    print("\n" + "=" * 72)


@app.function(image=image, gpu="B200", timeout=60 * 60)
def run_all():
    """Run all kernels sequentially."""
    _run_cmd(["python", "bench.py"])


@app.local_entrypoint()
def main(which: str = "all", batch_size: int = 128):
    dispatch = {
        "bench": run_bench,
        "ncu": run_ncu,
        "all": run_all,
    }
    fn = dispatch.get(which)
    if fn is None:
        raise ValueError(
            f"Unknown option: {which}. Choose from {' / '.join(dispatch)}"
        )
    if which == "ncu":
        fn.remote(batch_size=batch_size)
    else:
        fn.remote()
