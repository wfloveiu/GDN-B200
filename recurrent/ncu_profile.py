"""
NCU profiling script for CUDA DeltaNet recurrent kernel.

Usage (under ncu):
    ncu --set full --kernel-name deltanet_recurrent_kernel \
        --launch-skip 10 --launch-count 1 \
        --csv --page raw \
        python ncu_profile.py [batch_size]

Or standalone (just runs the kernel, useful for ncu wrapping):
    python ncu_profile.py [batch_size]
"""
import sys
import torch
from CUDA_recurrent import kernel as cuda_kernel

NUM_KHEADS = 8
NUM_VHEADS = 16
DK = 128
DV = 128


def get_inputs(batch_size):
    B, T, H, K, V = batch_size, 1, NUM_KHEADS, DK, DV
    HV = NUM_VHEADS
    return dict(
        q=torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16),
        k=torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16),
        v=torch.randn(B, T, HV, V, device="cuda", dtype=torch.bfloat16),
        state=torch.zeros(B, HV, V, K, device="cuda", dtype=torch.float32),
        A_log=torch.randn(HV, device="cuda", dtype=torch.float32) * -1.0,
        a=torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16) * 0.1,
        dt_bias=torch.randn(HV, device="cuda", dtype=torch.float32) * 0.1,
        b=torch.randn(B, T, HV, device="cuda", dtype=torch.bfloat16),
        scale=1.0,
    )


def main():
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 128

    torch.manual_seed(42)
    inputs = get_inputs(batch_size)
    B, T, H, K = inputs["q"].shape
    HV = inputs["v"].shape[2]
    V = inputs["v"].shape[3]

    output = torch.empty(B, T, HV, V, device="cuda", dtype=torch.bfloat16)
    new_state = torch.empty(B, HV, V, K, device="cuda", dtype=torch.float32)

    # Warmup (ncu --launch-skip skips these)
    for _ in range(20):
        cuda_kernel(
            inputs["q"], inputs["k"], inputs["v"],
            inputs["state"].clone(),
            inputs["A_log"], inputs["a"], inputs["dt_bias"], inputs["b"],
            inputs["scale"],
            output, new_state,
        )
    torch.cuda.synchronize()

    # Profiled launch
    cuda_kernel(
        inputs["q"], inputs["k"], inputs["v"],
        inputs["state"].clone(),
        inputs["A_log"], inputs["a"], inputs["dt_bias"], inputs["b"],
        inputs["scale"],
        output, new_state,
    )
    torch.cuda.synchronize()

    print(f"[ncu_profile] Done. B={batch_size}, H={NUM_KHEADS}, HV={NUM_VHEADS}, K={DK}, V={DV}")


if __name__ == "__main__":
    main()
