modal run run_on_modal.py                   # 默认运行全部三个

modal run run_on_modal.py --which fla       # 只运行 FLA 基准

modal run run_on_modal.py --which recurrent # 只运行 Triton 核验证

modal run run_on_modal.py --which cuda      # 只运行 CUDA vs Triton 对比

modal run run_on_modal.py --which all       # 全部运行