import torch

# GPUが利用可能か確認
if torch.cuda.is_available():
    print("GPUは利用可能です")
    print(f"利用可能なGPUの数: {torch.cuda.device_count()}")
    print(
        f"利用中のGPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("GPUは利用できません")
