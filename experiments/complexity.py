import torch
from fvcore.nn import FlopCountAnalysis
import time
from tabulate import tabulate
from timm.models import create_model
import torch.amp
from tqdm import tqdm
import argparse
from utils.fvcore import jits
import octic_vits.deit_models
import deit.vit

WARMUP_ITERATIONS = 10
NUM_ITERATIONS = 100
BATCH_SIZE = 64

headers = ["Model Name", "Params (10⁶)", "Throughput (im/s)", "FLOPS (10⁹)", "Peak Mem (MB)"]

_MODEL_NAMES = [
    # DeiT III
    "deit_huge_patch14_LS",
    "d8_inv_early_deit_huge_patch14",
    "hybrid_deit_huge_patch14",

    "deit_large_patch16_LS",
    "d8_inv_early_deit_large_patch16",
    "hybrid_deit_large_patch16",
]

def compute_peak_mem(model, batch_size=8, device='cuda', amp=True):
    torch.cuda.reset_peak_memory_stats()
    img = torch.randn(batch_size, 3, 224, 224, device=device)
    with torch.amp.autocast(device_type='cuda', enabled=amp):
        model(img)
    peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)  # in MB
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    return peak_memory

def compute_throughput(model, img):
    timing = []
    for _ in range(WARMUP_ITERATIONS):
        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
            model(img)

    torch.cuda.synchronize()
    for _ in tqdm(range(NUM_ITERATIONS)):
        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
            start = time.time()
            model(img)
            torch.cuda.synchronize()
            timing.append(time.time() - start)

    timing = torch.as_tensor(timing, dtype=torch.float32)
    
    return timing.mean()


@torch.no_grad()
def compute_complexity(model_name, args):
    torch.cuda.empty_cache()

    device = 'cuda'

    model = create_model(model_name, pretrained=False )
    model.eval()
    model.to(device)

    B = BATCH_SIZE
    img = torch.randn(B, 3, 224, 224, device=device, requires_grad=False)

    params = sum(p.numel() for p in model.parameters())

    # flops = FlopCountAnalysis(model, (img))
    # jits(flops)
    # flops_per_image = flops.total() / B / 1e9  # in GFLOPs
    
    flops_per_image = 0
    if args.compile:
        model = torch.compile(model) # , mode='max-autotune')
        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
            y = model(img)

    peak_memory = compute_peak_mem(model, batch_size=B)
    imgs_per_sec = B/compute_throughput(model, img)
    # imgs_per_sec = 0
    # peak_memory = 0
    
    return [model_name, f"{params/1e6:.4f}", f"{imgs_per_sec:.0f}", f"{flops_per_image:.1f}", f"{peak_memory:.0f}"]


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description='Compute model complexity')
    parser.add_argument('--amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    args = parser.parse_args()

    data = []
    for model_name in tqdm(_MODEL_NAMES, desc='Computing model complexity'):
        data.append(compute_complexity(model_name, args))

    print(tabulate(data, headers=headers, tablefmt="grid"))