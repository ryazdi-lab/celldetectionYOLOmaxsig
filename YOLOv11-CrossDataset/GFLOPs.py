# pip install ptflops
import os
import torch
from ultralytics import YOLO
from yolov11_maxsig_integration import modify_yolov11_backbone
from yolov11_maxsig_configs import create_cell_detection_config
import csv

try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None


IMG = int(os.environ.get('IMG', '640'))
VARIANT = os.environ.get('Y11_VARIANT', 'yolo11n')


def count_params_millions(core: torch.nn.Module) -> float:
    return sum(p.numel() for p in core.parameters()) / 1e6


def flops_ptflops(core: torch.nn.Module, h: int, w: int) -> float:
    if get_model_complexity_info is None:
        raise RuntimeError('ptflops not installed. pip install ptflops')
    macs, _ = get_model_complexity_info(
        core, (3, h, w),
        as_strings=False, print_per_layer_stat=False, verbose=False
    )
    return macs / 1e9


def flops_thop(core: torch.nn.Module, h: int, w: int) -> float:
    if thop_profile is None:
        raise RuntimeError('thop not installed. pip install thop')
    core.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, h, w)
        flops, _ = thop_profile(core, inputs=(x,), verbose=False)
    return flops / 1e9


def robust_flops(core: torch.nn.Module, h: int, w: int) -> float:
    # Try ptflops, fall back to thop
    try:
        return flops_ptflops(core, h, w)
    except Exception:
        return flops_thop(core, h, w)


def gflops_and_params(model_or_path) -> tuple[float, float]:
    model = model_or_path if isinstance(model_or_path, YOLO) else YOLO(model_or_path)
    core = model.model.eval()
    gflops = robust_flops(core, IMG, IMG)
    mparams = count_params_millions(core)
    return gflops, mparams


IMG = 640

# List of variants to evaluate
VARIANTS = ["n", "s", "m", "l", "x"]

def gflops_ptflops_core(yolo_obj: YOLO, h=640, w=640):
    core = yolo_obj.model.eval()
    macs, _ = get_model_complexity_info(
        core, (3, h, w),
        as_strings=False, print_per_layer_stat=False, verbose=False
    )
    params_m = sum(p.numel() for p in core.parameters()) / 1e6
    return macs / 1e9, params_m

results = []
cfg = create_cell_detection_config()['maxsig_config']

for variant in VARIANTS:
    ckpt_path = f"yolo11{variant}.pt"
    if not os.path.exists(ckpt_path):
        print(f"Skipping yolo11{variant}: checkpoint '{ckpt_path}' not found")
        continue


    # Baseline (robust FLOPs + manual params)
    baseline = YOLO(ckpt_path)
    b_g, b_p = gflops_and_params(baseline)

    # Enhanced (MaxSig-wrapped, robust FLOPs + manual params)
    enhanced = modify_yolov11_backbone(ckpt_path, cfg)
    e_g, e_p = gflops_and_params(enhanced)

    print(f"yolo11{variant} @ {IMG}: baseline {b_g:.2f} GFLOPs/{b_p:.2f} M, enhanced {e_g:.2f} GFLOPs/{e_p:.2f} M")