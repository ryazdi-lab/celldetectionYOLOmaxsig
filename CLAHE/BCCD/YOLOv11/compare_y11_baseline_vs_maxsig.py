import os
from collections import Counter

import torch
from ultralytics import YOLO

from yolov11_maxsig_configs import create_cell_detection_config
from yolov11_maxsig_integration import modify_yolov11_backbone


def print_header(title: str) -> None:
	print("\n" + "=" * 80)
	print(title)
	print("=" * 80)


def summarize_ultralytics_model(yolo_obj: YOLO, tag: str) -> None:
	"""Print a compact summary: module types, parameter counts, and model.info()."""
	core = getattr(yolo_obj, 'model', None)
	if core is None:
		print(f"[{tag}] No core model found.")
		return

	# Parameter stats
	params = sum(p.numel() for p in core.parameters())
	trainable = sum(p.numel() for p in core.parameters() if p.requires_grad)

	# Module histogram
	mods = [type(m).__name__ for m in core.modules()]
	cnts = Counter(mods)
	maxsig_present = any('MaxSig' in m for m in cnts)
	maxsig_kinds = [m for m in sorted(cnts) if 'MaxSig' in m]

	print_header(f"{tag} :: High-level Summary")
	print(f"Parameters (total): {params:,}")
	print(f"Parameters (trainable): {trainable:,}")
	print(f"Distinct module types: {len(cnts)}")
	print(f"Top-20 module counts: {cnts.most_common(20)}")
	print(f"MaxSig present: {maxsig_present} -> {maxsig_kinds}")

	# Ultralytics built-in model info (safe)
	try:
		print_header(f"{tag} :: Ultralytics model.info()")
		yolo_obj.info(verbose=True)
	except Exception as e:
		print(f"[info()] Failed with: {e}")

	# PyTorch summary fallback (print module tree head/tail)
	try:
		print_header(f"{tag} :: torch.nn.Module tree (head)")
		text = str(core)
		print("\n".join(text.splitlines()[:60]))
		print("... (truncated) ...")
	except Exception:
		pass


def main():
	variant = os.environ.get('Y11_VARIANT', 'yolo11n')
	base_weights = f"{variant}.pt"

	# 1) Baseline YOLOv11
	print_header("Loading baseline YOLOv11 model")
	baseline = YOLO(base_weights)
	summarize_ultralytics_model(baseline, tag=f"Baseline ({variant})")

	# 2) Enhanced YOLOv11 (wrapped with MaxSig)
	print_header("Wrapping YOLOv11 with MaxSig (C2PSA/C3k2 + optional neck C2f)")
	cfg = create_cell_detection_config()['maxsig_config']
	# Ensure wrapping is enabled by default
	cfg['wrap_backbone_c2fpsa'] = max(1, int(cfg.get('wrap_backbone_c2fpsa', 0)))
	cfg['wrap_backbone_c3k2'] = max(1, int(cfg.get('wrap_backbone_c3k2', 0)))
	cfg['wrap_neck_c2f'] = max(1, int(cfg.get('wrap_neck_c2f', 0)))

	enhanced = modify_yolov11_backbone(base_weights, cfg)
	summarize_ultralytics_model(enhanced, tag=f"Enhanced+MaxSig ({variant})")

	print_header("Done")


if __name__ == '__main__':
	main()


