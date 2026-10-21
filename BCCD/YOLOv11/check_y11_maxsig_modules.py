import os
from collections import Counter
from ultralytics import YOLO

WEIGHTS_PATH = r'cell_detection/yolo11n_maxsig/weights/best.pt'

print('exists:', os.path.exists(WEIGHTS_PATH))
if not os.path.exists(WEIGHTS_PATH):
	raise SystemExit(0)

model = YOLO(WEIGHTS_PATH)
core = getattr(model, 'model', None)
mods = [m.__class__.__name__ for m in (core.modules() if core else [])]
counts = Counter(mods)

print('found_MaxSigLayernormalizedAdvanced:', 'MaxSigLayernormalizedAdvanced' in counts)
print('found_MaxSigC2fPSA:', 'MaxSigC2fPSA' in counts)
print('found_MaxSigC3k2:', 'MaxSigC3k2' in counts)
print('matches:', [n for n in counts if 'MaxSig' in n])
