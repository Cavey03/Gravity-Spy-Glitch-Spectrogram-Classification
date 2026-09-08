"""
config.py — Single source of truth for the whole project.

Every other module imports values from here. If you want to change a
hyperparameter, a path, or the seed, change it in THIS file only.

Design rule: this file only *declares* values. It contains no training
logic, no I/O, no side effects — importing it must be instant and safe.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the "gravity-spy/" folder (the directory this file lives in).
# We derive everything else relative to it so the repo is portable — no
# absolute "C:/Users/..." paths that only work on one machine.
PROJECT_ROOT = Path(__file__).resolve().parent

# >>> THE ONE LINE YOU EDIT TO POINT AT YOUR DATA <<<
# Put your downloaded trainingsetv1d1.h5 here (or change this path).
DATA_ROOT = PROJECT_ROOT / "dataset"
H5_PATH = DATA_ROOT / "trainingsetv1d1.h5"

# Alternative raw-PNG layout (only used if LOADER == "png"). Folder layout is
# expected to be  <PNG_ROOT>/<label>/<sample_type>/<gravityspy_id>_<dur>.png
PNG_ROOT = DATA_ROOT / "trainingsetv1d1"

# The metadata CSV with columns: gravityspy_id, label, sample_type.
# (The .h5 file also carries these fields internally; see data/dataset.py.
#  This path is a fallback / for the PNG loader.)
METADATA_CSV = DATA_ROOT / "trainingset_v1d1_metadata.csv"

# Where all run artifacts go. Each model gets outputs/<model_name>/.
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"

# ---------------------------------------------------------------------------
# Data source selection
# ---------------------------------------------------------------------------
# "h5"  -> load pre-cropped 1x140x170 arrays from the .h5 file (preferred).
# "png" -> load raw PNGs and crop axes with [66:532, 105:671, :3].
LOADER = "h5"

# Which duration-view to use. The Gravity Spy set has 4 views per glitch:
# 0.5, 1.0, 2.0, 4.0 seconds. We use 2.0s by default (good time/freq balance).
# Change this to experiment; the dataset resolves the right key/file per view.
DURATION = 2.0
VALID_DURATIONS = (0.5, 1.0, 2.0, 4.0)

# ---------------------------------------------------------------------------
# Classes  (CANONICAL ORDER — do not reorder)
# ---------------------------------------------------------------------------
# The model outputs an integer 0..21. This list is the ONLY thing that maps
# that integer to a human-readable class. Every model and every metric uses
# this exact ordering. index == class id.
CLASSES = [
    "1080Lines",
    "1400Ripples",
    "Air_Compressor",
    "Blip",
    "Chirp",
    "Extremely_Loud",
    "Helix",
    "Koi_Fish",
    "Light_Modulation",
    "Low_Frequency_Burst",
    "Low_Frequency_Lines",
    "No_Glitch",
    "None_of_the_Above",
    "Paired_Doves",
    "Power_Line",
    "Repeating_Blips",
    "Scattered_Light",
    "Scratchy",
    "Tomte",
    "Violin_Mode",
    "Wandering_Line",
    "Whistle",
]
NUM_CLASSES = len(CLASSES)  # derived, == 22

# Convenience lookups used across the codebase.
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASSES)}

# ---------------------------------------------------------------------------
# Image geometry
# ---------------------------------------------------------------------------
# Pre-cropped Gravity Spy arrays are 140 (H) x 170 (W). We resize to a square
# that ImageNet-pretrained backbones are happy with. Grey images get replicated
# to 3 channels (done in transforms), so backbones expecting RGB still work.
IMG_SIZE = 224
IN_CHANNELS = 3  # after grey -> 3-channel replication

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# A single global seed applied to torch / numpy / random, plus cudnn
# deterministic mode. See utils/seed.py. Changing this changes every run.
SEED = 42

# ---------------------------------------------------------------------------
# Training hyperparameters (defaults; overridable via train.py CLI flags)
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
NUM_WORKERS = 0          # DataLoader worker processes. Default 0 for Windows
                         # safety (spawn-based multiprocessing can hang here).
                         # On Linux/GPU set higher via `--num-workers 4` for
                         # speed; train.py also auto-retries 0 if spawning errors.
EPOCHS = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

# Early stopping: stop if the monitored metric hasn't improved for this many
# epochs. We monitor validation macro-F1 (the headline metric for imbalance).
EARLY_STOPPING_PATIENCE = 7
MONITOR_METRIC = "val_macro_f1"  # what train.py watches for "best" checkpoint

# Class imbalance handling. When True, train.py builds inverse-frequency class
# weights from the TRAIN split and passes them to CrossEntropyLoss.
USE_CLASS_WEIGHTS = True

# Augmentation master switch (train split only). Individual augmentation
# strengths live in data/transforms.py.
USE_AUGMENTATION = True

# Horizontal flip = time reversal. OFF by default (a time-reversed glitch is
# usually not a physically valid example). Set True only if you decide it is
# acceptable for your experiment. Vertical flip / rotation are NEVER allowed.
USE_HFLIP = False

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
# Resolved lazily so importing config never requires torch to be installed
# for non-training tools (e.g. inspect_data can run without a GPU).
def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
