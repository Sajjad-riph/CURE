"""Project paths and dataset layout.

Edit DATA_ROOT here if the datasets move. Everything else is derived from it.
Set the environment variable RFMID_DATA_ROOT to override without editing.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
# Datasets live in the paper-3 folder; the implementation lives in paper-1.
DATA_ROOT = Path(os.environ.get(
    "RFMID_DATA_ROOT",
    r"C:\Users\Sajjad\Documents\Documents\paper-3\datasets",
))

# paper-1 root = parent of this src/ folder, so it works from any working dir.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MANIFEST_DIR = OUTPUT_DIR / "manifests"
FIGURE_DIR = OUTPUT_DIR / "figures"

SPLITS = ("train", "val", "test")

# ---------------------------------------------------------------------------
# RFMiD 1.0
#   images : <DATA_ROOT>/RFMiD1_0/1_Original_Images/<split folder>
#   labels : <DATA_ROOT>/RFMiD1_0/2_Groundtruths/<label file>
# ---------------------------------------------------------------------------
RFMID1_ROOT = DATA_ROOT / "RFMiD1_0"
RFMID1 = {
    "name": "RFMiD1.0",
    "images": {
        "train": RFMID1_ROOT / "1_Original_Images" / "a_Training_Set",
        "val":   RFMID1_ROOT / "1_Original_Images" / "b_Validation_Set",
        "test":  RFMID1_ROOT / "1_Original_Images" / "c_Testing_Set",
    },
    # File names as on disk; the loader adds .csv when the extension is omitted.
    "labels": {
        "train": RFMID1_ROOT / "2_Groundtruths" / "a_RFMiD_Training_Labels",
        "val":   RFMID1_ROOT / "2_Groundtruths" / "b_RFMiD_Validation_Labels",
        "test":  RFMID1_ROOT / "2_Groundtruths" / "c_RFMiD_Testing_Labels",
    },
}

# ---------------------------------------------------------------------------
# RFMiD 2.0
#   images : <DATA_ROOT>/RFMiD2_0/<split folder>
#   labels : <DATA_ROOT>/RFMiD2_0/<label file>
# ---------------------------------------------------------------------------
RFMID2_ROOT = DATA_ROOT / "RFMiD2_0"
RFMID2 = {
    "name": "RFMiD2.0",
    "images": {
        "train": RFMID2_ROOT / "Training_set",
        "val":   RFMID2_ROOT / "Validation_set",
        "test":  RFMID2_ROOT / "Test_set",
    },
    "labels": {
        "train": RFMID2_ROOT / "RFMiD_2_Training_labels",
        "val":   RFMID2_ROOT / "RFMiD_2_Validation_labels",
        "test":  RFMID2_ROOT / "RFMiD_2_Testing_labels",
    },
}

DATASETS = {"rfmid1": RFMID1, "rfmid2": RFMID2}

# Columns that are summary flags, not disease labels.
# Disease_Risk (RFMiD 1.0) = any disease; WNL (RFMiD 2.0) = within normal limits.
NON_DISEASE_COLUMNS = {"DISEASE_RISK", "WNL", "NORMAL"}

# ---------------------------------------------------------------------------
# Matching class names across releases (checked against both dataset papers:
# Pachade et al., Data 2021 and Panchal et al., Data 2023)
# ---------------------------------------------------------------------------
# Same disease, different spelling: RFMiD 2.0 column -> RFMiD 1.0 name.
LABEL_ALIASES = {
    "OPDM": "ODPM",   # optic disc pit maculopathy (2.0 CSV header misspelt)
}
# Same abbreviation, different disease: never treat as shared.
LABEL_COLLISIONS = {
    "MCA",            # RFMiD 1.0 = macroaneurysm, RFMiD 2.0 = microaneurysm
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
LABEL_EXTENSIONS = (".csv",)   # all RFMiD label tables are CSV


def ensure_output_dirs():
    for d in (OUTPUT_DIR, MANIFEST_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)
