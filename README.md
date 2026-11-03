# CURE: Congruent and Certified Updates of Retinal Classifiers Across Evolving Label Sets

Code for the article *CURE: Congruent and Certified Updates of Retinal Classifiers Across Evolving Label Sets*
(S. Muhammad, F. Shang, D. M. Alsekait, D. S. Abdelminaam), submitted to *Knowledge-Based Systems*.

When a retinal classifier in use (the legacy model) is retrained on a new release of its dataset, the update can be
more accurate on average and still miss findings that the legacy model reported correctly. CURE is a post-hoc
procedure that sets the decision thresholds of any trained multi-label update so that the expected share of legacy
findings missed per image stays below a level α chosen by the user. It works in three steps:

1. **Congruence constraint.** Each shared-class threshold is the F1-maximizing value among those that keep every
   positive the legacy model detected.
2. **Legacy anchoring.** These thresholds are fitted on the validation split of the earlier release.
3. **Certified shift.** One common shift of all shared-class thresholds is calibrated with conformal risk control
   (CRC) on the validation fold of the new release, or with Learn then Test (LTT) for a high-probability bound.

CURE trains nothing and leaves the scores of the update unchanged. The implementation is in `src/crc.py`,
`src/certify.py` and `src/cv.py`.

## Repository layout

```
CURE/
├── src/
│   ├── config.py         # all paths; set RFMID_DATA_ROOT and RETFOUND_WEIGHTS to override
│   ├── data.py           # label and image reading, class matching across releases, manifests
│   ├── audit.py          # SHA-256 and perceptual-hash overlap audit, 256 px image cache
│   ├── cure.py           # network (DINOv2 + LoRA + ML-Decoder head, ASL), training, update baselines, metrics
│   ├── cv.py             # grouped stratified folds, congruence-constrained thresholds, pooled metrics, bootstrap
│   ├── crc.py            # conformal risk control of the missed-legacy-findings loss
│   ├── certify.py        # legacy-anchored shapes, Learn then Test, envelope statistics
│   ├── frontier.py       # trade-off curves of missed-disease flips against new false positives
│   ├── backbones.py      # RETFound backbone for the second setting
│   ├── baselines2025.py  # HCP (AAAI 2025), RCLP (WACV 2025), CRCL (MICCAI 2025), multi-label adaptations
│   ├── baselines2026.py  # MPT (AAAI 2026), RA-LDL (ICLR 2026), multi-label adaptations
│   ├── cat.py            # CAT, class-distribution-aware thresholds (Xie et al., TPAMI 2026)
│   ├── cure_v2.py        # exploratory training-time variant (notebooks 03 and 04)
│   └── opc.py            # exploratory threshold-aware training (notebook 04)
├── notebooks/            # run in the order below, from the notebooks folder
├── outputs/
│   ├── manifests/        # cleaned file lists and label space (notebook 01)
│   ├── cache/            # image fingerprints (the cached images themselves are not released)
│   └── results/          # audit removals, folds, summaries and tables
├── requirements.txt
└── README.md
```

## Data

The datasets are not redistributed. Download them from the original sources:

- **RFMiD 1.0:** IEEE DataPort, https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid
- **RFMiD 2.0:** Zenodo, https://zenodo.org/records/7505822 (CC BY 4.0)

Unpack them into `datasets/` (or point `RFMID_DATA_ROOT` to another folder) with this layout:

```
datasets/
├── RFMiD1_0/
│   ├── 1_Original_Images/{a_Training_Set, b_Validation_Set, c_Testing_Set}/
│   └── 2_Groundtruths/{a_RFMiD_Training_Labels, b_RFMiD_Validation_Labels, c_RFMiD_Testing_Labels}.csv
└── RFMiD2_0/
    ├── {Training_set, Validation_set, Test_set}/
    └── {RFMiD_2_Training_labels, RFMiD_2_Validation_labels, RFMiD_2_Testing_labels}.csv
```

## Pretrained weights

- **DINOv2 ViT-B/14** (`vit_base_patch14_dinov2.lvd142m`) is downloaded automatically by `timm`.
- **RETFound** (`RETFound_mae_natureCFP.pth`, second setting only) is distributed by its authors through
  https://github.com/rmaphoh/RETFound_MAE. Place it in `weights/` or set `RETFOUND_WEIGHTS`.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux and macOS: source .venv/bin/activate
pip install torch torchvision          # choose the CUDA build for your system at https://pytorch.org
pip install -r requirements.txt
```

All experiments in the article ran on one NVIDIA RTX 4060 with 8 GB of memory.

## Reproducing the article

Every notebook resumes where it stopped and has a `QUICK` switch for a short smoke test. Set `QUICK = False`
for the article numbers (three seeds, five folds).

| Order | Notebook | What it produces | Article |
|---|---|---|---|
| 1 | `01_read_datasets` | class matching (38 shared, 7 discarded, 12 new), integrity report, manifests | Sec. 3.3 |
| 2 | `02_cure_pipeline` | overlap audit, image fingerprints, legacy model for each seed | Sec. 3.3, 3.4, Fig. 3(a) |
| 3 | `04_cv_opc` | grouped stratified five folds (`folds.csv`) | Sec. 3.3, Fig. 3(c), (d) |
| 4 | `07_affordable_certificate_two_updates` | FT, WiSE-FT, OE, anchored and certified thresholds, CRC and LTT, second setting with RETFound | Tables 5, 6 |
| 5 | `08_confirm_and_main_tables` | FD, ablations, legacy rows, main summaries | Table 5 |
| 6 | `09_cure_main_table` | KD, Joint, MPT, RA-LDL, CURE on every update | Table 6 |
| 7 | `10_cure_2025_baselines` | HCP, RCLP, CRCL, the full comparison table, envelope gaps, paired bootstrap | Table 6, Sec. 4.3 to 4.5 |
| 8 | `11_article_figures` | trade-off frontier, CURE on every update, per-class view, Figs. S1 and S2 | Figs. 7, 8, 10, S1 to S3 |
| 9 | `12_article_figures_update` | source of the regression, validity of the guarantee, examples | Figs. 6, 9, 11 |
| 10 | `13_cat_baseline` | CAT and CAT-T, per-class and overall F1 | Table 5, Table S1 |

Notebooks 11 to 13 train nothing and read the saved predictions of notebooks 07 to 10.

**Exploratory notebooks.** `03_cure_v2`, `05_frontier_check` and `06_ct_plugin_certified` record earlier steps
of the study, including the training-time alternatives mentioned in the Discussion of the article. No number in
the article depends on them.

The method diagrams (Figs. 1 to 5) were drawn separately and are not produced by the code.

## Released artefacts

So that the cleaned splits can be rebuilt from the original releases without any image being shared:

- `outputs/manifests/`: retained image identifiers of every split and the matched label space
- `outputs/cache/fingerprints_rfmid1.csv`, `fingerprints_rfmid2.csv`: SHA-256 digests and 64-bit perceptual hashes
- `outputs/results/audit_removed_rfmid2.csv`: every image removed by the overlap audit, with the reason
- `outputs/results/cv/folds.csv`: group and fold of each of the 748 retained RFMiD 2.0 images

## Citation

```bibtex
@article{muhammad2026cure,
  title   = {{CURE}: Congruent and Certified Updates of Retinal Classifiers Across Evolving Label Sets},
  author  = {Muhammad, Sajjad and Shang, Fengjun and Alsekait, Deema Mohammed and Abdelminaam, Diaa Salama},
  journal = {Knowledge-Based Systems},
  note    = {Under review},
  year    = {2026}
}
```

## Contact

Sajjad Muhammad, sajjad.iiui2016@gmail.com
