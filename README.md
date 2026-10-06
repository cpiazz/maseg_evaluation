# maseg Evaluation

This repository contains an evaluation pipeline for testing **maseg** on mammography images, currently using the **CBIS-DDSM** dataset.

`maseg` predicts three anatomical classes:

- `0` — background
- `1` — breast
- `2` — pectoral muscle

The pipeline contains two main stages:

1. **Batch inference** — run maseg on eligible mammograms and save the predicted segmentation masks.
2. **Qualitative review** — inspect the generated segmentations in a Streamlit interface and save structured reviewer feedback to CSV.

> This project evaluates breast/background/pectoral segmentation. It does **not** evaluate lesion segmentation.

---

## Repository structure

A suggested repository structure is:

```text
maseg-evaluation/
├── main.py
├── config.example.json
├── requirements.txt
├── README.md
├── .gitignore
│
├── scripts/
│   ├── __init__.py
│   ├── run_maseg_batch.py
│   └── review_maseg_streamlit.py
│
└── outputs/
    └── .gitkeep
```

This repository does **not** need to contain the original maseg repository or its pretrained checkpoint.

Users should clone maseg separately, retrieve the model checkpoint from the upstream repository, and then reference the local maseg installation through `config.json`.

---

## 1. Clone this evaluation repository

```bash
git clone <URL_TO_THIS_REPOSITORY>
cd maseg-evaluation
```

---

## 2. Create the Python environment

A dedicated Conda environment is recommended.

```bash
conda create -n maseg_env python=3.10 -y
conda activate maseg_env
pip install -r requirements.txt
```

If using Spyder with this environment:

```bash
pip install "spyder-kernels==3.0.*"
```

Then configure Spyder to use the Python interpreter from `maseg_env`.

---

## 3. Clone the original maseg repository

Clone the original maseg project separately:

```bash
git clone https://github.com/radboud-axti/maseg.git
cd maseg
```

The evaluation pipeline imports `RunModel` from this local maseg repository.

---

## 4. Retrieve `segmentation_weights.ckpt`

The pretrained maseg checkpoint is:

```text
segmentation_weights.ckpt
```

The checkpoint is stored in the original maseg repository using **Git LFS**.

If Git LFS is not installed:

```bash
sudo apt install git-lfs
git lfs install
```

Then, from inside the cloned maseg repository:

```bash
git lfs pull
```

Check that the checkpoint was downloaded correctly:

```bash
ls -lh segmentation_weights.ckpt
```

A correctly downloaded checkpoint should be approximately:

```text
111 MB
```

rather than a very small text file.

You can also inspect the file type:

```bash
file segmentation_weights.ckpt
```

A valid modern PyTorch checkpoint may be reported as a ZIP archive. This is expected; the usable filename remains:

```text
segmentation_weights.ckpt
```

### Git LFS placeholder problem

If Git LFS did not retrieve the real checkpoint, the file may contain text similar to:

```text
version https://git-lfs.github.com/spec/v1
oid sha256:...
size ...
```

Trying to load this placeholder can result in an error such as:

```text
UnpicklingError: invalid load key, 'v'
```

Running:

```bash
git lfs pull
```

from inside the maseg repository should replace the placeholder with the real checkpoint.

---

## 5. PyTorch compatibility

During this evaluation, PyTorch 2.6+ caused checkpoint-loading compatibility problems because the default behaviour of `torch.load()` changed to:

```python
weights_only=True
```

A working configuration used for the evaluation was:

```text
torch==2.5.1
torchvision==0.20.1
```

These versions should be included in `requirements.txt`.

If a trusted checkpoint is loaded manually, another possible approach is to explicitly use:

```python
weights_only=False
```

but this evaluation used PyTorch 2.5.1 for compatibility.

---

## 6. DICOM reader modification

The original maseg DICOM reader expects:

```python
dcm_image.ImagerPixelSpacing
```

Some CBIS-DDSM images do not contain this field.

This can result in an error such as:

```text
AttributeError: 'FileDataset' object has no attribute 'ImagerPixelSpacing'
```

The relevant maseg file is:

```text
maseg/utils/readers.py
```

The spacing logic can be modified to check alternative DICOM fields:

```python
if hasattr(dcm_image, "ImagerPixelSpacing"):
    spacing_2d = dcm_image.ImagerPixelSpacing

elif hasattr(dcm_image, "PixelSpacing"):
    spacing_2d = dcm_image.PixelSpacing

elif hasattr(dcm_image, "NominalScannedPixelSpacing"):
    spacing_2d = dcm_image.NominalScannedPixelSpacing

else:
    raise ValueError(
        f"No pixel spacing information found in DICOM: {filename}"
    )

spacing = np.array(spacing_2d, dtype=float)
spacing = tuple(np.append(spacing, 1.0))
```

If this repository includes a patch file:

```text
patches/maseg_readers.patch
```

it can be applied with:

```bash
cd /path/to/maseg
git apply /path/to/maseg-evaluation/patches/maseg_readers.patch
```

### Important limitation

Some CBIS-DDSM DICOMs used in this evaluation contain none of:

- `ImagerPixelSpacing`
- `PixelSpacing`
- `NominalScannedPixelSpacing`

For these images, the evaluation pipeline reads the mammogram with `pydicom` and passes the pixel array to maseg as a NumPy array.

This bypasses maseg's physical-spacing-based DICOM resampling and should therefore be documented as a dataset/model compatibility limitation.

---

## 7. Download the CBIS-DDSM dataset

The evaluation currently uses the **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)** collection from **The Cancer Imaging Archive (TCIA)**.

Official collection page:

```text
https://www.cancerimagingarchive.net/collection/cbis-ddsm/
```

CBIS-DDSM contains DICOM mammography data including:

- full mammograms;
- cropped lesion images;
- ROI masks;
- supporting case-description CSV files.

For the current maseg evaluation, the **full mammogram images** are the main images required.

The ROI masks in CBIS-DDSM describe lesions rather than whole-breast/background anatomy, so they cannot be used directly as a gold standard for maseg's breast/background/pectoral segmentation.

### Download using the TCIA Data Retriever

From the CBIS-DDSM collection page:

1. Select the full collection or the required subset.
2. Download the corresponding `.tcia` manifest file.
3. Open the manifest using the TCIA Data Retriever.
4. Select the destination directory.
5. Start the download.

Useful subsets include:

- Mass-Training full mammogram images
- Mass-Training ROI and cropped images
- Calc-Training full mammogram images
- Calc-Training ROI and cropped images
- Mass-Test full mammogram images
- Mass-Test ROI and cropped images
- Calc-Test full mammogram images
- Calc-Test ROI and cropped images

The complete collection is large, so downloading only the required subsets may be more practical.

### Supporting CSV files

The collection also provides case-description CSV files such as:

- `mass_case_description_train_set.csv`
- `mass_case_description_test_set.csv`
- `calc_case_description_train_set.csv`
- `calc_case_description_test_set.csv`

The evaluation pipeline also uses a TCIA metadata CSV to identify DICOM series. Its local path is specified using:

```json
"METADATA_CSV": "/path/to/metadata.csv"
```

The root directory containing the downloaded DICOM data is specified using:

```json
"DATA_ROOT": "/path/to/cbis_ddsm"
```

### Dataset DOI

CBIS-DDSM DOI:

```text
10.7937/K9/TCIA.2016.7O02S9CY
```

When publishing results based on the dataset, follow the citation and data-usage guidance provided on the official TCIA collection page.

---

## 8. Configuration file

All paths and main processing options are controlled through:

```text
config.json
```

This keeps machine-specific paths and run settings outside the Python scripts.

For a public repository, commit:

```text
config.example.json
```

but do **not** commit your local:

```text
config.json
```

### Example `config.json`

```json
{
  "METADATA_CSV": "/shared/maseg_evaluation/CBIS_DDSM/metadata/metadata.csv",
  "DATA_ROOT": "/shared/maseg_evaluation/CBIS_DDSM/cbis_ddsm",
  "MASEG_ROOT": "/shared/maseg_evaluation/maseg",
  "WEIGHTS_PATH": "/shared/maseg_evaluation/maseg/segmentation_weights.ckpt",
  "OUTPUT_DIR": "/shared/maseg_evaluation/maseg_outputs",

  "EVALUATION_OUTPUT_DIR": "/shared/maseg_evaluation/evaluations",
  "EVALUATION_LABEL": "review_1",

  "RUN_BATCH": true,
  "LIMIT": 50,
  "MLO_ONLY": true,
  "MAX_DIM": 2048
}
```

### Configuration fields

| Key | Description |
|---|---|
| `METADATA_CSV` | Path to the TCIA/CBIS-DDSM metadata CSV used to identify mammograms. |
| `DATA_ROOT` | Root directory containing the downloaded CBIS-DDSM DICOM folders. |
| `MASEG_ROOT` | Path to the locally cloned maseg repository. |
| `WEIGHTS_PATH` | Full path to `segmentation_weights.ckpt`. |
| `OUTPUT_DIR` | Directory where generated `.npy` segmentations and `processing_log.csv` are stored. |
| `EVALUATION_OUTPUT_DIR` | Directory where qualitative review CSV files are saved. |
| `EVALUATION_LABEL` | Default reviewer/evaluation label shown in the Streamlit interface. |
| `RUN_BATCH` | If `true`, run batch inference before opening Streamlit. If `false`, skip inference and review existing outputs. |
| `LIMIT` | Maximum number of eligible mammograms selected for batch processing. Use `null` for all eligible images. |
| `MLO_ONLY` | If `true`, restrict processing and review to MLO mammograms. |
| `MAX_DIM` | Maximum input image dimension passed to maseg before inference. |

### Create the local configuration

Copy the example:

```bash
cp config.example.json config.json
```

Then edit the paths and processing options for your environment.

Add the local configuration file to `.gitignore`:

```gitignore
config.json
```

---

## 9. Run the evaluation pipeline

The project should normally be started through:

```bash
conda activate maseg_env
python main.py
```

`main.py` is the single entry point for the workflow.

It:

1. loads `config.json`;
2. validates the configured paths and values;
3. creates output directories if required;
4. runs maseg batch inference when `RUN_BATCH` is `true`;
5. passes the relevant configuration to the Streamlit process;
6. launches the qualitative review interface.

The scripts inside `scripts/` are therefore components of the pipeline and normally do not need to be launched directly.

---

## 10. Batch inference

Batch inference is implemented in:

```text
scripts/run_maseg_batch.py
```

The module exposes:

```python
run_batch(config)
```

and is called by `main.py`.

### Example: process 50 MLO mammograms

```json
"RUN_BATCH": true,
"LIMIT": 50,
"MLO_ONLY": true,
"MAX_DIM": 2048
```

### Process all eligible mammograms

Use:

```json
"LIMIT": null
```

### Skip batch processing

If segmentation outputs already exist and only the review interface is needed:

```json
"RUN_BATCH": false
```

The batch script:

- loads CBIS-DDSM metadata;
- identifies original full mammograms;
- includes recognised originals whose `SeriesDescription` is blank;
- optionally restricts processing to MLO views;
- removes duplicate mammograms;
- applies the configured `LIMIT`;
- loads maseg once;
- runs inference on each selected mammogram;
- downsamples large images according to `MAX_DIM`;
- restores the categorical segmentation to the original image dimensions using nearest-neighbour interpolation;
- saves one `.npy` segmentation per mammogram;
- skips segmentations that already exist;
- saves and continuously updates `processing_log.csv`;
- catches CUDA out-of-memory and other per-case errors so processing can continue.

Example output:

```text
maseg_outputs/
├── P_00001_LEFT_MLO_maseg.npy
├── P_00001_RIGHT_MLO_maseg.npy
├── ...
└── processing_log.csv
```

### Meaning of `LIMIT`

`LIMIT` applies to the batch selection, not to the Streamlit reviewer.

For example:

```json
"LIMIT": 50
```

selects the first 50 eligible mammograms for that batch run.

If some of those cases have already been processed, they are skipped because their output `.npy` file already exists.

---

## 11. GPU memory limitation

Native-resolution CBIS-DDSM mammograms can be approximately:

```text
4500 × 3000 pixels
```

Running maseg directly at this resolution exceeded approximately **16 GB of GPU memory** during this evaluation.

The workaround used by the batch pipeline is:

1. downsample the mammogram before inference;
2. run maseg on the resized image;
3. resize the categorical segmentation back to the original dimensions using nearest-neighbour interpolation.

The initial setting used was:

```json
"MAX_DIM": 2048
```

If necessary, this can be reduced, for example:

```json
"MAX_DIM": 1536
```

or:

```json
"MAX_DIM": 1024
```

Because downsampling may affect segmentation performance, this preprocessing step should be reported when documenting the evaluation methodology.

---

## 12. Streamlit qualitative review

The review interface is implemented in:

```text
scripts/review_maseg_streamlit.py
```

It is normally launched automatically by:

```bash
python main.py
```

The Streamlit application receives its configuration from `main.py`; users do not need to re-enter dataset and output paths manually.

### Which cases are shown?

The number of cases available for review is determined automatically by the segmentation files already present in:

```text
OUTPUT_DIR
```

Specifically, the review application looks for files named like:

```text
P_00038_LEFT_MLO_maseg.npy
```

Therefore, the Streamlit interface does **not** use `LIMIT`.

For example, if 50 matching segmentation files exist in `OUTPUT_DIR`, those processed cases are available for review.

### Reviewer label

The default label comes from:

```json
"EVALUATION_LABEL": "review_1"
```

The reviewer can change the label from the Streamlit sidebar.

The label is used to create the evaluation CSV filename, for example:

```text
maseg_evaluation_Concetta.csv
```

or:

```text
maseg_evaluation_reviewer_A.csv
```

### Review interface

The reviewer can inspect:

- the original mammogram;
- the breast segmentation;
- the pectoral muscle segmentation;
- the raw segmentation mask;
- the combined breast + pectoral foreground mask.

The interface also shows:

- number of processed cases;
- number reviewed;
- number remaining;
- review progress;
- option to display only unreviewed cases.

The reviewer records:

- overall segmentation quality;
- breast boundary quality;
- background leakage;
- missing breast tissue;
- pectoral muscle segmentation;
- non-breast artefacts;
- overall acceptability;
- free-text comments.

If an image has already been reviewed using the same evaluation label, saving it again replaces the previous review for that image rather than creating a duplicate row.

---

## 13. Evaluation approach

CBIS-DDSM provides lesion annotations, but it does **not** provide a whole-breast/background ground-truth mask suitable for direct quantitative validation of maseg.

Therefore, the current breast/background evaluation is primarily:

```text
qualitative / visual
```

### Overall segmentation quality

- Good
- Minor errors
- Poor
- Unable to assess

### Breast boundary

- Correct
- Minor error
- Major error
- Unable to assess

### Background leakage

- None
- Minor
- Substantial
- Unable to assess

### Missing breast tissue

- None
- Minor
- Substantial
- Unable to assess

### Pectoral muscle segmentation

- Correct
- Minor error
- Major error
- Unable to assess

### Non-breast artefacts

- None
- Minor
- Substantial
- Unable to assess

### Acceptable segmentation

- Yes
- No
- Uncertain

---

## 14. Known setup and compatibility issues

### PyTorch checkpoint compatibility

PyTorch 2.6+ may fail to load the maseg checkpoint because of the `weights_only=True` default.

A working setup used:

```text
torch==2.5.1
torchvision==0.20.1
```

---

### Missing OpenCV dependency

`cv2` is required by the evaluation pipeline.

Install:

```bash
pip install opencv-python-headless
```

For a desktop environment, `opencv-python` may also be used.

---

### Missing SimpleITK dependency

SimpleITK is required by maseg's image reader.

Install:

```bash
pip install SimpleITK
```

---

### No complete upstream dependency specification

The maseg setup used for this evaluation did not provide a complete dependency specification sufficient to reproduce the working environment directly.

This evaluation repository therefore provides its own `requirements.txt`.

For exact environment reproduction, an additional lock file can be created after confirming the environment works:

```bash
pip freeze > requirements-lock.txt
```

---

### DICOM pixel-spacing compatibility

maseg expects DICOM pixel-spacing information.

Some CBIS-DDSM images contain none of:

```text
ImagerPixelSpacing
PixelSpacing
NominalScannedPixelSpacing
```

For these cases, the evaluation pipeline reads the image with `pydicom` and supplies the image array to maseg directly.

---

### High GPU memory requirement

Full-resolution mammograms may exceed available GPU memory.

The evaluation pipeline therefore supports image downsampling before inference and nearest-neighbour resizing of the categorical prediction back to the original dimensions.

---

## 15. Data and generated files

CBIS-DDSM is **not included** in this repository.

Users must obtain the dataset separately and specify its location in `config.json`.

Do not commit:

- DICOM images;
- downloaded CBIS-DDSM dataset folders;
- generated `.npy` segmentations;
- local evaluation CSVs;
- `config.json`;
- maseg model checkpoints unless redistribution is explicitly permitted.

---

## 16. Suggested `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/

# Environments
.venv/
venv/
env/

# Spyder
.spyproject/
.spyderproject

# Local configuration
config.json

# Data
CBIS_DDSM/
data/
*.dcm

# Generated maseg outputs
maseg_outputs/
*.npy
processing_log.csv

# Evaluation outputs
evaluations/
maseg_evaluation_*.csv

# Model weights
*.ckpt

# Editors / OS
.vscode/
.idea/
.DS_Store
```

If an empty `outputs/` directory is kept in Git with `.gitkeep`, use:

```gitignore
outputs/*
!outputs/.gitkeep
```

---

## 17. Reproducibility

The recommended workflow is:

```text
config.json
     │
     ▼
   main.py
     │
     ├── run_batch(config)
     │       │
     │       └── scripts/run_maseg_batch.py
     │
     └── Streamlit process
             │
             └── scripts/review_maseg_streamlit.py
```

`config.json` is the single source of configuration for the evaluation.

The batch script receives the configuration dictionary directly from `main.py`.

The Streamlit application runs in a separate process, so `main.py` passes the relevant configuration values through environment variables.

The Streamlit reviewer determines its available cases from the segmentation files actually present in `OUTPUT_DIR`, rather than from `LIMIT`.

For exact package versions after establishing a working environment:

```bash
pip freeze > requirements-lock.txt
```

---

## 18. Attribution

This repository contains evaluation code built around the **maseg** mammography segmentation model.

Original maseg repository:

```text
https://github.com/radboud-axti/maseg
```

The original maseg source code and pretrained checkpoint remain part of the upstream project.

Any local modifications used for this evaluation, particularly changes related to DICOM reading and pixel-spacing handling, should be documented clearly.
