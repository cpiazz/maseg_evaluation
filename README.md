# maseg Evaluation

This repository contains an evaluation pipeline for testing **maseg** on mammography images.

The pipeline currently supports:

- **CBIS-DDSM**
- **CMMD**

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

## 6. DICOM reader compatibility

The original maseg DICOM reader expects:

```python
dcm_image.ImagerPixelSpacing
```

Some DICOM images may not contain this field.

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

## 7. Supported datasets

### CBIS-DDSM

The **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)** collection is available from **The Cancer Imaging Archive (TCIA)**.

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

#### Download using the TCIA Data Retriever

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

#### Supporting CSV files

The collection also provides case-description CSV files such as:

- `mass_case_description_train_set.csv`
- `mass_case_description_test_set.csv`
- `calc_case_description
