#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Oct  6 12:51:58 2026

@author: c_piazzese
"""

"""
Batch maseg evaluation on mammography datasets.

This module is designed to be called from main.py.

Supported datasets
------------------
- CBIS_DDSM
- CMMD

The dataset is selected through DATASET_NAME in config.json.

Each dataset loader returns a common table containing:

    image_key
    PatientID
    image_path

This allows the same maseg inference pipeline to be used
for different datasets.

Expected config keys
--------------------
Required:
    DATASET_NAME
    DATA_ROOT
    MASEG_ROOT
    WEIGHTS_PATH
    OUTPUT_DIR

Dataset-specific:
    METADATA_CSV
        Required for CBIS_DDSM.

Optional:
    MLO_ONLY   -> default True
                  Used for CBIS_DDSM only.
    MAX_DIM    -> default 2048
    LIMIT      -> default None
"""


# ============================================================
# IMPORTS
# ============================================================

import os
import re
import glob
import sys
import gc

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch


# ============================================================
# GENERIC DICOM FUNCTIONS
# ============================================================

def load_dicom(path):

    """
    Load a DICOM image using pydicom.
    """

    ds = pydicom.dcmread(
        path
    )


    image = (
        ds.pixel_array
        .astype(
            np.float32
        )
    )


    return ds, image


# ============================================================
# PREPROCESS MAMMOGRAM
# ============================================================

def preprocess_mammogram(
    ds,
    image
):

    """
    Basic preprocessing used before passing the
    mammogram to maseg.

    Steps
    -----
    - convert to float32
    - invert MONOCHROME1 images
    - normalize to [0, 1]
    """

    image = image.astype(
        np.float32
    )


    photometric = getattr(
        ds,
        "PhotometricInterpretation",
        ""
    )


    # --------------------------------------------------------
    # Invert MONOCHROME1
    # --------------------------------------------------------

    if photometric == "MONOCHROME1":

        image = (
            image.max()
            -
            image
        )


    # --------------------------------------------------------
    # Normalize to [0, 1]
    # --------------------------------------------------------

    image = (
        image
        -
        image.min()
    )


    maximum = image.max()


    if maximum > 0:

        image = (
            image
            /
            maximum
        )


    return image


# ============================================================
# RESIZE IMAGE
# ============================================================

def resize_for_maseg(
    image,
    max_dim
):

    """
    Resize a mammogram so that its largest dimension
    does not exceed max_dim.

    Aspect ratio is preserved.

    If the image is already small enough, it is returned
    unchanged.
    """

    h, w = image.shape


    scale = min(
        max_dim / h,
        max_dim / w,
        1.0
    )


    new_h = int(
        round(
            h * scale
        )
    )


    new_w = int(
        round(
            w * scale
        )
    )


    if scale == 1.0:

        return image, scale


    resized = cv2.resize(
        image,
        (
            new_w,
            new_h
        ),
        interpolation=cv2.INTER_AREA
    )


    return resized, scale


# ============================================================
# RUN MASEG
# ============================================================

def run_maseg(
    model,
    image,
    max_dim
):

    """
    Run maseg inference.

    Large mammograms are downsampled before inference.

    The resulting categorical segmentation is resized
    back to the original image size using nearest-neighbour
    interpolation.
    """

    original_shape = image.shape


    image_small, scale = resize_for_maseg(
        image,
        max_dim
    )


    print(
        "Original:",
        original_shape,
        "| maseg input:",
        image_small.shape,
        "| scale:",
        round(
            scale,
            4
        )
    )


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    with torch.no_grad():

        segmentation_small = model.get_segmentation(
            image_small,
            fill_holes_in_breast=False,
            output_size=image_small.shape
        )


    segmentation_small = np.asarray(
        segmentation_small
    )


    segmentation = cv2.resize(
        segmentation_small.astype(
            np.uint8
        ),
        (
            original_shape[1],
            original_shape[0]
        ),
        interpolation=cv2.INTER_NEAREST
    )


    return segmentation, scale


# ============================================================
# CBIS-DDSM
# ============================================================

def get_cbis_image_key(patient_id):

    """
    Extract a mammogram key such as:

        P_00038_LEFT_MLO

    from strings such as:

        P_00038_LEFT_MLO.dcm

    or:

        Calc-Test_P_00038_LEFT_MLO_1
    """

    name = str(
        patient_id
    )


    match = re.search(
        r"(P_\d+_(?:LEFT|RIGHT)_(?:CC|MLO))",
        name
    )


    if match:

        return match.group(1)


    return None


# ============================================================

def find_cbis_dicom(
    patient_id,
    data_root
):

    """
    Find DICOM files underneath the directory
    corresponding to the supplied PatientID.

    The first matching DICOM is returned, preserving
    the behaviour of the original CBIS-DDSM pipeline.
    """

    patient_dir = os.path.join(
        data_root,
        str(patient_id)
    )


    if not os.path.exists(
        patient_dir
    ):

        return None


    files = sorted(
        glob.glob(
            os.path.join(
                patient_dir,
                "**",
                "*.dcm"
            ),
            recursive=True
        )
    )


    if len(files) == 0:

        return None


    return files[0]


# ============================================================

def load_cbis_ddsm(config):

    """
    Load eligible CBIS-DDSM mammograms.

    Returns a DataFrame containing:

        image_key
        PatientID
        image_path
    """

    METADATA_CSV = config[
        "METADATA_CSV"
    ]


    DATA_ROOT = config[
        "DATA_ROOT"
    ]


    MLO_ONLY = config.get(
        "MLO_ONLY",
        True
    )


    # ========================================================
    # LOAD METADATA
    # ========================================================

    df = pd.read_csv(
        METADATA_CSV
    )


    print(
        "\nTotal metadata rows:",
        len(df)
    )


    print(
        "\nSeries descriptions:"
    )


    print(
        df[
            "SeriesDescription"
        ]
        .value_counts(
            dropna=False
        )
    )


    # ========================================================
    # IMAGE KEYS
    # ========================================================

    df[
        "image_key"
    ] = (
        df[
            "PatientID"
        ]
        .apply(
            get_cbis_image_key
        )
    )


    # ========================================================
    # IDENTIFY FULL MAMMOGRAMS
    # ========================================================

    full_images = df[
        df[
            "SeriesDescription"
        ]
        .fillna("")
        .str.lower()
        .eq(
            "full mammogram images"
        )
    ].copy()


    # --------------------------------------------------------
    # Include original mammograms where description is blank
    # --------------------------------------------------------

    original_pattern = (
        r"^P_\d+_"
        r"(LEFT|RIGHT)_"
        r"(CC|MLO)"
        r"(\.dcm)?$"
    )


    missing_description_originals = df[
        df[
            "SeriesDescription"
        ].isna()
        &
        df[
            "PatientID"
        ].str.match(
            original_pattern,
            na=False
        )
    ].copy()


    full_images = pd.concat(
        [
            full_images,
            missing_description_originals
        ],
        ignore_index=True
    )


    # --------------------------------------------------------
    # Remove duplicate series
    # --------------------------------------------------------

    if (
        "SeriesInstanceUID"
        in full_images.columns
    ):

        full_images = (
            full_images
            .drop_duplicates(
                subset=[
                    "SeriesInstanceUID"
                ]
            )
        )


    # ========================================================
    # FILTER BY VIEW
    # ========================================================

    if MLO_ONLY:

        full_images = full_images[
            full_images[
                "image_key"
            ]
            .str.endswith(
                "_MLO",
                na=False
            )
        ].copy()


    # ========================================================
    # ONE ROW PER MAMMOGRAM
    # ========================================================

    images = (
        full_images[
            [
                "image_key",
                "PatientID"
            ]
        ]
        .dropna(
            subset=[
                "image_key"
            ]
        )
        .drop_duplicates(
            subset=[
                "image_key"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # FIND DICOM PATH
    # ========================================================

    images[
        "image_path"
    ] = images[
        "PatientID"
    ].apply(
        lambda patient_id:
        find_cbis_dicom(
            patient_id,
            DATA_ROOT
        )
    )


    missing_dicom = (
        images[
            "image_path"
        ]
        .isna()
        .sum()
    )


    if missing_dicom > 0:

        print(
            "\nCBIS-DDSM entries with no DICOM found:",
            missing_dicom
        )


    images = (
        images[
            images[
                "image_path"
            ].notna()
        ]
        .reset_index(
            drop=True
        )
    )


    print(
        "\nEligible CBIS-DDSM mammograms:",
        len(images)
    )


    return images


# ============================================================
# CMMD
# ============================================================

def load_cmmd(config):

    """
    Load all DICOM images from the CMMD dataset.

    Example structure
    -----------------

    DATA_ROOT/
        D1-0001/
            2010-07-18-79377/
                1-70244/
                    image1.dcm
                    image2.dcm

        D1-0002/
            ...

    Every DICOM is treated as a separate image.

    No filtering by MLO, CC, laterality or other
    mammography-view metadata is performed.

    Returns a DataFrame containing:

        image_key
        PatientID
        image_path
    """

    DATA_ROOT = config[
        "DATA_ROOT"
    ]


    records = []


    # ========================================================
    # FIND PATIENT DIRECTORIES
    # ========================================================

    patient_dirs = sorted(
        [
            path
            for path in glob.glob(
                os.path.join(
                    DATA_ROOT,
                    "*"
                )
            )
            if os.path.isdir(
                path
            )
        ]
    )


    print(
        "\nCMMD patient folders found:",
        len(patient_dirs)
    )


    # ========================================================
    # FIND ALL DICOMS
    # ========================================================

    for patient_dir in patient_dirs:

        patient_id = os.path.basename(
            patient_dir
        )


        dicom_files = sorted(
            glob.glob(
                os.path.join(
                    patient_dir,
                    "**",
                    "*.dcm"
                ),
                recursive=True
            )
        )


        if len(
            dicom_files
        ) == 0:

            print(
                "No DICOM files found for:",
                patient_id
            )

            continue


        print(
            patient_id,
            "- DICOM images:",
            len(dicom_files)
        )


        # ----------------------------------------------------
        # Every DICOM is a separate image
        # ----------------------------------------------------

        for image_index, image_path in enumerate(
            dicom_files,
            start=1
        ):

            image_key = (
                f"{patient_id}_"
                f"{image_index:02d}"
            )


            records.append(
                {
                    "image_key":
                        image_key,

                    "PatientID":
                        patient_id,

                    "image_path":
                        image_path
                }
            )


    # ========================================================
    # CREATE TABLE
    # ========================================================

    images = pd.DataFrame(
        records,
        columns=[
            "image_key",
            "PatientID",
            "image_path"
        ]
    )


    print(
        "\nTotal CMMD DICOM images:",
        len(images)
    )


    if len(
        images
    ) > 0:

        print(
            "Total CMMD patients:",
            images[
                "PatientID"
            ].nunique()
        )

    else:

        print(
            "Total CMMD patients: 0"
        )


    return images


# ============================================================
# DATASET DISPATCHER
# ============================================================

def load_dataset(config):

    """
    Load the dataset specified by DATASET_NAME.

    All loaders return a DataFrame with:

        image_key
        PatientID
        image_path
    """

    dataset_name = str(
        config[
            "DATASET_NAME"
        ]
    ).upper()


    print(
        "\n========================================"
    )

    print(
        "Loading dataset:",
        dataset_name
    )

    print(
        "========================================"
    )


    if dataset_name == "CBIS_DDSM":

        return load_cbis_ddsm(
            config
        )


    elif dataset_name == "CMMD":

        return load_cmmd(
            config
        )


    else:

        raise ValueError(
            f"Unsupported dataset: "
            f"{dataset_name}"
        )


# ============================================================
# MAIN BATCH FUNCTION
# ============================================================

def run_batch(config):

    """
    Run batch maseg inference using settings supplied
    through the config dictionary.

    Parameters
    ----------
    config : dict
        Configuration loaded by main.py from config.json.

    Returns
    -------
    pandas.DataFrame
        Processing results/log.
    """


    # ========================================================
    # CONFIGURATION
    # ========================================================

    DATASET_NAME = str(
        config[
            "DATASET_NAME"
        ]
    ).upper()


    DATA_ROOT = config[
        "DATA_ROOT"
    ]


    MASEG_ROOT = config[
        "MASEG_ROOT"
    ]


    WEIGHTS_PATH = config[
        "WEIGHTS_PATH"
    ]


    OUTPUT_DIR = config[
        "OUTPUT_DIR"
    ]


    MLO_ONLY = config.get(
        "MLO_ONLY",
        True
    )


    MAX_DIM = config.get(
        "MAX_DIM",
        2048
    )


    LIMIT = config.get(
        "LIMIT",
        None
    )


    PROCESSING_LOG = os.path.join(
        OUTPUT_DIR,
        "processing_log.csv"
    )


    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # ========================================================
    # IMPORT MASEG
    # ========================================================

    if MASEG_ROOT not in sys.path:

        sys.path.insert(
            0,
            MASEG_ROOT
        )


    try:

        from maseg.run_model import RunModel


    except ImportError as e:

        raise ImportError(
            "Could not import maseg.\n"
            f"MASEG_ROOT is currently:\n"
            f"{MASEG_ROOT}\n\n"
            "Check that this points to the cloned maseg "
            "repository."
        ) from e


    # ========================================================
    # SHOW SETTINGS
    # ========================================================

    print(
        "\nBatch configuration:"
    )


    print(
        "Dataset:",
        DATASET_NAME
    )


    if (
        DATASET_NAME == "CBIS_DDSM"
        and
        "METADATA_CSV" in config
    ):

        print(
            "Metadata CSV:",
            config[
                "METADATA_CSV"
            ]
        )


    print(
        "Data root:",
        DATA_ROOT
    )


    print(
        "maseg root:",
        MASEG_ROOT
    )


    print(
        "Weights:",
        WEIGHTS_PATH
    )


    print(
        "Output directory:",
        OUTPUT_DIR
    )


    if DATASET_NAME == "CBIS_DDSM":

        print(
            "MLO only:",
            MLO_ONLY
        )


    print(
        "Maximum image dimension:",
        MAX_DIM
    )


    if LIMIT is None:

        print(
            "Image limit: ALL"
        )

    else:

        print(
            "Image limit:",
            LIMIT
        )


    # ========================================================
    # LOAD DATASET
    # ========================================================

    unique_images = load_dataset(
        config
    )


    # ========================================================
    # VALIDATE LOADER OUTPUT
    # ========================================================

    required_columns = [
        "image_key",
        "PatientID",
        "image_path"
    ]


    missing_columns = [
        column
        for column in required_columns
        if column not in unique_images.columns
    ]


    if missing_columns:

        raise ValueError(
            "Dataset loader did not return the required "
            "columns: "
            + ", ".join(
                missing_columns
            )
        )


    total_available = len(
        unique_images
    )


    # ========================================================
    # APPLY LIMIT
    # ========================================================

    if LIMIT is not None:

        unique_images = (
            unique_images
            .head(
                LIMIT
            )
            .copy()
            .reset_index(
                drop=True
            )
        )


    print(
        "\nTotal eligible images:",
        total_available
    )


    print(
        "Images selected for this run:",
        len(
            unique_images
        )
    )


    # ========================================================
    # STOP IF NOTHING TO PROCESS
    # ========================================================

    if len(
        unique_images
    ) == 0:

        print(
            "\nNo eligible images found."
        )

        return pd.DataFrame()


    # ========================================================
    # LOAD MODEL ONCE
    # ========================================================

    print(
        "\nLoading maseg model..."
    )


    model = RunModel(
        WEIGHTS_PATH
    )


    print(
        "Model loaded."
    )


    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    else:

        print(
            "CUDA not available - running on CPU."
        )


    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    results = []


    for i, row in unique_images.iterrows():

        image_key = row[
            "image_key"
        ]


        patient_id = row[
            "PatientID"
        ]


        image_path = row[
            "image_path"
        ]


        print(
            "\n"
            + "=" * 70
        )


        print(
            f"[{i + 1}/{len(unique_images)}] "
            f"{image_key}"
        )


        print(
            "=" * 70
        )


        print(
            "Patient:",
            patient_id
        )


        print(
            "DICOM:",
            image_path
        )


        output_segmentation = os.path.join(
            OUTPUT_DIR,
            f"{image_key}_maseg.npy"
        )


        # ----------------------------------------------------
        # Skip if already processed
        # ----------------------------------------------------

        if os.path.exists(
            output_segmentation
        ):

            print(
                "Already processed - skipping."
            )


            results.append(
                {
                    "dataset":
                        DATASET_NAME,

                    "image_key":
                        image_key,

                    "patient_id":
                        patient_id,

                    "image_path":
                        image_path,

                    "status":
                        "already_processed",

                    "segmentation_path":
                        output_segmentation,

                    "error":
                        ""
                }
            )


            continue


        # ----------------------------------------------------
        # Initialise variables for safe cleanup
        # ----------------------------------------------------

        image_raw = None

        image = None

        segmentation = None


        try:

            # =================================================
            # CHECK DICOM EXISTS
            # =================================================

            if not os.path.isfile(
                image_path
            ):

                raise FileNotFoundError(
                    f"DICOM file not found:\n"
                    f"{image_path}"
                )


            # =================================================
            # LOAD IMAGE
            # =================================================

            ds, image_raw = load_dicom(
                image_path
            )


            print(
                "Raw shape:",
                image_raw.shape
            )


            # =================================================
            # PREPROCESS
            # =================================================

            image = preprocess_mammogram(
                ds,
                image_raw
            )


            # =================================================
            # RUN MASEG
            # =================================================

            segmentation, scale = run_maseg(
                model,
                image,
                MAX_DIM
            )


            print(
                "Output classes:",
                np.unique(
                    segmentation
                )
            )


            # =================================================
            # SAVE SEGMENTATION
            # =================================================

            np.save(
                output_segmentation,
                segmentation
            )


            print(
                "Saved:",
                output_segmentation
            )


            # =================================================
            # SAVE RESULT
            # =================================================

            results.append(
                {
                    "dataset":
                        DATASET_NAME,

                    "image_key":
                        image_key,

                    "patient_id":
                        patient_id,

                    "image_path":
                        image_path,

                    "original_height":
                        image.shape[0],

                    "original_width":
                        image.shape[1],

                    "scale_factor":
                        scale,

                    "status":
                        "success",

                    "segmentation_path":
                        output_segmentation,

                    "error":
                        ""
                }
            )


        # ====================================================
        # CUDA OUT-OF-MEMORY
        # ====================================================

        except torch.cuda.OutOfMemoryError as e:

            print(
                "CUDA OOM:",
                str(e)
            )


            results.append(
                {
                    "dataset":
                        DATASET_NAME,

                    "image_key":
                        image_key,

                    "patient_id":
                        patient_id,

                    "image_path":
                        image_path,

                    "status":
                        "cuda_oom",

                    "segmentation_path":
                        "",

                    "error":
                        str(e)
                }
            )


        # ====================================================
        # OTHER ERRORS
        # ====================================================

        except Exception as e:

            print(
                "ERROR:",
                str(e)
            )


            results.append(
                {
                    "dataset":
                        DATASET_NAME,

                    "image_key":
                        image_key,

                    "patient_id":
                        patient_id,

                    "image_path":
                        image_path,

                    "status":
                        "failed",

                    "segmentation_path":
                        "",

                    "error":
                        str(e)
                }
            )


        finally:

            # ------------------------------------------------
            # Save processing log continuously
            # ------------------------------------------------

            pd.DataFrame(
                results
            ).to_csv(
                PROCESSING_LOG,
                index=False
            )


            # ------------------------------------------------
            # Cleanup
            # ------------------------------------------------

            if image_raw is not None:

                del image_raw


            if image is not None:

                del image


            if segmentation is not None:

                del segmentation


            gc.collect()


            if torch.cuda.is_available():

                torch.cuda.empty_cache()


    # ========================================================
    # FINAL LOG
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    results_df.to_csv(
        PROCESSING_LOG,
        index=False
    )


    print(
        "\nBatch processing complete."
    )


    if (
        len(
            results_df
        ) > 0
        and
        "status"
        in results_df.columns
    ):

        print(
            "\nProcessing summary:"
        )


        print(
            results_df[
                "status"
            ].value_counts(
                dropna=False
            )
        )


    print(
        "\nProcessing log:"
    )


    print(
        PROCESSING_LOG
    )


    return results_df