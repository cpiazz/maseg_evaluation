#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:51:58 2026

@author: c_piazzese
"""

"""
Batch maseg evaluation on CBIS-DDSM mammograms.

This module is designed to be called from main.py.

The configuration is passed in as a dictionary loaded from config.json.

Expected config keys
--------------------
Required:
    METADATA_CSV
    DATA_ROOT
    MASEG_ROOT
    WEIGHTS_PATH
    OUTPUT_DIR

Optional:
    MLO_ONLY   -> default True
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
# HELPER FUNCTIONS
# ============================================================

def get_image_key(patient_id):

    """
    Extract a mammogram key such as:

        P_00038_LEFT_MLO

    from strings such as:

        P_00038_LEFT_MLO.dcm

    or:

        Calc-Test_P_00038_LEFT_MLO_1
    """

    name = str(patient_id)

    match = re.search(
        r"(P_\d+_(?:LEFT|RIGHT)_(?:CC|MLO))",
        name
    )

    if match:
        return match.group(1)

    return None


def find_dicom_for_patient(
    patient_id,
    data_root
):

    """
    Find DICOM files underneath the directory
    corresponding to the supplied PatientID.
    """

    patient_dir = os.path.join(
        data_root,
        str(patient_id)
    )

    if not os.path.exists(
        patient_dir
    ):

        return []


    files = glob.glob(
        os.path.join(
            patient_dir,
            "**",
            "*.dcm"
        ),
        recursive=True
    )

    return sorted(
        files
    )


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


def preprocess_mammogram(
    ds,
    image
):

    """
    Basic preprocessing used before passing the
    mammogram to maseg.

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

    METADATA_CSV = config[
        "METADATA_CSV"
    ]

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
            f"MASEG_ROOT is currently:\n{MASEG_ROOT}\n\n"
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
        "Metadata CSV:",
        METADATA_CSV
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
            get_image_key
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

    unique_images = (
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
        )


    print(
        "\nTotal eligible mammograms:",
        total_available
    )


    print(
        "Mammograms selected for this run:",
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
            "\nNo eligible mammograms found."
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
                    "image_key":
                        image_key,

                    "patient_id":
                        patient_id,

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
            # FIND DICOM
            # =================================================

            image_files = find_dicom_for_patient(
                patient_id,
                DATA_ROOT
            )


            if len(
                image_files
            ) == 0:

                raise FileNotFoundError(
                    f"No DICOM found for "
                    f"{patient_id}"
                )


            image_path = (
                image_files[0]
            )


            print(
                "DICOM:",
                image_path
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


            results.append(
                {
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
                    "image_key":
                        image_key,

                    "patient_id":
                        patient_id,

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
                    "image_key":
                        image_key,

                    "patient_id":
                        patient_id,

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