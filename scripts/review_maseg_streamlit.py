#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:58:54 2026

@author: c_piazzese
"""


"""
Interactive qualitative review interface for maseg outputs.

This Streamlit application is designed to be launched from main.py.

Configuration is passed from main.py through environment variables.
The number of cases available for review is determined automatically
from the segmentation files already present in OUTPUT_DIR.

The reviewer can:
- inspect the original mammogram
- inspect the maseg breast and pectoral segmentation
- inspect the raw segmentation mask
- provide qualitative visual feedback
- save one evaluation per image into a CSV
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import glob

import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import streamlit as st

from matplotlib.lines import Line2D


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="maseg evaluation",
    layout="wide"
)

st.title(
    "maseg Breast / Background Segmentation Evaluation"
)

st.caption(
    "Qualitative visual evaluation of maseg breast and "
    "pectoral segmentation."
)


# ============================================================
# CONFIGURATION FROM main.py
# ============================================================

DATA_ROOT = os.environ.get(
    "MASEG_DATA_ROOT"
)

OUTPUT_DIR = os.environ.get(
    "MASEG_OUTPUT_DIR"
)

METADATA_CSV = os.environ.get(
    "MASEG_METADATA_CSV"
)

EVALUATION_OUTPUT_DIR = os.environ.get(
    "MASEG_EVALUATION_OUTPUT_DIR"
)

EVALUATION_LABEL_DEFAULT = os.environ.get(
    "MASEG_EVALUATION_LABEL",
    "review_1"
)

MLO_ONLY = (
    os.environ.get(
        "MASEG_MLO_ONLY",
        "True"
    ).lower()
    ==
    "true"
)


# ============================================================
# CHECK CONFIGURATION
# ============================================================

required_config = {
    "DATA_ROOT": DATA_ROOT,
    "OUTPUT_DIR": OUTPUT_DIR,
    "METADATA_CSV": METADATA_CSV,
    "EVALUATION_OUTPUT_DIR": EVALUATION_OUTPUT_DIR
}

missing_config = [
    key
    for key, value
    in required_config.items()
    if not value
]

if missing_config:

    st.error(
        "Missing configuration values: "
        + ", ".join(
            missing_config
        )
    )

    st.error(
        "Start this application through main.py so that "
        "the configuration can be passed from config.json."
    )

    st.stop()


# ============================================================
# CREATE EVALUATION OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    EVALUATION_OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SIDEBAR - CONFIGURATION SUMMARY
# ============================================================

st.sidebar.header(
    "Configuration"
)

st.sidebar.caption(
    "CBIS-DDSM data root"
)

st.sidebar.code(
    DATA_ROOT
)

st.sidebar.caption(
    "maseg output directory"
)

st.sidebar.code(
    OUTPUT_DIR
)

st.sidebar.caption(
    "Metadata CSV"
)

st.sidebar.code(
    METADATA_CSV
)

st.sidebar.caption(
    "Evaluation output directory"
)

st.sidebar.code(
    EVALUATION_OUTPUT_DIR
)

st.sidebar.caption(
    "View filter"
)

if MLO_ONLY:

    st.sidebar.code(
        "MLO only"
    )

else:

    st.sidebar.code(
        "All available views"
    )


# ============================================================
# EVALUATION LABEL
# ============================================================

st.sidebar.header(
    "Evaluation"
)

evaluation_label = st.sidebar.text_input(
    "Evaluation / reviewer label",
    value=EVALUATION_LABEL_DEFAULT,
    help=(
        "Used to create the CSV filename. "
        "Examples: Concetta, reviewer_A, test_run."
    )
)

evaluation_label_safe = re.sub(
    r"[^A-Za-z0-9_-]+",
    "_",
    evaluation_label.strip()
)

if not evaluation_label_safe:

    evaluation_label_safe = (
        "review_1"
    )

EVALUATION_CSV = os.path.join(
    EVALUATION_OUTPUT_DIR,
    (
        "maseg_evaluation_"
        f"{evaluation_label_safe}.csv"
    )
)

st.sidebar.caption(
    "Evaluation file:"
)

st.sidebar.code(
    EVALUATION_CSV
)


# ============================================================
# CHECK PATHS
# ============================================================

path_errors = []

if not os.path.isdir(
    DATA_ROOT
):

    path_errors.append(
        f"Data root does not exist:\n{DATA_ROOT}"
    )

if not os.path.isdir(
    OUTPUT_DIR
):

    path_errors.append(
        f"maseg output directory does not exist:\n"
        f"{OUTPUT_DIR}"
    )

if not os.path.isfile(
    METADATA_CSV
):

    path_errors.append(
        f"Metadata CSV does not exist:\n"
        f"{METADATA_CSV}"
    )

if len(
    path_errors
) > 0:

    st.error(
        "One or more configured paths are invalid."
    )

    for error in path_errors:

        st.code(
            error
        )

    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_image_key(
    patient_id
):

    """
    Extract:

        P_00038_LEFT_MLO

    from PatientID strings such as:

        P_00038_LEFT_MLO.dcm

    or:

        Calc-Test_P_00038_LEFT_MLO_1
    """

    match = re.search(
        r"(P_\d+_(?:LEFT|RIGHT)_(?:CC|MLO))",
        str(
            patient_id
        )
    )

    if match:

        return match.group(
            1
        )

    return None


def find_dicom_for_patient(
    patient_id
):

    """
    Find all DICOM files underneath a patient's
    CBIS-DDSM directory.
    """

    patient_dir = os.path.join(
        DATA_ROOT,
        str(
            patient_id
        )
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


@st.cache_data
def load_metadata(
    metadata_csv
):

    df = pd.read_csv(
        metadata_csv
    )

    df["image_key"] = (
        df[
            "PatientID"
        ]
        .apply(
            get_image_key
        )
    )

    return df


@st.cache_data
def load_original_image(
    image_path
):

    ds = pydicom.dcmread(
        image_path
    )

    image = (
        ds.pixel_array
        .astype(
            np.float32
        )
    )

    photometric = getattr(
        ds,
        "PhotometricInterpretation",
        ""
    )

    if photometric == "MONOCHROME1":

        image = (
            image.max()
            -
            image
        )

    image = (
        image
        -
        image.min()
    )

    maximum = (
        image.max()
    )

    if maximum > 0:

        image = (
            image
            /
            maximum
        )

    return image


@st.cache_data
def load_segmentation(
    segmentation_path
):

    return np.load(
        segmentation_path
    )


def load_existing_evaluations():

    if os.path.exists(
        EVALUATION_CSV
    ):

        try:

            return pd.read_csv(
                EVALUATION_CSV
            )

        except Exception:

            return pd.DataFrame()

    return pd.DataFrame()


# ============================================================
# LOAD METADATA
# ============================================================

df = load_metadata(
    METADATA_CSV
)


# ============================================================
# IDENTIFY FULL MAMMOGRAMS
# ============================================================

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


# ------------------------------------------------------------
# Include original mammograms where SeriesDescription is blank
# ------------------------------------------------------------

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


# ============================================================
# FILTER BY VIEW
# ============================================================

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


# ============================================================
# ONE ROW PER MAMMOGRAM
# ============================================================

full_images = (
    full_images
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


# ============================================================
# FIND CASES WITH SAVED MASEG SEGMENTATION
# ============================================================
#
# No LIMIT is needed here.
#
# The number of cases available for review is determined by the
# number of matching *_maseg.npy files already present in
# OUTPUT_DIR.
# ============================================================

processed = []

for _, row in full_images.iterrows():

    image_key = (
        row[
            "image_key"
        ]
    )

    segmentation_path = os.path.join(
        OUTPUT_DIR,
        f"{image_key}_maseg.npy"
    )

    if os.path.exists(
        segmentation_path
    ):

        processed.append(
            {
                "image_key":
                    image_key,

                "PatientID":
                    row[
                        "PatientID"
                    ],

                "segmentation_path":
                    segmentation_path
            }
        )

cases = pd.DataFrame(
    processed
)

if len(
    cases
) == 0:

    if MLO_ONLY:

        message = (
            "No processed maseg .npy files were found "
            "for the MLO mammograms."
        )

    else:

        message = (
            "No processed maseg .npy files were found "
            "for the available mammograms."
        )

    st.error(
        message
    )

    st.stop()


# ============================================================
# LOAD EXISTING REVIEWS
# ============================================================

evaluations = (
    load_existing_evaluations()
)

evaluated_keys = set()

if (
    len(
        evaluations
    ) > 0
    and
    "image_key"
    in evaluations.columns
):

    evaluated_keys = set(
        evaluations[
            "image_key"
        ]
        .dropna()
    )


# ============================================================
# SIDEBAR PROGRESS
# ============================================================

st.sidebar.header(
    "Review progress"
)

n_processed = len(
    cases
)

n_reviewed = len(
    set(
        cases[
            "image_key"
        ]
    )
    &
    evaluated_keys
)

n_remaining = (
    n_processed
    -
    n_reviewed
)

st.sidebar.metric(
    "Processed",
    n_processed
)

st.sidebar.metric(
    "Reviewed",
    n_reviewed
)

st.sidebar.metric(
    "Remaining",
    n_remaining
)

if n_processed > 0:

    st.sidebar.progress(
        n_reviewed
        /
        n_processed
    )


# ============================================================
# FILTER
# ============================================================

show_only_unreviewed = (
    st.sidebar.checkbox(
        "Show only unreviewed",
        value=False
    )
)

if show_only_unreviewed:

    cases_display = (
        cases[
            ~cases[
                "image_key"
            ]
            .isin(
                evaluated_keys
            )
        ]
        .reset_index(
            drop=True
        )
    )

else:

    cases_display = (
        cases
        .reset_index(
            drop=True
        )
    )

if len(
    cases_display
) == 0:

    st.success(
        "All available cases have been reviewed."
    )

    st.stop()


# ============================================================
# CASE SELECTION
# ============================================================

st.sidebar.header(
    "Case selection"
)

case_number = (
    st.sidebar.number_input(
        "Case number",
        min_value=1,
        max_value=len(
            cases_display
        ),
        value=1,
        step=1
    )
)

row = (
    cases_display
    .iloc[
        case_number - 1
    ]
)

image_key = (
    row[
        "image_key"
    ]
)

patient_id = (
    row[
        "PatientID"
    ]
)

segmentation_path = (
    row[
        "segmentation_path"
    ]
)

st.sidebar.write(
    f"Case {case_number} "
    f"of {len(cases_display)}"
)

st.sidebar.code(
    image_key
)

if image_key in evaluated_keys:

    st.sidebar.success(
        "Already reviewed"
    )

else:

    st.sidebar.warning(
        "Not reviewed"
    )


# ============================================================
# FIND ORIGINAL DICOM
# ============================================================

dicom_files = (
    find_dicom_for_patient(
        patient_id
    )
)

if len(
    dicom_files
) == 0:

    st.error(
        f"No original DICOM found for "
        f"{patient_id}"
    )

    st.stop()

image_path = (
    dicom_files[0]
)


# ============================================================
# LOAD IMAGE + SEGMENTATION
# ============================================================

image = (
    load_original_image(
        image_path
    )
)

segmentation = (
    load_segmentation(
        segmentation_path
    )
)


# ============================================================
# CHECK DIMENSIONS
# ============================================================

if (
    image.shape
    !=
    segmentation.shape
):

    st.error(
        "Original image and segmentation have "
        "different dimensions."
    )

    st.write(
        f"Image: {image.shape}"
    )

    st.write(
        f"Segmentation: "
        f"{segmentation.shape}"
    )

    st.stop()


# ============================================================
# CASE INFORMATION
# ============================================================

st.subheader(
    image_key
)

info1, info2, info3 = (
    st.columns(
        3
    )
)

info1.metric(
    "Image height",
    image.shape[
        0
    ]
)

info2.metric(
    "Image width",
    image.shape[
        1
    ]
)

classes_present = (
    np.unique(
        segmentation
    )
    .tolist()
)

info3.metric(
    "Classes present",
    str(
        classes_present
    )
)

st.caption(
    "maseg classes: "
    "0 = background, "
    "1 = breast, "
    "2 = pectoral muscle"
)


# ============================================================
# DISPLAY ORIGINAL + OVERLAY
# ============================================================

col1, col2 = (
    st.columns(
        2
    )
)


# ------------------------------------------------------------
# Original
# ------------------------------------------------------------

with col1:

    st.subheader(
        "Original mammogram"
    )

    fig, ax = (
        plt.subplots(
            figsize=(
                7,
                10
            )
        )
    )

    ax.imshow(
        image,
        cmap="gray"
    )

    ax.axis(
        "off"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(
        fig
    )


# ------------------------------------------------------------
# maseg overlay
# ------------------------------------------------------------

with col2:

    st.subheader(
        "maseg segmentation"
    )

    fig, ax = plt.subplots(
        figsize=(
            7,
            10
        )
    )

    # Original mammogram
    ax.imshow(
        image,
        cmap="gray"
    )

    # --------------------------------------------------------
    # Breast mask
    # --------------------------------------------------------

    breast_mask = (
        segmentation == 1
    )

    if np.any(
        breast_mask
    ):

        ax.imshow(
            np.ma.masked_where(
                ~breast_mask,
                breast_mask
            ),
            alpha=0.20,
            cmap="Reds"
        )

        ax.contour(
            breast_mask,
            levels=[
                0.5
            ],
            linewidths=2,
            colors="red"
        )

    # --------------------------------------------------------
    # Pectoral muscle
    # --------------------------------------------------------

    pectoral_mask = (
        segmentation == 2
    )

    if np.any(
        pectoral_mask
    ):

        ax.imshow(
            np.ma.masked_where(
                ~pectoral_mask,
                pectoral_mask
            ),
            alpha=0.20,
            cmap="Blues"
        )

        ax.contour(
            pectoral_mask,
            levels=[
                0.5
            ],
            linewidths=2,
            colors="cyan"
        )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="red",
            lw=2,
            label="Breast"
        ),
        Line2D(
            [0],
            [0],
            color="cyan",
            lw=2,
            label="Pectoral muscle"
        )
    ]

    ax.legend(
        handles=legend_elements,
        loc="lower left"
    )

    ax.axis(
        "off"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(
        fig
    )


# ============================================================
# OPTIONAL MASK VIEWS
# ============================================================

with st.expander(
    "Show raw segmentation mask"
):

    fig, ax = (
        plt.subplots(
            figsize=(
                7,
                10
            )
        )
    )

    ax.imshow(
        segmentation,
        vmin=0,
        vmax=2
    )

    ax.axis(
        "off"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(
        fig
    )


with st.expander(
    "Show breast + pectoral foreground mask"
):

    foreground = (
        segmentation
        >
        0
    )

    fig, ax = (
        plt.subplots(
            figsize=(
                7,
                10
            )
        )
    )

    ax.imshow(
        foreground,
        cmap="gray"
    )

    ax.axis(
        "off"
    )

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(
        fig
    )


# ============================================================
# EXISTING ASSESSMENT FOR CURRENT IMAGE
# ============================================================

existing = None

if (
    len(
        evaluations
    ) > 0
    and
    "image_key"
    in evaluations.columns
):

    previous = evaluations[
        evaluations[
            "image_key"
        ]
        ==
        image_key
    ]

    if len(
        previous
    ) > 0:

        existing = (
            previous
            .iloc[
                -1
            ]
        )


def previous_value(
    column,
    default
):

    if existing is None:

        return default

    if column not in existing.index:

        return default

    value = (
        existing[
            column
        ]
    )

    if pd.isna(
        value
    ):

        return default

    return value


# ============================================================
# QUALITATIVE EVALUATION
# ============================================================

st.header(
    "Qualitative visual assessment"
)

st.info(
    "This is a qualitative visual assessment. "
    "No breast/background ground-truth mask is "
    "available for this dataset."
)


# ============================================================
# OVERALL QUALITY
# ============================================================

quality_options = [
    "Good",
    "Minor errors",
    "Poor",
    "Unable to assess"
]

previous_quality = (
    previous_value(
        "overall_quality",
        "Good"
    )
)

if previous_quality in quality_options:

    quality_index = (
        quality_options
        .index(
            previous_quality
        )
    )

else:

    quality_index = 0

overall_quality = (
    st.radio(
        "Overall segmentation quality",
        quality_options,
        index=quality_index,
        horizontal=True,
        key=(
            f"quality_"
            f"{image_key}"
        )
    )
)


# ============================================================
# DETAILED ASSESSMENT
# ============================================================

st.subheader(
    "Segmentation characteristics"
)

c1, c2, c3 = (
    st.columns(
        3
    )
)


# ------------------------------------------------------------
# Breast boundary
# ------------------------------------------------------------

boundary_options = [
    "Correct",
    "Minor error",
    "Major error",
    "Unable to assess"
]

previous_boundary = previous_value(
    "breast_boundary",
    "Correct"
)

boundary_index = (
    boundary_options.index(
        previous_boundary
    )
    if previous_boundary
    in boundary_options
    else 0
)

with c1:

    breast_boundary = (
        st.selectbox(
            "Breast boundary",
            boundary_options,
            index=boundary_index,
            key=(
                f"boundary_"
                f"{image_key}"
            )
        )
    )


# ------------------------------------------------------------
# Background leakage
# ------------------------------------------------------------

leakage_options = [
    "None",
    "Minor",
    "Substantial",
    "Unable to assess"
]

previous_leakage = previous_value(
    "background_leakage",
    "None"
)

leakage_index = (
    leakage_options.index(
        previous_leakage
    )
    if previous_leakage
    in leakage_options
    else 0
)

with c1:

    background_leakage = (
        st.selectbox(
            "Background leakage",
            leakage_options,
            index=leakage_index,
            key=(
                f"leakage_"
                f"{image_key}"
            )
        )
    )


# ------------------------------------------------------------
# Missing breast tissue
# ------------------------------------------------------------

missing_options = [
    "None",
    "Minor",
    "Substantial",
    "Unable to assess"
]

previous_missing = previous_value(
    "missing_breast_tissue",
    "None"
)

missing_index = (
    missing_options.index(
        previous_missing
    )
    if previous_missing
    in missing_options
    else 0
)

with c2:

    missing_tissue = (
        st.selectbox(
            "Missing breast tissue",
            missing_options,
            index=missing_index,
            key=(
                f"missing_"
                f"{image_key}"
            )
        )
    )


# ------------------------------------------------------------
# Pectoral muscle
# ------------------------------------------------------------

pectoral_options = [
    "Correct",
    "Minor error",
    "Major error",
    "Unable to assess"
]

previous_pectoral = previous_value(
    "pectoral_region",
    "Correct"
)

pectoral_index = (
    pectoral_options.index(
        previous_pectoral
    )
    if previous_pectoral
    in pectoral_options
    else 0
)

with c2:

    pectoral_region = (
        st.selectbox(
            "Pectoral muscle segmentation",
            pectoral_options,
            index=pectoral_index,
            key=(
                f"pectoral_"
                f"{image_key}"
            )
        )
    )


# ------------------------------------------------------------
# Artefacts
# ------------------------------------------------------------

artefact_options = [
    "None",
    "Minor",
    "Substantial",
    "Unable to assess"
]

previous_artefacts = previous_value(
    "artefacts",
    "None"
)

artefact_index = (
    artefact_options.index(
        previous_artefacts
    )
    if previous_artefacts
    in artefact_options
    else 0
)

with c3:

    artefacts = (
        st.selectbox(
            "Non-breast artefacts included",
            artefact_options,
            index=artefact_index,
            key=(
                f"artefacts_"
                f"{image_key}"
            )
        )
    )


# ------------------------------------------------------------
# Acceptability
# ------------------------------------------------------------

acceptable_options = [
    "Yes",
    "No",
    "Uncertain"
]

previous_acceptable = previous_value(
    "acceptable_segmentation",
    "Yes"
)

acceptable_index = (
    acceptable_options.index(
        previous_acceptable
    )
    if previous_acceptable
    in acceptable_options
    else 0
)

with c3:

    acceptable = (
        st.radio(
            "Acceptable segmentation?",
            acceptable_options,
            index=acceptable_index,
            horizontal=True,
            key=(
                f"acceptable_"
                f"{image_key}"
            )
        )
    )


# ============================================================
# COMMENTS
# ============================================================

previous_comments = previous_value(
    "comments",
    ""
)

comments = st.text_area(
    "Reviewer comments",
    value=str(
        previous_comments
    ),
    placeholder=(
        "Describe visible issues, for example: "
        "background leakage, missing breast tissue, "
        "pectoral boundary error, artefacts, etc."
    ),
    key=(
        f"comments_"
        f"{image_key}"
    )
)


# ============================================================
# SAVE QUALITATIVE REVIEW
# ============================================================

if st.button(
    "Save qualitative evaluation",
    type="primary"
):

    if image_key.endswith(
        "_MLO"
    ):

        view = "MLO"

    elif image_key.endswith(
        "_CC"
    ):

        view = "CC"

    else:

        view = ""

    new_row = pd.DataFrame(
        [
            {
                "image_key":
                    image_key,

                "patient_id":
                    patient_id,

                "view":
                    view,

                "overall_quality":
                    overall_quality,

                "acceptable_segmentation":
                    acceptable,

                "breast_boundary":
                    breast_boundary,

                "background_leakage":
                    background_leakage,

                "missing_breast_tissue":
                    missing_tissue,

                "pectoral_region":
                    pectoral_region,

                "artefacts":
                    artefacts,

                "comments":
                    comments,

                "evaluation_label":
                    evaluation_label_safe,

                "image_path":
                    image_path,

                "segmentation_path":
                    segmentation_path
            }
        ]
    )

    if os.path.exists(
        EVALUATION_CSV
    ):

        old = pd.read_csv(
            EVALUATION_CSV
        )

        if (
            "image_key"
            in old.columns
        ):

            # Replace previous review of same case
            old = old[
                old[
                    "image_key"
                ]
                !=
                image_key
            ]

        evaluation = pd.concat(
            [
                old,
                new_row
            ],
            ignore_index=True
        )

    else:

        evaluation = (
            new_row
        )

    evaluation.to_csv(
        EVALUATION_CSV,
        index=False
    )

    st.success(
        f"Evaluation saved for {image_key}"
    )

    st.caption(
        f"Saved to: {EVALUATION_CSV}"
    )

    st.rerun()


# ============================================================
# SHOW COMPLETED EVALUATIONS
# ============================================================

with st.expander(
    "Show completed evaluations"
):

    if os.path.exists(
        EVALUATION_CSV
    ):

        current_evaluations = (
            pd.read_csv(
                EVALUATION_CSV
            )
        )

        st.dataframe(
            current_evaluations,
            use_container_width=True
        )

        st.write(
            f"{len(current_evaluations)} "
            "cases reviewed."
        )

    else:

        st.write(
            "No evaluations saved yet."
        )
