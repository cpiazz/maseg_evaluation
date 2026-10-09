#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Oct  6 14:13:16 2026

@author: c_piazzese
"""


"""
Main entry point for the maseg evaluation pipeline.

Workflow
--------
1. Load configuration from config.json
2. Validate configuration values and paths
3. Run maseg batch processing if RUN_BATCH is true
4. Launch the Streamlit review interface

All main settings are controlled from config.json.

Supported datasets
------------------
- CBIS_DDSM
- CMMD

Dataset-specific loading is handled downstream in
scripts/dataset_loaders.py.
"""

import os
import json
import subprocess
import sys

from scripts.run_maseg_batch import run_batch


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "config.json"
)


# ============================================================
# LOAD CONFIGURATION
# ============================================================

def load_config(config_path):

    if not os.path.isfile(config_path):

        raise FileNotFoundError(
            f"Configuration file not found:\n"
            f"{config_path}\n\n"
            f"Copy config.example.json to config.json "
            f"and edit the paths/settings."
        )


    with open(
        config_path,
        "r"
    ) as f:

        config = json.load(f)


    # --------------------------------------------------------
    # Required configuration entries shared by all datasets
    # --------------------------------------------------------

    required_keys = [
        "DATASET_NAME",
        "DATA_ROOT",
        "MASEG_ROOT",
        "WEIGHTS_PATH",
        "OUTPUT_DIR"
    ]


    missing = [
        key
        for key in required_keys
        if key not in config
    ]


    if missing:

        raise KeyError(
            "Missing required configuration values: "
            + ", ".join(missing)
        )


    # --------------------------------------------------------
    # Dataset name
    # --------------------------------------------------------

    dataset_name = str(
        config[
            "DATASET_NAME"
        ]
    ).upper()


    supported_datasets = [
        "CBIS_DDSM",
        "CMMD"
    ]


    if dataset_name not in supported_datasets:

        raise ValueError(
            "Unsupported DATASET_NAME: "
            f"{dataset_name}\n"
            "Supported datasets: "
            + ", ".join(
                supported_datasets
            )
        )


    # Store normalised dataset name
    config[
        "DATASET_NAME"
    ] = dataset_name


    # --------------------------------------------------------
    # Dataset-specific required configuration
    # --------------------------------------------------------

    if dataset_name == "CBIS_DDSM":

        if "METADATA_CSV" not in config:

            raise KeyError(
                "METADATA_CSV is required "
                "when DATASET_NAME is CBIS_DDSM."
            )


    # --------------------------------------------------------
    # Optional settings and defaults
    # --------------------------------------------------------

    config.setdefault(
        "EVALUATION_OUTPUT_DIR",
        os.path.join(
            PROJECT_ROOT,
            "evaluations"
        )
    )

    config.setdefault(
        "EVALUATION_LABEL",
        "review_1"
    )

    config.setdefault(
        "RUN_BATCH",
        True
    )

    config.setdefault(
        "LIMIT",
        None
    )

    config.setdefault(
        "MLO_ONLY",
        True
    )

    config.setdefault(
        "MAX_DIM",
        2048
    )


    # --------------------------------------------------------
    # Validate optional settings
    # --------------------------------------------------------

    if not isinstance(
        config["RUN_BATCH"],
        bool
    ):

        raise ValueError(
            "RUN_BATCH must be true or false."
        )


    if (
        config["LIMIT"] is not None
        and (
            not isinstance(
                config["LIMIT"],
                int
            )
            or config["LIMIT"] <= 0
        )
    ):

        raise ValueError(
            "LIMIT must be a positive integer or null."
        )


    if not isinstance(
        config["MLO_ONLY"],
        bool
    ):

        raise ValueError(
            "MLO_ONLY must be true or false."
        )


    if (
        not isinstance(
            config["MAX_DIM"],
            int
        )
        or config["MAX_DIM"] <= 0
    ):

        raise ValueError(
            "MAX_DIM must be a positive integer."
        )


    return config


# ============================================================
# VALIDATE PATHS
# ============================================================

def validate_config(config):

    dataset_name = config[
        "DATASET_NAME"
    ]


    # --------------------------------------------------------
    # Dataset-specific paths
    # --------------------------------------------------------

    if dataset_name == "CBIS_DDSM":

        if not os.path.isfile(
            config["METADATA_CSV"]
        ):

            raise FileNotFoundError(
                "Metadata CSV not found:\n"
                + config["METADATA_CSV"]
            )


    # --------------------------------------------------------
    # Common paths
    # --------------------------------------------------------

    if not os.path.isdir(
        config["DATA_ROOT"]
    ):

        raise FileNotFoundError(
            "Data root not found:\n"
            + config["DATA_ROOT"]
        )


    if not os.path.isdir(
        config["MASEG_ROOT"]
    ):

        raise FileNotFoundError(
            "maseg root not found:\n"
            + config["MASEG_ROOT"]
        )


    if not os.path.isfile(
        config["WEIGHTS_PATH"]
    ):

        raise FileNotFoundError(
            "Model checkpoint not found:\n"
            + config["WEIGHTS_PATH"]
        )


    # --------------------------------------------------------
    # Create output directories if needed
    # --------------------------------------------------------

    os.makedirs(
        config["OUTPUT_DIR"],
        exist_ok=True
    )


    os.makedirs(
        config["EVALUATION_OUTPUT_DIR"],
        exist_ok=True
    )


# ============================================================
# PRINT CONFIGURATION
# ============================================================

def print_configuration(config):

    print(
        "\n========================================"
    )

    print(
        "Configuration"
    )

    print(
        "========================================"
    )


    print(
        "Dataset:",
        config["DATASET_NAME"]
    )


    if "METADATA_CSV" in config:

        print(
            "Metadata CSV:",
            config["METADATA_CSV"]
        )


    print(
        "Data root:",
        config["DATA_ROOT"]
    )

    print(
        "maseg root:",
        config["MASEG_ROOT"]
    )

    print(
        "Weights:",
        config["WEIGHTS_PATH"]
    )

    print(
        "Output directory:",
        config["OUTPUT_DIR"]
    )

    print(
        "Evaluation output directory:",
        config["EVALUATION_OUTPUT_DIR"]
    )

    print(
        "Evaluation label:",
        config["EVALUATION_LABEL"]
    )

    print(
        "Run batch:",
        config["RUN_BATCH"]
    )

    print(
        "MLO only:",
        config["MLO_ONLY"]
    )

    print(
        "Maximum image dimension:",
        config["MAX_DIM"]
    )


    if config["LIMIT"] is None:

        print(
            "Image limit: ALL"
        )

    else:

        print(
            "Image limit:",
            config["LIMIT"]
        )


    print(
        "========================================\n"
    )


# ============================================================
# START STREAMLIT
# ============================================================

def start_streamlit(config):

    streamlit_script = os.path.join(
        PROJECT_ROOT,
        "scripts",
        "review_maseg_streamlit.py"
    )


    if not os.path.isfile(
        streamlit_script
    ):

        raise FileNotFoundError(
            "Streamlit review script not found:\n"
            + streamlit_script
        )


    # --------------------------------------------------------
    # Pass configuration values to Streamlit
    #
    # Streamlit runs as a separate process, so the config
    # dictionary cannot be passed directly.
    # Environment variables are used instead.
    # --------------------------------------------------------

    env = os.environ.copy()


    for key, value in config.items():

        if value is None:

            value = ""

        env[
            f"MASEG_{key}"
        ] = str(
            value
        )


    print(
        "\n========================================"
    )

    print(
        "Starting Streamlit review interface"
    )

    print(
        "========================================\n"
    )


    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            streamlit_script
        ],
        env=env,
        check=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    print(
        "\nLoading configuration..."
    )


    config = load_config(
        CONFIG_PATH
    )


    validate_config(
        config
    )


    print(
        "Configuration loaded successfully."
    )


    print_configuration(
        config
    )


    # ========================================================
    # 1. RUN MASEG BATCH PROCESSING
    # ========================================================

    if config["RUN_BATCH"]:

        print(
            "\n========================================"
        )

        print(
            "Running maseg batch processing"
        )


        print(
            "Dataset:",
            config[
                "DATASET_NAME"
            ]
        )


        if config["LIMIT"] is None:

            print(
                "Images selected: ALL"
            )

        else:

            print(
                "Images selected:",
                config["LIMIT"]
            )


        print(
            "========================================\n"
        )


        run_batch(
            config
        )


        print(
            "\nBatch processing finished."
        )


    else:

        print(
            "\nRUN_BATCH is false."
        )

        print(
            "Skipping maseg batch processing."
        )


    # ========================================================
    # 2. START STREAMLIT REVIEW
    # ========================================================

    start_streamlit(
        config
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()