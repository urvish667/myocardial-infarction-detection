#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example usage of the medical_image_processor.py script.

This script demonstrates how to use the main functions from the
medical_image_processor.py script in your own Python code.
"""

import logging
from pathlib import Path
from medical_image_processor import (
    process_mat_files,
    extract_frames_from_videos,
    match_frames_with_masks,
    rename_folders_remove_spaces,
    setup_directories,
    logger
)

def example_workflow():
    """
    Example workflow demonstrating how to use the medical image processing functions.
    """
    # Define base directory and paths
    base_dir = Path(".")  # Current directory
    mat_folder = base_dir / "archive/LV Ground-truth Segmentation Masks"
    video_folder = base_dir / "HMC-QU/A4C"

    # Setup output directories
    output_dirs = setup_directories(base_dir, ["Frames/A4C", "Masks"])
    frames_folder = output_dirs["Frames/A4C"]
    masks_folder = output_dirs["Masks"]

    # Step 1: Process .mat files to extract masks
    logger.info("Step 1: Processing .mat files to extract masks")
    process_mat_files(mat_folder, masks_folder)

    # Step 2: Extract frames from videos
    logger.info("Step 2: Extracting frames from videos")
    extract_frames_from_videos(video_folder, frames_folder, desired_fps=25)

    # Step 3: Standardize folder names
    logger.info("Step 3: Standardizing folder names")
    rename_folders_remove_spaces(frames_folder)
    rename_folders_remove_spaces(masks_folder)

    # Step 4: Match frames with masks and ensure consistent frame counts
    logger.info("Step 4: Matching frames with masks")
    match_frames_with_masks(frames_folder, masks_folder)

    logger.info("Processing completed successfully")


def custom_processing_example():
    """
    Example of custom processing using only specific functions.
    """
    # Define custom paths
    base_dir = Path(".")
    custom_mat_folder = base_dir / "my_mat_files"
    custom_masks_folder = base_dir / "my_masks"

    # Only process .mat files
    logger.info("Custom processing: only extracting masks from .mat files")
    process_mat_files(custom_mat_folder, custom_masks_folder)

    # Rename folders to standardize names
    rename_folders_remove_spaces(custom_masks_folder)

    logger.info("Custom processing completed successfully")


def fix_missing_folders_example():
    """
    Example of how to fix missing frame or mask folders.
    """
    # Enable debug logging
    logger.setLevel(logging.DEBUG)

    # Define base directory and paths
    base_dir = Path(".")
    frames_folder = base_dir / "Frames/A4C"
    masks_folder = base_dir / "Masks"

    # First do a dry run to see what would happen
    logger.info("Dry run: checking for missing folders without making changes")
    match_frames_with_masks(
        frames_folder,
        masks_folder,
        create_missing_frames=True,  # Create missing frame folders
        create_missing_masks=True,   # Create missing mask folders
        dry_run=True,                # Don't actually make changes
        flexible_matching=True       # Try case-insensitive matching
    )

    # Now actually create the missing folders
    logger.info("Creating missing folders")
    match_frames_with_masks(
        frames_folder,
        masks_folder,
        create_missing_frames=True,
        create_missing_masks=True,
        dry_run=False,
        flexible_matching=True
    )

    logger.info("Missing folder fix completed successfully")


if __name__ == "__main__":
    # Choose which example to run
    example_workflow()
    # custom_processing_example()
    # fix_missing_folders_example()
