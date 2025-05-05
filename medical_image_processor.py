#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical Image Processing Script

This script processes medical imaging data by:
1. Extracting frames from video files
2. Processing segmentation masks from .mat files
3. Organizing and matching frames with corresponding masks
4. Ensuring consistent naming and frame counts

Author: Augment Agent
"""

import os
import glob
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
import scipy.io
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("medical_image_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_directories(base_dir: Path, dirs: List[str]) -> Dict[str, Path]:
    """
    Create necessary directories if they don't exist.

    Args:
        base_dir: Base directory path
        dirs: List of directory names to create

    Returns:
        Dictionary mapping directory names to Path objects
    """
    paths = {}
    for dir_name in dirs:
        dir_path = base_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        paths[dir_name] = dir_path
        logger.info(f"Directory setup: {dir_path}")
    return paths


def standardize_name(name: str) -> str:
    """
    Standardize folder/file names by removing spaces.

    Args:
        name: Original name

    Returns:
        Standardized name
    """
    return name.replace(" ", "")


def process_mat_files(mat_folder: Path, output_folder: Path) -> int:
    """
    Process .mat files containing segmentation masks and save as images.

    Args:
        mat_folder: Path to folder containing .mat files
        output_folder: Path to save extracted mask images

    Returns:
        Number of successfully processed .mat files
    """
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get all .mat files in the directory
    mat_files = list(mat_folder.glob("*.mat"))
    logger.info(f"Found {len(mat_files)} .mat files in {mat_folder}")

    processed_count = 0

    for mat_file in mat_files:
        try:
            # Get the name of the file without the extension to use as a folder name
            name = mat_file.stem
            std_name = standardize_name(name)

            # Create a directory for saving the images
            mask_output_folder = output_folder / std_name
            mask_output_folder.mkdir(exist_ok=True)

            # Load the .mat file
            logger.info(f"Processing {mat_file}")
            mat_data = scipy.io.loadmat(str(mat_file))

            # Extract the image data from 'predicted'
            if 'predicted' not in mat_data:
                logger.warning(f"'predicted' key not found in {mat_file}. Available keys: {list(mat_data.keys())}")
                continue

            image_data = mat_data['predicted']

            # Log shape and data information for debugging
            logger.debug(f"Image data shape: {image_data.shape}, dtype: {image_data.dtype}")
            logger.debug(f"Value range: {image_data.min()} to {image_data.max()}")

            # Loop through all images and save each with a unique name
            for i in range(image_data.shape[0]):
                # Normalize and convert to uint8
                img = (image_data[i] * 255).astype(np.uint8)
                image = Image.fromarray(img)

                # Save each image with a unique name in the output folder
                output_image_path = mask_output_folder / f'frame_{i+1:04d}.png'
                image.save(str(output_image_path))

            logger.info(f"Saved {image_data.shape[0]} images from '{name}' to '{mask_output_folder}'")
            processed_count += 1

        except Exception as e:
            logger.error(f"Error processing {mat_file}: {str(e)}")

    logger.info(f"Successfully processed {processed_count} out of {len(mat_files)} .mat files")
    return processed_count


def extract_frames_from_videos(video_folder: Path, frames_folder: Path, desired_fps: int = 25) -> int:
    """
    Extract frames from video files at a specified frame rate.

    Args:
        video_folder: Path to folder containing video files
        frames_folder: Path to save extracted frames
        desired_fps: Target frames per second

    Returns:
        Number of successfully processed videos
    """
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV (cv2) is not installed. Please install it with: pip install opencv-python")
        return 0

    # Create frames folder if it doesn't exist
    frames_folder.mkdir(parents=True, exist_ok=True)

    # Get all video files in the directory
    video_files = list(video_folder.glob("*.avi"))
    logger.info(f"Found {len(video_files)} video files in {video_folder}")

    processed_count = 0

    for video_file in video_files:
        try:
            # Get the base name of the video file to use as the folder name
            video_name = video_file.stem
            std_video_name = standardize_name(video_name)

            # Create a subfolder for the current video to store its frames
            video_frames_folder = frames_folder / std_video_name
            video_frames_folder.mkdir(exist_ok=True)

            # Open the video using OpenCV
            video_cap = cv2.VideoCapture(str(video_file))
            if not video_cap.isOpened():
                logger.error(f"Could not open video file: {video_file}")
                continue

            # Get the FPS of the video
            original_fps = video_cap.get(cv2.CAP_PROP_FPS)
            logger.info(f"Processing video: {video_name}, Original FPS: {original_fps}")

            # Calculate the interval between frames to match the desired FPS
            frame_interval = max(1, int(original_fps / desired_fps))
            logger.info(f"Extracting frames at an interval of every {frame_interval} frames to match {desired_fps} FPS")

            frame_number = 0
            extracted_frame_count = 0

            # Read frames from the video and save them
            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break  # Break if no frame is returned

                # Only save the frame if it's the correct interval
                if frame_number % frame_interval == 0:
                    # Save the frame as an image in the corresponding subfolder
                    frame_filename = video_frames_folder / f'frame_{extracted_frame_count + 1:04d}.png'
                    cv2.imwrite(str(frame_filename), frame)
                    extracted_frame_count += 1

                frame_number += 1

            # Release the video capture object
            video_cap.release()

            logger.info(f"Extracted {extracted_frame_count} frames from '{video_name}' to '{video_frames_folder}'")
            processed_count += 1

        except Exception as e:
            logger.error(f"Error processing video {video_file}: {str(e)}")

    logger.info(f"Successfully processed {processed_count} out of {len(video_files)} videos")
    return processed_count


def match_frames_with_masks(frames_folder: Path, masks_folder: Path, mask_prefix: str = "Mask_",
                           create_missing_frames: bool = False,
                           create_missing_masks: bool = False,
                           dry_run: bool = False,
                           flexible_matching: bool = True) -> Tuple[int, int]:
    """
    Match frame folders with corresponding mask folders and ensure consistent frame counts.

    Args:
        frames_folder: Path to folder containing frame folders
        masks_folder: Path to folder containing mask folders
        mask_prefix: Prefix used for mask folder names
        create_missing_frames: Whether to create frame folders for masks that don't have matching frames
        create_missing_masks: Whether to create mask folders for frames that don't have matching masks
        dry_run: If True, only report what would be done without making changes
        flexible_matching: If True, try more flexible matching strategies (case-insensitive, etc.)

    Returns:
        Tuple of (matching_pairs, non_matching_pairs)
    """
    # Get the list of subfolder names from both directories
    frame_subfolders = [f for f in frames_folder.iterdir() if f.is_dir()]
    mask_subfolders = [f for f in masks_folder.iterdir() if f.is_dir()]

    logger.info(f"Found {len(frame_subfolders)} frame folders and {len(mask_subfolders)} mask folders")

    # Standardize folder names for comparison
    frame_names = {standardize_name(f.name): f for f in frame_subfolders}
    mask_names = {standardize_name(f.name): f for f in mask_subfolders}

    # For flexible matching, create case-insensitive dictionaries
    if flexible_matching:
        frame_names_lower = {k.lower(): v for k, v in frame_names.items()}
        mask_names_lower = {k.lower(): v for k, v in mask_names.items()}

    matching_pairs = 0
    non_matching_pairs = 0
    created_folders = 0

    # Check for frames without matching masks
    for frame_std_name, frame_path in frame_names.items():
        matching_mask_std_name = f"{mask_prefix}{frame_std_name}"

        # Try standard matching first
        if matching_mask_std_name in mask_names:
            mask_path = mask_names[matching_mask_std_name]
            found_match = True
        # Try flexible matching if enabled
        elif flexible_matching and matching_mask_std_name.lower() in mask_names_lower:
            mask_path = mask_names_lower[matching_mask_std_name.lower()]
            found_match = True
            logger.info(f"Flexible match found: '{frame_path.name}' -> '{mask_path.name}'")
        else:
            found_match = False

        if not found_match:
            logger.warning(f"Frame folder '{frame_path.name}' does not have a matching mask folder")

            if create_missing_masks and not dry_run:
                # Create a new mask folder
                new_mask_folder = masks_folder / f"{mask_prefix}{frame_std_name}"
                try:
                    new_mask_folder.mkdir(exist_ok=True)
                    logger.info(f"Created empty mask folder '{new_mask_folder}' to match frame folder")
                    created_folders += 1
                    matching_pairs += 1
                except Exception as e:
                    logger.error(f"Error creating mask folder {new_mask_folder}: {str(e)}")
                    non_matching_pairs += 1
            else:
                non_matching_pairs += 1
        else:
            matching_pairs += 1

            # Count the number of mask images
            mask_files = list(mask_path.glob("*.png"))
            num_masks = len(mask_files)

            # Count the number of frames
            frame_files = list(frame_path.glob("*.png"))
            num_frames = len(frame_files)

            logger.info(f"Matched: '{frame_path.name}' ({num_frames} frames) with '{mask_path.name}' ({num_masks} masks)")

            # If the number of frames exceeds the number of masks, delete excess frames
            if num_frames > num_masks and not dry_run:
                excess_frame_count = num_frames - num_masks
                logger.info(f"Deleting {excess_frame_count} excess frames from '{frame_path.name}'")

                if dry_run:
                    logger.info(f"[DRY RUN] Would delete {excess_frame_count} excess frames from '{frame_path.name}'")
                else:
                    # Sort frames by name to ensure we delete from the end
                    frame_files.sort(key=lambda x: x.name)

                    # Delete excess frames (from the end)
                    for i in range(excess_frame_count):
                        try:
                            frame_to_delete = frame_files[-(i+1)]
                            frame_to_delete.unlink()
                        except Exception as e:
                            logger.error(f"Error deleting frame {frame_to_delete}: {str(e)}")

    # Check for masks without matching frames
    for mask_std_name, mask_path in mask_names.items():
        if mask_std_name.startswith(mask_prefix):
            frame_std_name = mask_std_name[len(mask_prefix):]

            # Try standard matching first
            if frame_std_name in frame_names:
                # Already counted in the previous loop
                continue
            # Try flexible matching if enabled
            elif flexible_matching and frame_std_name.lower() in frame_names_lower:
                logger.info(f"Flexible match found: '{mask_path.name}' -> '{frame_names_lower[frame_std_name.lower()].name}'")
                continue
            else:
                logger.warning(f"Mask folder '{mask_path.name}' does not have a matching frame folder")

                if create_missing_frames and not dry_run:
                    # Create a new frame folder
                    new_frame_folder = frames_folder / frame_std_name
                    try:
                        new_frame_folder.mkdir(exist_ok=True)
                        logger.info(f"Created empty frame folder '{new_frame_folder}' to match mask folder")
                        created_folders += 1
                        matching_pairs += 1
                    except Exception as e:
                        logger.error(f"Error creating frame folder {new_frame_folder}: {str(e)}")
                        non_matching_pairs += 1
                else:
                    non_matching_pairs += 1

    if dry_run:
        logger.info("[DRY RUN] No actual changes were made")

    if created_folders > 0:
        logger.info(f"Created {created_folders} new folders to match existing ones")

    logger.info(f"Summary: {matching_pairs} matching pairs, {non_matching_pairs} non-matching pairs")
    return matching_pairs, non_matching_pairs


def rename_folders_remove_spaces(directory: Path) -> int:
    """
    Rename folders by removing spaces from their names.

    Args:
        directory: Path to directory containing folders to rename

    Returns:
        Number of renamed folders
    """
    # Get the list of subfolder names
    subfolders = [f for f in directory.iterdir() if f.is_dir()]
    renamed_count = 0

    for subfolder in subfolders:
        # Generate the new name by removing spaces
        new_name = standardize_name(subfolder.name)

        # If the name has changed, rename the folder
        if subfolder.name != new_name:
            try:
                new_path = subfolder.parent / new_name
                subfolder.rename(new_path)
                logger.info(f"Renamed folder: '{subfolder}' -> '{new_path}'")
                renamed_count += 1
            except Exception as e:
                logger.error(f"Error renaming folder {subfolder}: {str(e)}")

    logger.info(f"Renamed {renamed_count} folders in {directory}")
    return renamed_count


def main(args):
    """
    Main function to run the medical image processing pipeline.

    Args:
        args: Command line arguments
    """
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    # Convert string paths to Path objects
    base_dir = Path(args.base_dir)
    mat_folder = Path(args.mat_folder) if args.mat_folder else base_dir / "archive/LV Ground-truth Segmentation Masks"
    video_folder = Path(args.video_folder) if args.video_folder else base_dir / "HMC-QU/A4C"

    # Setup output directories
    output_dirs = setup_directories(base_dir, ["Frames/A4C", "Masks"])
    frames_folder = output_dirs["Frames/A4C"]
    masks_folder = output_dirs["Masks"]

    # Process steps based on command line arguments
    if args.rename_only:
        logger.info("Running folder renaming only")
        rename_folders_remove_spaces(frames_folder)
        rename_folders_remove_spaces(masks_folder)
        return

    if args.process_masks:
        logger.info("Processing .mat files to extract masks")
        process_mat_files(mat_folder, masks_folder)

    if args.extract_frames:
        logger.info("Extracting frames from videos")
        extract_frames_from_videos(video_folder, frames_folder, args.fps)

    if args.match_frames:
        logger.info("Matching frames with masks and ensuring consistent frame counts")
        # First rename folders to standardize names
        rename_folders_remove_spaces(frames_folder)
        rename_folders_remove_spaces(masks_folder)
        # Then match frames with masks
        match_frames_with_masks(
            frames_folder,
            masks_folder,
            create_missing_frames=args.create_missing_frames,
            create_missing_masks=args.create_missing_masks,
            dry_run=args.dry_run,
            flexible_matching=args.flexible_matching
        )

    logger.info("Processing completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Image Processing Script")

    # Path and general options
    parser.add_argument("--base-dir", type=str, default=".",
                        help="Base directory for the project")
    parser.add_argument("--mat-folder", type=str, default=None,
                        help="Path to folder containing .mat files")
    parser.add_argument("--video-folder", type=str, default=None,
                        help="Path to folder containing video files")
    parser.add_argument("--fps", type=int, default=25,
                        help="Target frames per second for extraction")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose (debug) logging")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")

    # Processing steps
    step_group = parser.add_argument_group("Processing Steps")
    step_group.add_argument("--process-masks", action="store_true",
                        help="Process .mat files to extract masks")
    step_group.add_argument("--extract-frames", action="store_true",
                        help="Extract frames from videos")
    step_group.add_argument("--match-frames", action="store_true",
                        help="Match frames with masks and ensure consistent frame counts")
    step_group.add_argument("--rename-only", action="store_true",
                        help="Only rename folders to remove spaces")
    step_group.add_argument("--all", action="store_true",
                        help="Run all processing steps")

    # Matching options
    match_group = parser.add_argument_group("Matching Options")
    match_group.add_argument("--create-missing-frames", action="store_true",
                        help="Create empty frame folders for masks that don't have matching frames")
    match_group.add_argument("--create-missing-masks", action="store_true",
                        help="Create empty mask folders for frames that don't have matching masks")
    match_group.add_argument("--no-flexible-matching", action="store_false", dest="flexible_matching",
                        help="Disable flexible matching (case-insensitive, etc.)")
    match_group.add_argument("--force-match-all", action="store_true",
                        help="Force matching all folders by creating missing ones")

    args = parser.parse_args()

    # If --all is specified, enable all processing steps
    if args.all:
        args.process_masks = True
        args.extract_frames = True
        args.match_frames = True

    # If --force-match-all is specified, enable both create-missing options
    if args.force_match_all:
        args.create_missing_frames = True
        args.create_missing_masks = True

    # If no specific steps are enabled, show help
    if not any([args.process_masks, args.extract_frames, args.match_frames, args.rename_only]):
        parser.print_help()
    else:
        main(args)
