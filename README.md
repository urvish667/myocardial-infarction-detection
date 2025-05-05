# Medical Image Processor

A simple tool for organizing and processing medical images from heart ultrasound videos.

## What This Script Does

This script helps you organize and process medical images, specifically echocardiogram (heart ultrasound) videos and their corresponding segmentation masks. Here's what it does in simple terms:

### Main Tasks

1. **Extract Frames from Videos**:
   - Takes video files (.avi) of heart ultrasounds
   - Extracts individual image frames at a specific rate (like 25 frames per second)
   - Saves these frames as separate image files in organized folders

2. **Process Mask Files**:
   - Takes special files (.mat) that contain segmentation masks (outlines of heart structures)
   - Extracts the mask images from these files
   - Saves them as regular image files in organized folders

3. **Match Frames with Masks**:
   - Makes sure each video frame has a corresponding mask image
   - Organizes them in matching folders with consistent naming
   - Ensures the number of frames matches the number of masks

4. **Fix Common Problems**:
   - Standardizes folder names (removes spaces, etc.)
   - Creates missing folders if needed
   - Adjusts frame counts to match mask counts

### Why It's Useful

This organization is essential for:
- Training machine learning models that need paired images and masks
- Ensuring data consistency for medical image analysis
- Making it easier to work with large datasets of medical images

## Requirements

- Python 3.6+
- Required packages:
  - numpy
  - scipy
  - Pillow (PIL)
  - opencv-python (for video processing)

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use It

You can run the script with different options depending on what you need:

### Basic Usage

```bash
python medical_image_processor.py --all
```

### Common Tasks

1. **Process everything at once**:
   ```bash
   python medical_image_processor.py --all
   ```

2. **Just extract frames from videos**:
   ```bash
   python medical_image_processor.py --extract-frames
   ```

3. **Just process mask files**:
   ```bash
   python medical_image_processor.py --process-masks
   ```

4. **Match existing frames and masks**:
   ```bash
   python medical_image_processor.py --match-frames
   ```

5. **Fix missing folders**:
   ```bash
   python medical_image_processor.py --match-frames --force-match-all
   ```

6. **Preview changes without making them**:
   ```bash
   python medical_image_processor.py --match-frames --force-match-all --dry-run
   ```

## Output Structure

The script creates the following directory structure:

```
base_dir/
├── Frames/
│   └── A4C/
│       ├── ES0001_4CH_1/
│       │   ├── frame_0001.png
│       │   ├── frame_0002.png
│       │   └── ...
│       └── ...
└── Masks/
    ├── Mask_ES0001_4CH_1/
    │   ├── frame_0001.png
    │   ├── frame_0002.png
    │   └── ...
    └── ...
```

## Advanced Options

For more advanced usage, the script provides many options:

### Path and General Options
- `--base-dir PATH`: Base directory for the project (default: current directory)
- `--mat-folder PATH`: Path to folder containing .mat files (default: "archive/LV Ground-truth Segmentation Masks")
- `--video-folder PATH`: Path to folder containing video files (default: "HMC-QU/A4C")
- `--fps N`: Target frames per second for extraction (default: 25)
- `--verbose`, `-v`: Enable verbose (debug) logging
- `--dry-run`: Show what would be done without making changes

### Processing Steps
- `--process-masks`: Process .mat files to extract masks
- `--extract-frames`: Extract frames from videos
- `--match-frames`: Match frames with masks and ensure consistent frame counts
- `--rename-only`: Only rename folders to remove spaces
- `--all`: Run all processing steps

### Matching Options
- `--create-missing-frames`: Create empty frame folders for masks that don't have matching frames
- `--create-missing-masks`: Create empty mask folders for frames that don't have matching masks
- `--no-flexible-matching`: Disable flexible matching (case-insensitive, etc.)
- `--force-match-all`: Force matching all folders by creating missing ones
