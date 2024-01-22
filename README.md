# Visual Odometry Scripts

## Overview

This repository contains two Python scripts for visual odometry using different feature detection algorithms:

1. **VO.py**: Visual odometry using ORB. Please note that it is currently not fully operational.

2. **VO2.py**: Visual odometry using SIFT. This implementation is functional.

## Usage

Before running the scripts, make sure to configure the necessary paths by providing the following information:

- **Path to Data Sequence**: `<path_to_data_sequence>`
- **Path to Calibration File**: `<path_to_calib_file>`
- **Path to Ground Truth**: `<path_to_ground_truth>`
- **Path to Store Generated Data**: `<path_to_generated_data>`

### Example Usage for SIFT Odometry

```bash
# Visualize trajectory
evo_traj kitti VOPoses/SIFT_02.txt --ref=GTPoses/02.txt -p --plot_mode=xz

# Compute Absolute Pose Error (APE) and visualize results
evo_ape kitti GTPoses/02.txt VOPoses/SIFT_02.txt -va --plot --plot_mode xz --save_results results/SIFT02.zip
```

Replace the placeholder paths (`<...>`) with your specific file and directory paths.

## Important Note

- **VO.py Status**: The ORB-based visual odometry in `VO.py` is currently not fully functional.
- **VO2.py Status**: The SIFT-based visual odometry in `VO2.py` is functional and ready for use.

Feel free to experiment and contribute to the improvement of the ORB-based visual odometry implementation. If you encounter any issues or have suggestions, please submit them through the repository's issue tracker.

Happy coding!
