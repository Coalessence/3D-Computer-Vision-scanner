# 3D Computer Vision Scanner

A turntable-based 3D scanner built as part of a university project in geometric and 3D computer vision. This repository provides the code, documentation, and data to calibrate, drive a laser + camera setup, scan objects, and reconstruct 3D geometry.

## Overview

3D scanning is accomplished by combining a rotating turntable, a laser (planar line laser), and a camera. As the object rotates, the laser projects a line onto the surface, and the deformation of the line observed by the camera encodes depth. By sweeping many views and triangulating, a 3D point cloud or mesh is reconstructed.

This project was developed for the *Geometric and 3D Computer Vision* course at Ca'Foscari University of Venice Computer Science Department, and demonstrates:

* Camera calibration
* Laser pose estimation
* Turntable control and synchronization
* Point cloud reconstruction

---

## Features

* **Camera calibration** (intrinsics, distortion) via checkerboard images
* **Extrinsic calibration** to locate laser plane relative to camera
* Automated **turntable & scanning loop**
* Reconstruction of **3D points** from laser sweep
* Utilities and helper functions for IO, transformations, visualization
* Sample datasets & scanning results included


## Getting Started

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Coalessence/3D-Computer-Vision-scanner.git
   cd 3D-Computer-Vision-scanner
   ```

2. Install dependencies with pip.

3. Place calibration or scan images under the `G3DCV_turntable_scanner_data/` folder (or configure your own paths).

### Calibration

1. Capture multiple checkerboard images for intrinsic calibration.

2. Run:

   ```bash
   python calibration.py path/to/checkerboard_images/
   ```

3. For extrinsic / laser calibration:

   ```bash
   python calibration.py path/to/laser/calibration_images/
   ```

4. This will generate calibration parameters (camera matrix, distortion coefficients, laser plane parameters) saved to files.

### Scanning Workflow

Once calibration is done:

```bash
python scanner.py --config path/to/calib_params.json --out output_folder
```

This script will:

* Rotate the turntable step by step
* At each step, project the laser, capture an image
* Detect the laser line, triangulate depth points
* Aggregate the points into a full 3D point cloud

You can visualize the result using a point cloud viewer (e.g. via PyntCloud, Meshlab).

You can adjust parameters (e.g. number of steps, motor delay) in the scanner config or within `scanner.py`.


