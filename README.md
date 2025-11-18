# Multispectral Image Alignment

This script aligns multispectral images from slightly misaligned cameras using phase correlation.

## Features

- **High-precision alignment**: Uses phase correlation with 100x upsampling for sub-pixel accuracy
- **Automatic cropping**: Crops all images to the overlapping valid region
- **Debug visualization**: Color-coded overlay to verify alignment quality
- **Flexible reference selection**: Choose any band as the reference (default: band 1)
- **Preserves data types**: Maintains original bit depth (8-bit or 16-bit TIFF)

## Usage

### Basic usage (align images in test_data):
```bash
python align.py test_data
```
Output will be saved to `test_data/aligned_output/` by default.

### Use a different band as reference (e.g., band 3):
```bash
python align.py test_data --reference 3
```

### Enable debug visualization:
```bash
python align.py test_data --debug
```

### Save to a specific output directory:
```bash
python align.py test_data --output my_aligned_images
```

### Process only specific image set:
```bash
python align.py test_data --base-name IMG_0002
```

### Align the thermal image (band 6):
```bash
python align.py test_data --align-thermal
```
This upsamples the thermal band to the reference resolution and uses Mutual Information (MI) for robust multi-modal translation alignment, then crops to the same valid region and saves to the output directory.

### Manual alignment for the thermal image (interactive):
```bash
python align.py test_data --align-thermal --manual-thermal --thermal-range-min 28924 --thermal-range-max 30754
```
- Arrow keys: move thermal (Up/Down/Left/Right)
- +/- keys: zoom in/out (small steps)
- Buttons: Up/Down/Left/Right/Zoom+/Zoom-/Confirm/Reset
- Enter: confirm and save
- The console prints the correction: `shift_y`, `shift_x`, and `scale`
- Only the thermal display is min–max rescaled (by the provided DN range) for visualization; saved files are unchanged in radiometry

### Apply known thermal correction (non-interactive):
```bash
python align.py test_data --align-thermal \
   --thermal-shift-y -72.000 --thermal-shift-x -43.000 --thermal-scale 1.17258
```
- Applies the provided translation and scale to the thermal band, saves the cropped, aligned thermal TIFF.
- Combine with `--debug` and optional `--thermal-range-min/max` to visually verify overlay.
- Coordinate convention: positive `shift_y` moves thermal down; positive `shift_x` moves thermal right.

### Maximum precision alignment:
```bash
python align.py test_data --upsample 200 --interpolation lanczos
```
Higher upsample values (up to 1000) give better sub-pixel precision but are slower.

## How it works

1. **Image loading**: Loads all images (first 5 from each set) and normalizes them
2. **Shift detection**: Uses phase cross-correlation to detect sub-pixel shifts between each image and the reference
3. **Alignment**: Applies translational shifts using high-quality interpolation (cubic by default)
4. **Cropping**: Calculates the valid region present in all images and crops to that region
5. **Saving**: Saves aligned images to output directory (default: `input_dir/aligned_output/`)

## Maximizing Alignment Precision

The script uses several techniques for maximum precision:

1. **Phase Cross-Correlation**: Instead of feature matching, uses Fourier-based phase correlation which is:
   - More accurate for translational shifts
   - Robust to intensity differences
   - Works well with texture (like wheat fields)

2. **Sub-pixel Upsampling**: Default factor of 100 means 0.01 pixel precision
   - Use `--upsample 200` for 0.005 pixel precision
   - Use `--upsample 500` or `1000` for ultimate precision (slower)
   
3. **High-Quality Interpolation**:
   - `--interpolation cubic` (default): Good balance of speed and quality
   - `--interpolation lanczos`: Highest quality, slightly slower
   - `--interpolation linear`: Fastest, lower quality

4. **Tips for best results**:
   - Use the band with the most texture/contrast as reference
   - Ensure images are properly exposed (not over/under-exposed)
   - For wheat, green or red bands typically work best as reference
   - The script preserves the full precision throughout (float32 operations)

## Debug Mode

When `--debug` is enabled, the script displays:
- **Color overlay**: Each band is assigned a different color (Red, Green, Blue, Cyan, Magenta, Yellow)
  - Perfect alignment = uniform colors without color fringing
  - Misalignment = visible color edges
- **Individual bands**: Grid view of all aligned bands

## Output

- Aligned images are saved to `aligned_output/` subdirectory by default
- Files keep their original names for easy tracking
- All images are cropped to the same size to ensure perfect pixel-to-pixel alignment
- Original bit depth (8-bit or 16-bit) is preserved

## Requirements

- Python 3.7+
- numpy
- opencv-python
- scikit-image
- matplotlib
 - SimpleITK

Install with:
```bash
pip install -r requirements.txt
```
