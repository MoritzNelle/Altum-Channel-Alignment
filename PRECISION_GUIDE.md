# Alignment Precision Guide

## Current Setup

Your script now uses state-of-the-art alignment techniques with the following precision capabilities:

### Default Settings (Good for most cases)
- **Sub-pixel precision**: 0.01 pixels (1/100th of a pixel)
- **Interpolation**: Bicubic (high quality)
- **Speed**: Fast (~5 seconds for 5 images)

```bash
python align.py test_data
```

## Increasing Precision

### High Precision Mode
For critical applications where maximum accuracy is needed:

```bash
python align.py test_data --upsample 200 --interpolation lanczos
```
- **Sub-pixel precision**: 0.005 pixels (1/200th of a pixel)
- **Interpolation**: Lanczos (highest quality)
- **Speed**: Moderate (~10 seconds)

### Ultra-High Precision Mode
For research-grade alignment:

```bash
python align.py test_data --upsample 500
```
- **Sub-pixel precision**: 0.002 pixels (1/500th of a pixel)
- **Speed**: Slower (~20 seconds)

### Maximum Precision (Overkill Mode)
```bash
python align.py test_data --upsample 1000 --interpolation lanczos
```
- **Sub-pixel precision**: 0.001 pixels (1/1000th of a pixel)
- **Speed**: Slow (~40 seconds)

## What Limits Alignment Precision?

Even with perfect software, several factors limit real-world precision:

1. **Lens distortion** (~0.5-2 pixels): Different lenses have different distortions
2. **Atmospheric effects** (~0.1-0.5 pixels): Heat shimmer, turbulence
3. **Camera vibration** (~0.05-0.2 pixels): Mechanical vibrations during capture
4. **Sensor noise** (~0.01-0.05 pixels): Especially in low-light conditions
5. **Motion blur** (~0.5-1 pixels): If scene has moving elements

## Practical Recommendations

### For Vegetation Indices (NDVI, etc.)
- Default settings (--upsample 100) are sufficient
- Vegetation indices are robust to small misalignments
```bash
python align.py test_data
```

### For Change Detection
- Use high precision (--upsample 200)
```bash
python align.py test_data --upsample 200
```

### For Scientific Publications
- Use ultra-high precision with Lanczos interpolation
```bash
python align.py test_data --upsample 500 --interpolation lanczos
```

## Technical Details

### Phase Cross-Correlation
The script uses FFT-based phase correlation which:
- Operates in frequency domain for global optimization
- Is invariant to intensity changes
- Handles texture-rich scenes (like wheat) very well
- Is more accurate than feature-based methods for translational shifts

### Interpolation Methods

**Linear** (`--interpolation linear`)
- Fastest
- Good for large datasets where speed matters
- Adequate for most applications

**Cubic** (`--interpolation cubic`) [DEFAULT]
- Excellent quality/speed balance
- Recommended for most users
- Smooth results, minimal artifacts

**Lanczos** (`--interpolation lanczos`)
- Highest quality
- Best edge preservation
- Slightly slower
- Recommended for scientific work

## Verification

Always use `--debug` mode to visually verify alignment:

```bash
python align.py test_data --debug --upsample 200
```

Look for:
- ✅ No color fringing at edges (perfect alignment)
- ❌ Visible color separation (needs adjustment)

## Output Organization

```
test_data/
  ├── IMG_0002_1.tif          # Original images (preserved)
  ├── IMG_0002_2.tif
  ├── ...
  └── aligned_output/          # Aligned images (cropped to valid region)
      ├── IMG_0002_1.tif
      ├── IMG_0002_2.tif
      └── ...
```

Your original images are never overwritten!
