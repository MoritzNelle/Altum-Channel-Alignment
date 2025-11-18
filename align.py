"""
Multispectral Image Alignment Script
Aligns multispectral images from slightly misaligned cameras using phase correlation.
Designed for MicaSense Altum or similar multispectral camera systems.
"""

import numpy as np
import cv2
from pathlib import Path
import argparse
import re
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation


def find_image_sets(directory: Path) -> Dict[str, List[Path]]:
    """
    Find all image sets in the directory.
    Groups images by their base name (everything before the last underscore and number).
    
    Args:
        directory: Path to the directory containing images
        
    Returns:
        Dictionary mapping base names to lists of image paths
    """
    image_sets = {}
    
    # Find all TIF files
    for img_path in directory.glob("*.tif"):
        # Extract base name and number
        match = re.match(r"(.+)_(\d+)\.tif$", img_path.name)
        if match:
            base_name = match.group(1)
            number = int(match.group(2))
            
            if base_name not in image_sets:
                image_sets[base_name] = []
            
            image_sets[base_name].append((number, img_path))
    
    # Sort each set by number
    for base_name in image_sets:
        image_sets[base_name].sort(key=lambda x: x[0])
        image_sets[base_name] = [path for _, path in image_sets[base_name]]
    
    return image_sets


def load_image(path: Path) -> np.ndarray:
    """
    Load a TIFF image and normalize to float32.
    
    Args:
        path: Path to the image file
        
    Returns:
        Normalized image as float32 array
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    
    # Normalize to 0-1 range
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    
    return img


def compute_shift(reference: np.ndarray, target: np.ndarray, upsample_factor: int = 100) -> Tuple[float, float]:
    """
    Compute the shift between reference and target images using phase correlation.
    
    Args:
        reference: Reference image
        target: Image to align to reference
        upsample_factor: Upsampling factor for sub-pixel accuracy (higher = more precise but slower)
        
    Returns:
        Tuple of (shift_y, shift_x) in pixels
    """
    # Use phase cross-correlation for sub-pixel accuracy
    shift, error, diffphase = phase_cross_correlation(reference, target, upsample_factor=upsample_factor)
    
    print(f"  Detected shift: ({shift[0]:.3f}, {shift[1]:.3f}) pixels, error: {error:.6f}")
    
    return shift[0], shift[1]


def apply_shift(image: np.ndarray, shift_y: float, shift_x: float, interpolation: str = 'cubic') -> np.ndarray:
    """
    Apply a translational shift to an image.
    
    Args:
        image: Input image
        shift_y: Vertical shift in pixels
        shift_x: Horizontal shift in pixels
        interpolation: Interpolation method ('linear', 'cubic', or 'lanczos')
        
    Returns:
        Shifted image
    """
    # Create translation matrix
    M = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
    
    # Select interpolation method
    interp_flags = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    flags = interp_flags.get(interpolation, cv2.INTER_CUBIC)
    
    # Apply translation with high-quality interpolation
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), 
                            flags=flags, borderMode=cv2.BORDER_CONSTANT, 
                            borderValue=0)
    
    return shifted


def compute_valid_region(shifts: List[Tuple[float, float]], image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """
    Compute the valid region that is present in all aligned images.
    
    Args:
        shifts: List of (shift_y, shift_x) for each image
        image_shape: Shape of the images (height, width)
        
    Returns:
        Tuple of (top, bottom, left, right) pixel coordinates defining valid region
    """
    height, width = image_shape
    
    # Find the maximum shifts in each direction
    max_shift_up = max(0, max(shift_y for shift_y, _ in shifts))
    max_shift_down = max(0, -min(shift_y for shift_y, _ in shifts))
    max_shift_left = max(0, max(shift_x for _, shift_x in shifts))
    max_shift_right = max(0, -min(shift_x for _, shift_x in shifts))
    
    # Define valid region
    top = int(np.ceil(max_shift_up))
    bottom = int(height - np.ceil(max_shift_down))
    left = int(np.ceil(max_shift_left))
    right = int(width - np.ceil(max_shift_right))
    
    return top, bottom, left, right


def visualize_alignment(images: List[np.ndarray], names: List[str], title: str = "Alignment Check"):
    """
    Visualize aligned images by assigning different colors to each band.
    
    Args:
        images: List of aligned images (normalized 0-1)
        names: List of image names
        title: Title for the plot
    """
    # Define colors for up to 6 bands (RGB and CMY)
    colors = [
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
        [0, 1, 1],  # Cyan
        [1, 0, 1],  # Magenta
        [1, 1, 0],  # Yellow
    ]
    
    # Create RGB composite
    height, width = images[0].shape
    composite = np.zeros((height, width, 3), dtype=np.float32)
    
    for i, img in enumerate(images):
        color = colors[i % len(colors)]
        for c in range(3):
            composite[:, :, c] += img * color[c]
    
    # Normalize composite
    composite = np.clip(composite, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Show composite
    axes[0].imshow(composite)
    axes[0].set_title(f"{title}\n(Color overlay)")
    axes[0].axis('off')
    
    # Create legend
    legend_text = "\n".join([f"{names[i]}: {['Red', 'Green', 'Blue', 'Cyan', 'Magenta', 'Yellow'][i % 6]}" 
                            for i in range(len(images))])
    axes[0].text(0.02, 0.98, legend_text, transform=axes[0].transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
    
    # Show individual bands as grid
    axes[1].set_title("Individual Bands")
    axes[1].axis('off')
    
    # Create small thumbnails
    n_images = len(images)
    thumb_size = 150
    margin = 10
    
    rows = int(np.ceil(np.sqrt(n_images)))
    cols = int(np.ceil(n_images / rows))
    
    canvas_height = rows * thumb_size + (rows + 1) * margin
    canvas_width = cols * thumb_size + (cols + 1) * margin
    canvas = np.ones((canvas_height, canvas_width), dtype=np.float32)
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        # Resize image
        thumb = cv2.resize(img, (thumb_size, thumb_size))
        
        # Place in canvas
        y_start = row * thumb_size + (row + 1) * margin
        x_start = col * thumb_size + (col + 1) * margin
        canvas[y_start:y_start+thumb_size, x_start:x_start+thumb_size] = thumb
    
    axes[1].imshow(canvas, cmap='gray')
    
    plt.tight_layout()
    plt.show()


def align_images(image_paths: List[Path], reference_idx: int = 0, debug: bool = False, 
                output_dir: Path = None, upsample_factor: int = 100) -> None:
    """
    Align a set of multispectral images.
    
    Args:
        image_paths: List of paths to images to align (should be 5 images)
        reference_idx: Index of the reference image (default: 0)
        debug: Whether to show debug visualization
        output_dir: Output directory for aligned images (default: aligned_output)
        upsample_factor: Upsampling factor for sub-pixel accuracy (default: 100)
    """
    print(f"\n{'='*60}")
    print(f"Aligning image set: {image_paths[0].stem.rsplit('_', 1)[0]}")
    print(f"{'='*60}")
    
    # Validate inputs
    if reference_idx < 0 or reference_idx >= len(image_paths):
        raise ValueError(f"Reference index {reference_idx} out of range [0, {len(image_paths)-1}]")
    
    if output_dir is None:
        output_dir = image_paths[0].parent / "aligned_output"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load all images
    print("\nLoading images...")
    images = []
    original_dtypes = []
    original_ranges = []
    
    for path in image_paths:
        print(f"  Loading {path.name}")
        # Load original for dtype info
        img_original = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        original_dtypes.append(img_original.dtype)
        original_ranges.append((img_original.min(), img_original.max()))
        
        # Load normalized version for processing
        img = load_image(path)
        images.append(img)
        print(f"    Shape: {img.shape}, dtype: {img.dtype}, range: [{img.min():.4f}, {img.max():.4f}]")
    
    reference_img = images[reference_idx]
    print(f"\nUsing {image_paths[reference_idx].name} as reference")
    
    # Compute shifts for all images
    print("\nComputing shifts...")
    shifts = []
    
    for i, (img, path) in enumerate(zip(images, image_paths)):
        if i == reference_idx:
            print(f"  {path.name}: Reference image (no shift)")
            shifts.append((0.0, 0.0))
        else:
            print(f"  {path.name}:")
            shift_y, shift_x = compute_shift(reference_img, img, upsample_factor)
            shifts.append((shift_y, shift_x))
    
    # Apply shifts
    print(f"\nApplying shifts (interpolation: cubic)...")
    aligned_images = []
    
    for i, (img, (shift_y, shift_x), path) in enumerate(zip(images, shifts, image_paths)):
        if i == reference_idx:
            print(f"  {path.name}: No shift needed")
            aligned_images.append(img)
        else:
            print(f"  {path.name}: Applying shift ({shift_y:.3f}, {shift_x:.3f})")
            aligned = apply_shift(img, shift_y, shift_x, interpolation='cubic')
            aligned_images.append(aligned)
    
    # Compute valid region
    print("\nComputing valid region...")
    top, bottom, left, right = compute_valid_region(shifts, reference_img.shape)
    print(f"  Valid region: top={top}, bottom={bottom}, left={left}, right={right}")
    print(f"  Output size: {bottom-top}x{right-left} pixels")
    
    if bottom <= top or right <= left:
        raise ValueError("Invalid crop region - shifts may be too large!")
    
    # Crop to valid region
    print("\nCropping to valid region...")
    cropped_images = []
    for img in aligned_images:
        cropped = img[top:bottom, left:right]
        cropped_images.append(cropped)
    
    # Debug visualization
    if debug:
        print("\nShowing alignment visualization...")
        visualize_alignment(cropped_images, [p.name for p in image_paths])
    
    # Save aligned images
    print("\nSaving aligned images...")
    for i, (img, path, orig_dtype) in enumerate(zip(cropped_images, image_paths, original_dtypes)):
        output_path = output_dir / path.name
        
        # Convert back to original dtype
        if orig_dtype == np.uint16:
            img_save = (img * 65535).astype(np.uint16)
        elif orig_dtype == np.uint8:
            img_save = (img * 255).astype(np.uint8)
        else:
            img_save = img
        
        cv2.imwrite(str(output_path), img_save)
        print(f"  Saved {output_path.name}")
    
    print(f"\n{'='*60}")
    print("Alignment complete!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Align multispectral images from slightly misaligned cameras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Align images in test_data directory, using image 1 as reference
  python align.py test_data
  
  # Use image 3 as reference
  python align.py test_data --reference 3
  
  # Enable debug visualization
  python align.py test_data --debug
  
  # Specify output directory
  python align.py test_data --output aligned_output
        """
    )
    
    parser.add_argument("input_dir", type=str, 
                       help="Directory containing images to align")
    parser.add_argument("--reference", type=int, default=1,
                       help="Band number to use as reference (default: 1)")
    parser.add_argument("--debug", action="store_true",
                       help="Show debug visualization after alignment")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for aligned images (default: input_dir/aligned_output)")
    parser.add_argument("--base-name", type=str, default=None,
                       help="Process only images matching this base name (default: process all)")
    parser.add_argument("--upsample", type=int, default=100,
                       help="Upsampling factor for sub-pixel precision (default: 100, higher=more precise)")
    parser.add_argument("--interpolation", type=str, default="cubic",
                       choices=["linear", "cubic", "lanczos"],
                       help="Interpolation method for image warping (default: cubic)")
    
    args = parser.parse_args()
    
    # Parse paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    output_dir = Path(args.output) if args.output else None
    
    # Find image sets
    print("Scanning for image sets...")
    image_sets = find_image_sets(input_dir)
    
    if not image_sets:
        print("No image sets found in the directory")
        return
    
    print(f"Found {len(image_sets)} image set(s):")
    for base_name, paths in image_sets.items():
        print(f"  {base_name}: {len(paths)} images")
    
    # Filter by base name if specified
    if args.base_name:
        if args.base_name not in image_sets:
            print(f"Error: Base name '{args.base_name}' not found")
            return
        image_sets = {args.base_name: image_sets[args.base_name]}
    
    # Process each image set
    for base_name, paths in image_sets.items():
        # Only use first 5 images
        paths_to_process = paths[:5]
        
        if len(paths_to_process) < 2:
            print(f"Skipping {base_name}: need at least 2 images")
            continue
        
        # Convert reference from 1-indexed to 0-indexed
        reference_idx = args.reference - 1
        
        if reference_idx < 0 or reference_idx >= len(paths_to_process):
            print(f"Warning: Reference band {args.reference} out of range for {base_name}")
            print(f"         Using band 1 instead")
            reference_idx = 0
        
        try:
            align_images(paths_to_process, reference_idx=reference_idx, 
                        debug=args.debug, output_dir=output_dir, 
                        upsample_factor=args.upsample)
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
