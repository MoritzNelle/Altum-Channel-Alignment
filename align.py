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
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class AlignmentResult:
    reference_full: np.ndarray
    crop_region: Tuple[int, int, int, int]
    output_dir: Path
    cropped_images: List[np.ndarray]
    cropped_names: List[str]
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button
from skimage.registration import phase_cross_correlation
import SimpleITK as sitk


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


def resize_to_shape(image: np.ndarray, target_shape: Tuple[int, int], interpolation: str = 'cubic') -> np.ndarray:
    """
    Resize a single-channel image to the target (height, width).
    """
    interp_flags = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4,
        'nearest': cv2.INTER_NEAREST,
    }
    flags = interp_flags.get(interpolation, cv2.INTER_CUBIC)
    return cv2.resize(image, (int(target_shape[1]), int(target_shape[0])), interpolation=flags)


def register_translation_mi(reference: np.ndarray, moving: np.ndarray, 
                            shrink_factors=(4, 2, 1), smoothing_sigmas=(2, 1, 0),
                            number_of_bins: int = 50) -> Tuple[float, float]:
    """
    Estimate pure translation between reference and moving using Mutual Information (multi-modal robust).

    Returns (shift_y, shift_x) to align moving to reference.
    """
    # SimpleITK expects images with origin/spacing; convert from numpy (float32)
    ref_itk = sitk.GetImageFromArray(reference.astype(np.float32))
    mov_itk = sitk.GetImageFromArray(moving.astype(np.float32))

    transform = sitk.TranslationTransform(2)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=number_of_bins)
    registration.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                         minStep=1e-4,
                                                         numberOfIterations=300,
                                                         relaxationFactor=0.5)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetShrinkFactorsPerLevel(shrink_factors)
    registration.SetSmoothingSigmasPerLevel(smoothing_sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(transform, inPlace=False)

    final_transform = registration.Execute(ref_itk, mov_itk)
    # Extract parameters; for pure TranslationTransform, params are (offset_x, offset_y)
    try:
        params = tuple(final_transform.GetParameters())
    except Exception:
        params = tuple(sitk.Transform(final_transform).GetParameters())
    if len(params) >= 2:
        offset_x, offset_y = params[0], params[1]
    else:
        # Fallback, no movement
        offset_x, offset_y = 0.0, 0.0
    # Convert to (shift_y, shift_x)
    return float(offset_y), float(offset_x)


def align_and_save_thermal(reference_full: np.ndarray,
                           crop_region: Tuple[int, int, int, int],
                           thermal_path: Path,
                           output_dir: Path,
                           interpolation: str = 'cubic',
                           debug: bool = False) -> Tuple[np.ndarray, np.dtype]:
    """
    Align thermal image (lower resolution) to the reference image using MI and save cropped result.

    reference_full: reference image BEFORE cropping
    crop_region: (top, bottom, left, right) used on the aligned visible bands
    thermal_path: path to the thermal TIFF (image 6)
    output_dir: where to save the aligned thermal TIFF
    """
    print("\nProcessing thermal image (band 6)...")
    thermal_orig = cv2.imread(str(thermal_path), cv2.IMREAD_UNCHANGED)
    if thermal_orig is None:
        raise ValueError(f"Could not load thermal image: {thermal_path}")

    orig_dtype = thermal_orig.dtype

    # Normalize for processing
    if thermal_orig.dtype == np.uint16:
        thermal_proc = (thermal_orig.astype(np.float32) / 65535.0)
    elif thermal_orig.dtype == np.uint8:
        thermal_proc = (thermal_orig.astype(np.float32) / 255.0)
    else:
        thermal_proc = thermal_orig.astype(np.float32)

    # Upsample thermal to reference size
    ref_h, ref_w = reference_full.shape
    therm_h, therm_w = thermal_proc.shape[:2]
    print(f"  Thermal original size: {therm_h}x{therm_w}, upsampling to {ref_h}x{ref_w}")
    thermal_up = resize_to_shape(thermal_proc, (ref_h, ref_w), interpolation)

    # MI-based translation registration
    print("  Estimating translation via Mutual Information...")
    ty, tx = register_translation_mi(reference_full, thermal_up)
    print(f"  Thermal shift (y, x): ({ty:.3f}, {tx:.3f})")

    # Apply shift to upsampled thermal
    thermal_up_aligned = apply_shift(thermal_up, ty, tx, interpolation)

    # Crop to common valid region
    top, bottom, left, right = crop_region
    thermal_cropped = thermal_up_aligned[top:bottom, left:right]

    # Optional stand-alone overlay for thermal-only debug (kept but not used in combined view)
    if debug:
        print("  Showing thermal alignment overlay (reference in green, thermal in magenta)...")
        comp = np.zeros((bottom-top, right-left, 3), dtype=np.float32)
        ref_cropped = reference_full[top:bottom, left:right]
        comp[:, :, 1] = ref_cropped
        comp[:, :, 0] = thermal_cropped
        comp[:, :, 2] = thermal_cropped
        plt.figure(figsize=(8, 6))
        plt.imshow(np.clip(comp, 0, 1))
        plt.title("Thermal vs Reference Overlay")
        plt.axis('off')
        plt.show()

    # Convert back to original dtype and save
    if orig_dtype == np.uint16:
        out_img = np.clip(thermal_cropped * 65535.0, 0, 65535).astype(np.uint16)
    elif orig_dtype == np.uint8:
        out_img = np.clip(thermal_cropped * 255.0, 0, 255).astype(np.uint8)
    else:
        out_img = thermal_cropped

    out_path = output_dir / thermal_path.name
    cv2.imwrite(str(out_path), out_img)
    print(f"  Saved aligned thermal: {out_path.name}")

    # Return the cropped thermal in float32 [0,1] for visualization purposes
    return thermal_cropped.astype(np.float32), orig_dtype


def manual_align_and_save_thermal(reference_full: np.ndarray,
                                  crop_region: Tuple[int, int, int, int],
                                  thermal_path: Path,
                                  output_dir: Path,
                                  base_images: List[np.ndarray],
                                  base_names: List[str],
                                  interpolation: str = 'cubic',
                                  thermal_range_dn: Optional[Tuple[float, float]] = None) -> Tuple[float, float, float]:
    """
    Manual interactive alignment of the thermal image using arrow keys/buttons for translation and zoom in/out.
    Returns (shift_y, shift_x, scale) applied (in reference pixel units), saves aligned thermal on confirm.
    """
    print("\nManual thermal alignment UI starting...\n")
    # Load thermal
    thermal_orig = cv2.imread(str(thermal_path), cv2.IMREAD_UNCHANGED)
    if thermal_orig is None:
        raise ValueError(f"Could not load thermal image: {thermal_path}")
    orig_dtype = thermal_orig.dtype
    # Normalize for processing
    if thermal_orig.dtype == np.uint16:
        thermal_proc = thermal_orig.astype(np.float32) / 65535.0
        div = 65535.0
    elif thermal_orig.dtype == np.uint8:
        thermal_proc = thermal_orig.astype(np.float32) / 255.0
        div = 255.0
    else:
        thermal_proc = thermal_orig.astype(np.float32)
        div = 1.0

    ref_h, ref_w = reference_full.shape
    thermal_up = resize_to_shape(thermal_proc, (ref_h, ref_w), interpolation)
    top, bottom, left, right = crop_region
    disp_h, disp_w = bottom - top, right - left

    # Precompute background (reference in green channel)
    def build_composite(therm_img_cropped: np.ndarray, enabled_mask: List[bool]) -> np.ndarray:
        # Build composite from enabled base cropped layers and thermal cropped layer
        comp = np.zeros((disp_h, disp_w, 3), dtype=np.float32)
        # Colors reused from visualize function
        colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [0, 1, 1],  # Cyan
            [1, 0, 1],  # Magenta
            [1, 1, 0],  # Yellow
        ]
        # Add base bands
        for i, img in enumerate(base_images):
            if not enabled_mask[i]:
                continue
            color = colors[i % len(colors)]
            for c in range(3):
                comp[:, :, c] += img * color[c]
        # Thermal display scaling
        if thermal_range_dn is not None:
            rmin, rmax = thermal_range_dn
            if rmax > rmin:
                tmin = rmin / div
                tmax = rmax / div
                therm_disp = np.clip((therm_img_cropped - tmin) / max(1e-8, (tmax - tmin)), 0, 1)
            else:
                therm_disp = np.clip(therm_img_cropped, 0, 1)
        else:
            # fallback: min-max scale of current thermal layer
            mn, mx = float(np.min(therm_img_cropped)), float(np.max(therm_img_cropped))
            therm_disp = (therm_img_cropped - mn) / max(1e-8, (mx - mn)) if mx > mn else np.zeros_like(therm_img_cropped)
        # Thermal in magenta (R+B) if enabled (last checkbox)
        if enabled_mask[len(base_images)]:
            comp[:, :, 0] += therm_disp
            comp[:, :, 2] += therm_disp
        return np.clip(comp, 0, 1)

    def warp(image: np.ndarray, scale: float, ty: float, tx: float) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, 0, scale)  # 2x3
        M[0, 2] += tx
        M[1, 2] += ty
        return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Interactive state
    ty, tx, scale = 0.0, 0.0, 1.0
    step = 1.0
    zoom_step = 0.01

    enabled = [True] * (len(base_images) + 1)  # base bands + thermal
    warped = warp(thermal_up, scale, ty, tx)
    warped_cropped = warped[top:bottom, left:right]
    comp = build_composite(warped_cropped, enabled)

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    fig.subplots_adjust(left=0.35)
    im = ax.imshow(comp)
    ax.set_title("Manual thermal alignment (arrows=move, +/-=zoom, Enter=confirm, R=reset)")
    ax.axis('off')

    # Buttons
    ax_up = plt.axes((0.80, 0.05, 0.05, 0.05))
    ax_down = plt.axes((0.80, 0.00, 0.05, 0.05))
    ax_left = plt.axes((0.75, 0.00, 0.05, 0.05))
    ax_right = plt.axes((0.85, 0.00, 0.05, 0.05))
    ax_zoom_in = plt.axes((0.92, 0.05, 0.06, 0.05))
    ax_zoom_out = plt.axes((0.92, 0.00, 0.06, 0.05))
    ax_confirm = plt.axes((0.02, 0.02, 0.12, 0.06))
    ax_reset = plt.axes((0.16, 0.02, 0.10, 0.06))

    b_up = Button(ax_up, 'Up')
    b_down = Button(ax_down, 'Down')
    b_left = Button(ax_left, 'Left')
    b_right = Button(ax_right, 'Right')
    b_zoom_in = Button(ax_zoom_in, 'Zoom +')
    b_zoom_out = Button(ax_zoom_out, 'Zoom -')
    b_confirm = Button(ax_confirm, 'Confirm')
    b_reset = Button(ax_reset, 'Reset')

    # Checkboxes for toggling base layers and thermal
    labels = list(base_names) + [thermal_path.name + " (Thermal)"]
    rax = plt.axes((0.02, 0.12, 0.30, 0.80))
    checks = CheckButtons(rax, labels, enabled)

    def on_toggle(label):
        try:
            idx = labels.index(label)
        except ValueError:
            return
        enabled[idx] = not enabled[idx]
        redraw()

    checks.on_clicked(on_toggle)

    def redraw():
        nonlocal warped, warped_cropped
        warped = warp(thermal_up, scale, ty, tx)
        warped_cropped = warped[top:bottom, left:right]
        im.set_data(build_composite(warped_cropped, enabled))
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal ty, tx, scale
        if event.key in ('up', 'uparrow'):
            ty -= step
        elif event.key in ('down', 'downarrow'):
            ty += step
        elif event.key in ('left', 'leftarrow'):
            tx -= step
        elif event.key in ('right', 'rightarrow'):
            tx += step
        elif event.key in ('+', 'equal'):
            scale *= (1.0 + zoom_step)
        elif event.key in ('-', 'underscore'):
            scale /= (1.0 + zoom_step)
        elif event.key in ('r', 'R'):
            ty, tx, scale = 0.0, 0.0, 1.0
        elif event.key in ('enter',):
            plt.close(fig)
            return
        redraw()

    def on_up(event):
        nonlocal ty
        ty -= step; redraw()
    def on_down(event):
        nonlocal ty
        ty += step; redraw()
    def on_left(event):
        nonlocal tx
        tx -= step; redraw()
    def on_right(event):
        nonlocal tx
        tx += step; redraw()
    def on_zoom_in(event):
        nonlocal scale
        scale *= (1.0 + zoom_step); redraw()
    def on_zoom_out(event):
        nonlocal scale
        scale /= (1.0 + zoom_step); redraw()
    def on_confirm(event):
        plt.close(fig)

    b_up.on_clicked(on_up)
    b_down.on_clicked(on_down)
    b_left.on_clicked(on_left)
    b_right.on_clicked(on_right)
    b_zoom_in.on_clicked(on_zoom_in)
    b_zoom_out.on_clicked(on_zoom_out)
    b_confirm.on_clicked(on_confirm)
    def on_reset(event):
        nonlocal ty, tx, scale
        ty, tx, scale = 0.0, 0.0, 1.0
        redraw()
    b_reset.on_clicked(on_reset)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    # After confirm: apply final transform and save
    thermal_up_aligned = warp(thermal_up, scale, ty, tx)
    top, bottom, left, right = crop_region
    thermal_cropped = thermal_up_aligned[top:bottom, left:right]
    # Save
    if orig_dtype == np.uint16:
        out_img = np.clip(thermal_cropped * 65535.0, 0, 65535).astype(np.uint16)
    elif orig_dtype == np.uint8:
        out_img = np.clip(thermal_cropped * 255.0, 0, 255).astype(np.uint8)
    else:
        out_img = thermal_cropped
    out_path = output_dir / thermal_path.name
    cv2.imwrite(str(out_path), out_img)
    print(f"Manual alignment confirmed. Thermal correction: shift_y={ty:.3f}, shift_x={tx:.3f}, scale={scale:.5f}")
    print(f"Saved aligned thermal: {out_path}")
    return ty, tx, scale


def apply_fixed_thermal_correction(reference_full: np.ndarray,
                                   crop_region: Tuple[int, int, int, int],
                                   thermal_path: Path,
                                   output_dir: Path,
                                   shift_y: float,
                                   shift_x: float,
                                   scale: float,
                                   interpolation: str = 'cubic') -> Tuple[np.ndarray, np.dtype]:
    """
    Apply a user-provided thermal correction (translation + scale) and save the cropped result.

    Returns (thermal_cropped_float32, original_dtype) for visualization.
    """
    print("\nApplying fixed thermal correction (band 6)...")
    thermal_orig = cv2.imread(str(thermal_path), cv2.IMREAD_UNCHANGED)
    if thermal_orig is None:
        raise ValueError(f"Could not load thermal image: {thermal_path}")

    orig_dtype = thermal_orig.dtype

    # Normalize for processing
    if orig_dtype == np.uint16:
        thermal_proc = thermal_orig.astype(np.float32) / 65535.0
    elif orig_dtype == np.uint8:
        thermal_proc = thermal_orig.astype(np.float32) / 255.0
    else:
        thermal_proc = thermal_orig.astype(np.float32)

    # Upsample to reference size
    ref_h, ref_w = reference_full.shape
    therm_up = resize_to_shape(thermal_proc, (ref_h, ref_w), interpolation)

    # Build affine (scale then translate)
    def warp(image: np.ndarray, sc: float, ty: float, tx: float) -> np.ndarray:
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, 0, sc)
        M[0, 2] += tx
        M[1, 2] += ty
        interp_flags = {
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
            'nearest': cv2.INTER_NEAREST,
        }
        flags = interp_flags.get(interpolation, cv2.INTER_CUBIC)
        return cv2.warpAffine(image, M, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    thermal_up_aligned = warp(therm_up, scale, shift_y, shift_x)

    # Crop
    top, bottom, left, right = crop_region
    thermal_cropped = thermal_up_aligned[top:bottom, left:right]

    # Convert back to original dtype and save
    if orig_dtype == np.uint16:
        out_img = np.clip(thermal_cropped * 65535.0, 0, 65535).astype(np.uint16)
    elif orig_dtype == np.uint8:
        out_img = np.clip(thermal_cropped * 255.0, 0, 255).astype(np.uint8)
    else:
        out_img = thermal_cropped

    out_path = output_dir / thermal_path.name
    cv2.imwrite(str(out_path), out_img)
    print(f"  Applied manual thermal correction: shift_y={shift_y:.3f}, shift_x={shift_x:.3f}, scale={scale:.5f}")
    print(f"  Saved aligned thermal: {out_path.name}")

    return thermal_cropped.astype(np.float32), orig_dtype


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


def visualize_alignment(images: List[np.ndarray], names: List[str], title: str = "Alignment Check",
                        rescale_indices: Optional[set] = None,
                        rescale_map: Optional[Dict[int, Tuple[float, float]]] = None,
                        enable_layer_toggle: bool = False):
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
    
    # Prepare images for display (optionally per-image min-max rescale)
    if rescale_indices is None:
        rescale_indices = set()
    if rescale_map is None:
        rescale_map = {}
    display_images: List[np.ndarray] = []
    eps = 1e-8
    for idx, img in enumerate(images):
        if idx in rescale_map:
            rmin, rmax = rescale_map[idx]
            if rmax - rmin > eps:
                disp = (img - rmin) / (rmax - rmin)
            else:
                disp = np.zeros_like(img)
        elif idx in rescale_indices:
            imin = float(np.min(img))
            imax = float(np.max(img))
            if imax - imin > eps:
                disp = (img - imin) / (imax - imin)
            else:
                disp = np.zeros_like(img)
        else:
            disp = np.clip(img, 0, 1)
        display_images.append(disp.astype(np.float32))

    # Helper to build composite from enabled layers
    def build_composite(enabled_mask: List[bool]) -> np.ndarray:
        height, width = display_images[0].shape
        comp = np.zeros((height, width, 3), dtype=np.float32)
        for i, img in enumerate(display_images):
            if not enabled_mask[i]:
                continue
            color = colors[i % len(colors)]
            for c in range(3):
                comp[:, :, c] += img * color[c]
        return np.clip(comp, 0, 1)

    enabled = [True] * len(display_images)

    # Create figure: leave space on the left for checkboxes if toggling is enabled
    if enable_layer_toggle:
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        fig.subplots_adjust(left=0.35)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    composite = build_composite(enabled)
    im = ax.imshow(composite)
    ax.set_title(f"{title}\n(Color overlay)")
    ax.axis('off')

    if enable_layer_toggle:
        # Checkbuttons for toggling layers
        # Build labels with color names
        color_names = ['Red', 'Green', 'Blue', 'Cyan', 'Magenta', 'Yellow']
        labels = [f"{names[i]}: {color_names[i % 6]}" for i in range(len(display_images))]
        rax = fig.add_axes((0.05, 0.2, 0.25, 0.6))
        check = CheckButtons(rax, labels, enabled)

        def on_toggle(label):
            try:
                idx = labels.index(label)
            except ValueError:
                return
            enabled[idx] = not enabled[idx]
            im.set_data(build_composite(enabled))
            fig.canvas.draw_idle()

        check.on_clicked(on_toggle)

    plt.show()


def align_images(image_paths: List[Path], reference_idx: int = 0, debug: bool = False, 
                output_dir: Optional[Path] = None, upsample_factor: int = 100,
                warp_interpolation: str = 'cubic') -> AlignmentResult:
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
        if img_original is None:
            raise ValueError(f"Could not load image: {path}")
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
    print(f"\nApplying shifts (interpolation: {warp_interpolation})...")
    aligned_images = []
    
    for i, (img, (shift_y, shift_x), path) in enumerate(zip(images, shifts, image_paths)):
        if i == reference_idx:
            print(f"  {path.name}: No shift needed")
            aligned_images.append(img)
        else:
            print(f"  {path.name}: Applying shift ({shift_y:.3f}, {shift_x:.3f})")
            aligned = apply_shift(img, shift_y, shift_x, interpolation=warp_interpolation)
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
    
    # Debug visualization (only for the five MS bands)
    if debug:
        print("\nShowing alignment visualization...")
        visualize_alignment(cropped_images, [p.name for p in image_paths], enable_layer_toggle=True)
    
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

    # Return info for potential thermal alignment
    return AlignmentResult(reference_full=reference_img,
                           crop_region=(top, bottom, left, right),
                           output_dir=output_dir,
                           cropped_images=cropped_images,
                           cropped_names=[p.name for p in image_paths])


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
    parser.add_argument("--align-thermal", action="store_true",
                       help="Also align the 6th thermal image (auto MI-based unless --manual-thermal)")
    parser.add_argument("--manual-thermal", action="store_true",
                       help="Use manual interactive alignment for the thermal image (arrows/buttons/zoom)")
    parser.add_argument("--thermal-shift-y", type=float, default=None,
                       help="Apply a fixed manual thermal shift in pixels (Y, +down)")
    parser.add_argument("--thermal-shift-x", type=float, default=None,
                       help="Apply a fixed manual thermal shift in pixels (X, +right)")
    parser.add_argument("--thermal-scale", type=float, default=None,
                       help="Apply a fixed manual thermal scale (1.0 = no scale)")
    parser.add_argument("--thermal-range-min", type=float, default=None,
                       help="Thermal display min (raw DN). Applied only in debug visualization.")
    parser.add_argument("--thermal-range-max", type=float, default=None,
                       help="Thermal display max (raw DN). Applied only in debug visualization.")
    
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
            # If thermal will be aligned, defer combined debug until after thermal is ready
            visible_debug = args.debug and not args.align_thermal
            result = align_images(paths_to_process, reference_idx=reference_idx, 
                        debug=visible_debug, output_dir=output_dir, 
                        upsample_factor=args.upsample,
                        warp_interpolation=args.interpolation)
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Optionally process thermal (6th) image
        if args.align_thermal and len(paths) >= 6:
            thermal_path = paths[5]
            ref_full = result.reference_full
            crop_region = result.crop_region
            out_dir = result.output_dir
            try:
                # Precedence: fixed params > manual UI > automatic MI
                fixed_params_provided = (args.thermal_shift_y is not None
                                         or args.thermal_shift_x is not None
                                         or args.thermal_scale is not None)
                if fixed_params_provided:
                    ty = 0.0 if args.thermal_shift_y is None else float(args.thermal_shift_y)
                    tx = 0.0 if args.thermal_shift_x is None else float(args.thermal_shift_x)
                    scale = 1.0 if args.thermal_scale is None else float(args.thermal_scale)
                    thermal_cropped, thermal_dtype = apply_fixed_thermal_correction(
                        ref_full, crop_region, thermal_path, out_dir,
                        shift_y=ty, shift_x=tx, scale=scale,
                        interpolation=args.interpolation,
                    )
                    # Combined debug view if requested
                    if args.debug:
                        all_images = list(result.cropped_images) + [thermal_cropped]
                        all_names = list(result.cropped_names) + [thermal_path.name]
                        rescale_map = {}
                        # Thermal display range mapping (optional)
                        thermal_orig = cv2.imread(str(thermal_path), cv2.IMREAD_UNCHANGED)
                        if thermal_orig is not None:
                            tdtype = thermal_orig.dtype
                        else:
                            tdtype = thermal_dtype
                        if args.thermal_range_min is not None and args.thermal_range_max is not None:
                            if tdtype == np.uint16:
                                div = 65535.0
                            elif tdtype == np.uint8:
                                div = 255.0
                            else:
                                div = 1.0
                            tmin = float(args.thermal_range_min) / div
                            tmax = float(args.thermal_range_max) / div
                            if tmax > tmin:
                                rescale_map[len(all_images) - 1] = (tmin, tmax)
                            else:
                                print("Warning: thermal-range-max must be greater than thermal-range-min; ignoring fixed range.")
                        print("\nShowing combined alignment visualization (including thermal)...")
                        visualize_alignment(all_images, all_names, rescale_map=rescale_map, enable_layer_toggle=True)
                elif args.manual_thermal:
                    # Manual interactive alignment
                    trange = None
                    if args.thermal_range_min is not None and args.thermal_range_max is not None:
                        trange = (float(args.thermal_range_min), float(args.thermal_range_max))
                    ty, tx, scale = manual_align_and_save_thermal(
                        ref_full, crop_region, thermal_path, out_dir,
                        result.cropped_images, result.cropped_names,
                        interpolation=args.interpolation,
                        thermal_range_dn=trange
                    )
                else:
                    # Automatic MI-based alignment
                    thermal_cropped, thermal_dtype = align_and_save_thermal(ref_full, crop_region, thermal_path, out_dir,
                                           interpolation=args.interpolation,
                                           debug=False)
                    # Show combined debug view (bands 1-5 + thermal) if requested
                    if args.debug:
                        all_images = list(result.cropped_images) + [thermal_cropped]
                        all_names = list(result.cropped_names) + [thermal_path.name]
                        # If user specified a thermal display range, use it in raw DN units
                        rescale_map = {}
                        if args.thermal_range_min is not None and args.thermal_range_max is not None:
                            if thermal_dtype == np.uint16:
                                div = 65535.0
                            elif thermal_dtype == np.uint8:
                                div = 255.0
                            else:
                                div = 1.0
                            tmin = float(args.thermal_range_min) / div
                            tmax = float(args.thermal_range_max) / div
                            if tmax > tmin:
                                rescale_map[len(all_images) - 1] = (tmin, tmax)
                            else:
                                print("Warning: thermal-range-max must be greater than thermal-range-min; ignoring fixed range.")
                        print("\nShowing combined alignment visualization (including thermal)...")
                        visualize_alignment(all_images, all_names, rescale_map=rescale_map, enable_layer_toggle=True)
            except Exception as e:
                print(f"Error aligning thermal image for {base_name}: {e}")


if __name__ == "__main__":
    main()
