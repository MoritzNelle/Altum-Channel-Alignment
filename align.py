"""
Multispectral Image Alignment Script
Aligns multispectral images from slightly misaligned cameras using phase correlation.
Designed for MicaSense Altum or similar multispectral camera systems.
"""

# ===== RECOMMENDED COMMANDS =====
# 
# DRAFT MODE (50-100x faster, ultra-fast rough alignment):
# python align.py input --alignment-mode affine --edge-metrics --metrics-out metrics.csv --affine-tolerance 0.0001 --align-thermal --manual-thermal --upsample 1 --ecc-iterations 50 --mi-iterations 10
#
# FAST MODE (5-10x faster, excellent quality for most applications):
# python align.py input --alignment-mode affine --edge-metrics --metrics-out metrics.csv --affine-tolerance 0.0001 --align-thermal --manual-thermal --upsample 25 --ecc-iterations 600 --mi-iterations 150
#
# HIGH QUALITY MODE (slower, maximum precision for publication):
# python align.py input --alignment-mode affine --edge-metrics --metrics-out metrics.csv --affine-tolerance 0.0001 --align-thermal --manual-thermal --upsample 100 --ecc-iterations 2000 --mi-iterations 300
#
# Manual thermal only (fast):
# python align.py input --align-thermal --manual-thermal --upsample 25 --mi-iterations 150
# ================================

import numpy as np
import cv2
from pathlib import Path
import argparse
import re
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class AlignmentMetrics:
    """Stores alignment quality metrics for a single image."""
    image_name: str
    mode: str  # 'translation', 'affine', 'homography'
    rms_error: float  # Root mean square error after alignment
    correlation: float  # Cross-correlation coefficient with reference
    transform_params: Dict[str, Any]  # Mode-specific parameters
    

@dataclass
class AlignmentResult:
    reference_full: np.ndarray
    crop_region: Tuple[int, int, int, int]
    output_dir: Path
    cropped_images: List[np.ndarray]
    cropped_names: List[str]
    metrics: List[AlignmentMetrics]  # Per-image alignment quality
import csv
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


def compute_rms_error(reference: Optional[np.ndarray], aligned: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute root mean square error between reference and aligned image.
    
    Args:
        reference: Reference image
        aligned: Aligned image
        mask: Optional mask of valid pixels (1=valid, 0=invalid)
        
    Returns:
        RMS error (lower is better)
    """
    if reference is None:
        return float('inf')
    if mask is not None:
        valid = mask > 0.5
        if np.sum(valid) == 0:
            return float('inf')
        diff = (reference[valid] - aligned[valid]) ** 2
    else:
        diff = (reference - aligned) ** 2
    
    return float(np.sqrt(np.mean(diff)))


def compute_correlation(reference: Optional[np.ndarray], aligned: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """
    Compute normalized cross-correlation between reference and aligned image.
    
    Args:
        reference: Reference image
        aligned: Aligned image
        mask: Optional mask of valid pixels
        
    Returns:
        Correlation coefficient (1.0 = perfect match, 0.0 = no correlation)
    """
    if reference is None:
        return 0.0
    if mask is not None:
        valid = mask > 0.5
        if np.sum(valid) == 0:
            return 0.0
        ref = reference[valid]
        aln = aligned[valid]
    else:
        ref = reference.flatten()
        aln = aligned.flatten()
    
    # Normalize
    ref_mean = np.mean(ref)
    aln_mean = np.mean(aln)
    ref_std = np.std(ref)
    aln_std = np.std(aln)
    
    if ref_std < 1e-8 or aln_std < 1e-8:
        return 0.0
    
    ref_norm = (ref - ref_mean) / ref_std
    aln_norm = (aln - aln_mean) / aln_std
    
    corr = np.mean(ref_norm * aln_norm)
    return float(np.clip(corr, -1.0, 1.0))


def compute_shift(reference: np.ndarray, target: np.ndarray, upsample_factor: int = 25) -> Tuple[float, float]:
    """
    Compute the shift between reference and target images using phase correlation.
    
    Args:
        reference: Reference image
        target: Image to align to reference
        upsample_factor: Upsampling factor for sub-pixel accuracy (higher = more precise but slower)
                        Default 25 gives ~0.04 pixel precision (good balance of speed/accuracy)
        
    Returns:
        Tuple of (shift_y, shift_x) in pixels
    """
    # Use phase cross-correlation for sub-pixel accuracy
    shift, error, diffphase = phase_cross_correlation(reference, target, upsample_factor=upsample_factor)
    
    print(f"  Detected shift: ({shift[0]:.3f}, {shift[1]:.3f}) pixels, error: {error:.6f}")
    
    return shift[0], shift[1]


def decompose_affine_matrix(affine_matrix: np.ndarray) -> Dict[str, float]:
    """
    Decompose 2x3 affine matrix into rotation, scale, shear, and translation.
    
    Returns:
        Dictionary with keys: rotation_deg, scale_x, scale_y, shear, translation_x, translation_y
    """
    # Extract components
    a, b = affine_matrix[0, 0], affine_matrix[0, 1]
    c, d = affine_matrix[1, 0], affine_matrix[1, 1]
    tx, ty = affine_matrix[0, 2], affine_matrix[1, 2]
    
    # Compute scale and rotation
    scale_x = float(np.sqrt(a**2 + c**2))
    scale_y = float(np.sqrt(b**2 + d**2))
    
    # Rotation angle
    rotation_rad = float(np.arctan2(c, a))
    rotation_deg = float(np.degrees(rotation_rad))
    
    # Shear (normalized)
    shear = float((a*b + c*d) / (scale_x * scale_y)) if (scale_x * scale_y) > 1e-8 else 0.0
    
    return {
        'rotation_deg': rotation_deg,
        'scale_x': scale_x,
        'scale_y': scale_y,
        'shear': shear,
        'translation_x': float(tx),
        'translation_y': float(ty)
    }


def structural_rep(img: np.ndarray) -> np.ndarray:
    """Compute normalized Sobel gradient magnitude used for structural comparisons."""
    img32 = img.astype(np.float32)
    sobelx = cv2.Sobel(img32, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img32, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    mmax = float(np.max(mag))
    if mmax > 1e-8:
        mag /= mmax
    return mag


def compute_affine_transform(reference: np.ndarray, target: np.ndarray, 
                             upsample_factor: int = 25,
                             improvement_tolerance: float = 0.005,
                             use_structural: bool = True,
                             ecc_iterations: int = 600) -> Tuple[np.ndarray, float, Dict[str, float]]:
    """
    Compute affine transformation matrix between reference and target using ECC.
    Accept only if RMS improves by more than improvement_tolerance.
    If use_structural is True, performs ECC on gradient magnitude representations.
    
    Args:
        ecc_iterations: Maximum ECC iterations (default 600, good balance of speed/convergence)
    """
    # Start with translation estimate from phase correlation
    shift_y, shift_x = compute_shift(reference, target, upsample_factor)

    # Translation-only warp matrix (baseline)
    translation_matrix = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)

    # Compute baseline translation RMS for later comparison
    h, w = target.shape[:2]
    translated = cv2.warpAffine(target, translation_matrix, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    translation_rms = compute_rms_error(reference, translated)

    # Prepare inputs for ECC (structural or raw)
    if use_structural:
        ref_base = structural_rep(reference)
        tgt_base = structural_rep(target)
    else:
        ref_base = reference.astype(np.float32)
        tgt_base = target.astype(np.float32)

    ref_mean, ref_std = float(np.mean(ref_base)), float(np.std(ref_base))
    tgt_mean, tgt_std = float(np.mean(tgt_base)), float(np.std(tgt_base))
    ref_proc = (ref_base - ref_mean) / ref_std if ref_std > 1e-8 else ref_base - ref_mean
    tgt_proc = (tgt_base - tgt_mean) / tgt_std if tgt_std > 1e-8 else tgt_base - tgt_mean

    # Initialize affine matrix with translation
    warp_matrix = translation_matrix.copy()

    # ECC termination criteria (usually converges much earlier than max iterations)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_iterations, 1e-6)

    cc = 0.0
    affine_rms = float('inf')
    used_affine = False
    try:
        cc, warp_matrix = cv2.findTransformECC(
            ref_proc,
            tgt_proc,
            warp_matrix,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5
        )
        # Evaluate affine RMS
        affine_aligned = cv2.warpAffine(target, warp_matrix, (w, h), flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        affine_rms = compute_rms_error(reference, affine_aligned)
        improvement = translation_rms - affine_rms
        if improvement > improvement_tolerance:
            used_affine = True
            print(f"  Affine refinement accepted (CC={cc:.6f}, ΔRMS={improvement:+.6f})")
        else:
            print(f"  Affine refinement discarded (CC={cc:.6f}, ΔRMS={improvement:+.6f} <= tolerance {improvement_tolerance})")
            warp_matrix = translation_matrix
            cc = 0.0
            affine_rms = translation_rms
    except cv2.error as e:
        print(f"  Warning: ECC affine refinement failed ({e.msg}), using translation only")
        warp_matrix = translation_matrix
        cc = 0.0
        affine_rms = translation_rms

    stats = {
        'translation_rms': float(translation_rms),
        'affine_rms': float(affine_rms),
        'used_affine': bool(used_affine)
    }
    return warp_matrix, float(cc), stats


def compute_homography(reference: np.ndarray, target: np.ndarray, 
                       min_match_count: int = 10) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """
    Compute homography (perspective transform) between reference and target using feature matching.
    Handles complex parallax distortions including perspective effects.
    
    Args:
        reference: Reference image
        target: Image to align to reference
        min_match_count: Minimum number of feature matches required
        
    Returns:
        3x3 homography matrix, or None if insufficient matches
    """
    # Convert to uint8 for feature detection
    ref_u8 = (np.clip(reference, 0, 1) * 255).astype(np.uint8)
    tgt_u8 = (np.clip(target, 0, 1) * 255).astype(np.uint8)
    
    # Use ORB (fast and patent-free) for feature detection
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8, 
                         edgeThreshold=15, patchSize=31)
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(ref_u8, None)
    kp2, des2 = orb.detectAndCompute(tgt_u8, None)
    
    if des1 is None or des2 is None or len(kp1) < min_match_count or len(kp2) < min_match_count:
        print(f"  Warning: Insufficient features detected (ref: {len(kp1) if kp1 else 0}, "
              f"tgt: {len(kp2) if kp2 else 0})")
        return (None, {})
    
    # Match features using BFMatcher with Hamming distance for ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    print(f"  Found {len(good_matches)} good feature matches")
    
    if len(good_matches) < min_match_count:
        print(f"  Warning: Insufficient good matches ({len(good_matches)} < {min_match_count})")
        return (None, {})
    
    # Extract matched keypoint coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography using RANSAC
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print("  Warning: Homography estimation failed")
        return (None, {})
    
    matches_mask = mask.ravel().tolist()
    inliers = sum(matches_mask)
    inlier_ratio = inliers / len(good_matches) if len(good_matches) > 0 else 0.0
    print(f"  Homography inliers: {inliers}/{len(good_matches)} ({inlier_ratio:.1%})")
    
    # Compute reprojection error for inliers
    inlier_pts_src = src_pts[mask.ravel() == 1]
    inlier_pts_dst = dst_pts[mask.ravel() == 1]
    if len(inlier_pts_src) > 0:
        projected = cv2.perspectiveTransform(inlier_pts_dst, H)
        reproj_error = float(np.mean(np.linalg.norm(inlier_pts_src - projected, axis=2)))
    else:
        reproj_error = float('inf')
    
    stats = {
        'total_matches': len(good_matches),
        'inliers': inliers,
        'inlier_ratio': float(inlier_ratio),
        'reprojection_error': reproj_error
    }
    
    return H, stats


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


def apply_affine(image: np.ndarray, affine_matrix: np.ndarray, 
                 interpolation: str = 'cubic') -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply affine transformation to an image and return the warped image with valid mask.
    
    Args:
        image: Input image
        affine_matrix: 2x3 affine transformation matrix
        interpolation: Interpolation method
        
    Returns:
        Tuple of (warped_image, valid_mask) where mask is 1 for valid pixels, 0 for invalid
    """
    interp_flags = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    flags = interp_flags.get(interpolation, cv2.INTER_CUBIC)
    
    h, w = image.shape[:2]
    warped = cv2.warpAffine(image, affine_matrix, (w, h), 
                           flags=flags, borderMode=cv2.BORDER_CONSTANT, 
                           borderValue=0)
    
    # Create mask for valid (non-zero border) regions
    mask = np.ones_like(image, dtype=np.uint8)
    mask_warped = cv2.warpAffine(mask, affine_matrix, (w, h),
                                 flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    
    return warped, mask_warped.astype(np.float32)


def apply_homography(image: np.ndarray, homography_matrix: np.ndarray,
                     interpolation: str = 'cubic') -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply homography (perspective transform) to an image with valid mask.
    
    Args:
        image: Input image
        homography_matrix: 3x3 homography matrix
        interpolation: Interpolation method
        
    Returns:
        Tuple of (warped_image, valid_mask)
    """
    interp_flags = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    flags = interp_flags.get(interpolation, cv2.INTER_CUBIC)
    
    h, w = image.shape[:2]
    warped = cv2.warpPerspective(image, homography_matrix, (w, h),
                                 flags=flags, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    
    # Create mask for valid regions
    mask = np.ones_like(image, dtype=np.uint8)
    mask_warped = cv2.warpPerspective(mask, homography_matrix, (w, h),
                                      flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=0)
    
    return warped, mask_warped.astype(np.float32)


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
                            number_of_bins: int = 50,
                            mi_iterations: int = 150) -> Tuple[float, float]:
    """
    Estimate pure translation between reference and moving using Mutual Information (multi-modal robust).

    Args:
        mi_iterations: Maximum MI registration iterations (default 150, good balance for thermal)

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
                                                         numberOfIterations=mi_iterations,
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
                           debug: bool = False,
                           mi_iterations: int = 150) -> Tuple[np.ndarray, np.dtype]:
    """
    Align thermal image to the reference image using MI and save cropped result.
    Note: Thermal band may have different native resolution and will be upsampled to match reference.

    reference_full: reference image BEFORE cropping
    crop_region: (top, bottom, left, right) used on the aligned visible bands
    thermal_path: path to the thermal TIFF (image 6)
    output_dir: where to save the aligned thermal TIFF
    mi_iterations: Maximum MI iterations (default 150 for speed)
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

    # Resize thermal to match reference size (if needed)
    ref_h, ref_w = reference_full.shape
    therm_h, therm_w = thermal_proc.shape[:2]
    if therm_h != ref_h or therm_w != ref_w:
        print(f"  Thermal original size: {therm_h}x{therm_w}, resizing to {ref_h}x{ref_w}")
        thermal_up = resize_to_shape(thermal_proc, (ref_h, ref_w), interpolation)
    else:
        print(f"  Thermal size: {therm_h}x{therm_w} (matches reference, no resizing needed)")
        thermal_up = thermal_proc

    # MI-based translation registration
    print("  Estimating translation via Mutual Information...")
    ty, tx = register_translation_mi(reference_full, thermal_up, mi_iterations=mi_iterations)
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
    # Save without compression
    cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
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
                                  thermal_range_dn: Optional[Tuple[float, float]] = None,
                                  initial_shift_y: float = 0.0,
                                  initial_shift_x: float = 0.0,
                                  initial_scale: float = 1.0) -> Tuple[float, float, float]:
    """
    Manual interactive alignment of the thermal image using arrow keys/buttons for translation and zoom in/out.
    Returns (shift_y, shift_x, scale) applied (in reference pixel units), saves aligned thermal on confirm.
    
    Args:
        initial_shift_y: Starting Y shift (from previous alignment)
        initial_shift_x: Starting X shift (from previous alignment)
        initial_scale: Starting scale factor (from previous alignment)
    """
    print("\nManual thermal alignment UI starting...")
    if initial_shift_y != 0.0 or initial_shift_x != 0.0 or initial_scale != 1.0:
        print(f"  Starting position: shift_y={initial_shift_y:.3f}, shift_x={initial_shift_x:.3f}, scale={initial_scale:.5f}\n")
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
            # Only use valid (non-zero) pixels for scaling to avoid amplifying border artifacts
            valid_thermal = therm_img_cropped[therm_img_cropped > 1e-6]
            if len(valid_thermal) > 100:  # Ensure sufficient valid pixels
                mn, mx = float(np.min(valid_thermal)), float(np.max(valid_thermal))
                therm_disp = np.clip((therm_img_cropped - mn) / max(1e-8, (mx - mn)), 0, 1)
            else:
                # Not enough valid thermal data, just clip to [0,1]
                therm_disp = np.clip(therm_img_cropped, 0, 1)
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

    # Interactive state - initialize with provided starting values
    ty, tx, scale = initial_shift_y, initial_shift_x, initial_scale
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
            ty, tx, scale = initial_shift_y, initial_shift_x, initial_scale
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
        ty, tx, scale = initial_shift_y, initial_shift_x, initial_scale
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
    # Save without compression
    cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
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
    # Save without compression
    cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    print(f"  Applied manual thermal correction: shift_y={shift_y:.3f}, shift_x={shift_x:.3f}, scale={scale:.5f}")
    print(f"  Saved aligned thermal: {out_path.name}")

    return thermal_cropped.astype(np.float32), orig_dtype


def compute_valid_region(shifts: Optional[List[Tuple[float, float]]] = None, 
                        image_shape: Optional[Tuple[int, int]] = None,
                        masks: Optional[List[np.ndarray]] = None) -> Tuple[int, int, int, int]:
    """
    Compute the valid region that is present in all aligned images.
    
    Args:
        shifts: List of (shift_y, shift_x) for each image (for translation mode)
        image_shape: Shape of the images (height, width) (for translation mode)
        masks: List of valid masks for each image (for affine/homography mode)
        
    Returns:
        Tuple of (top, bottom, left, right) pixel coordinates defining valid region
    """
    # For affine/homography mode, use masks
    if masks is not None:
        # Combine all masks to find common valid region
        combined_mask = np.ones_like(masks[0], dtype=np.float32)
        for mask in masks:
            combined_mask = np.minimum(combined_mask, mask)
        
        # Find bounding box of valid region (where mask > 0.5)
        valid_pixels = np.where(combined_mask > 0.5)
        if len(valid_pixels[0]) == 0:
            raise ValueError("No valid overlapping region found")
        
        top = int(np.min(valid_pixels[0]))
        bottom = int(np.max(valid_pixels[0])) + 1
        left = int(np.min(valid_pixels[1]))
        right = int(np.max(valid_pixels[1])) + 1
        
        return top, bottom, left, right
    
    # For translation mode, use shifts
    if shifts is not None and image_shape is not None:
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
    
    raise ValueError("Must provide either (shifts and image_shape) or masks")


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
                output_dir: Optional[Path] = None, upsample_factor: int = 25,
                warp_interpolation: str = 'cubic', alignment_mode: str = 'translation',
                affine_tolerance: float = 0.005, edge_metrics: bool = False,
                metrics_out: Optional[Path] = None, ecc_iterations: int = 600) -> AlignmentResult:
    """
    Align a set of multispectral images.

    Added parameters:
      affine_tolerance: minimum RMS improvement to accept affine refinement
      edge_metrics: if True compute structural (Sobel magnitude) RMS/correlation
      metrics_out: optional CSV path to append per-image metrics
      ecc_iterations: maximum ECC iterations for affine refinement (default 600)
    """
    print(f"\n{'='*60}")
    print(f"Aligning image set: {image_paths[0].stem.rsplit('_', 1)[0]}")
    print(f"{'='*60}")

    if reference_idx < 0 or reference_idx >= len(image_paths):
        raise ValueError(f"Reference index {reference_idx} out of range [0, {len(image_paths)-1}]")

    if output_dir is None:
        output_dir = image_paths[0].parent / "aligned_output"
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\nLoading images...")
    images = []
    original_dtypes = []
    for path in image_paths:
        print(f"  Loading {path.name}")
        img_original = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img_original is None:
            raise ValueError(f"Could not load image: {path}")
        original_dtypes.append(img_original.dtype)
        img = load_image(path)
        images.append(img)
        print(f"    Shape: {img.shape}, range: [{img.min():.4f}, {img.max():.4f}]")

    reference_img = images[reference_idx]
    print(f"\nUsing {image_paths[reference_idx].name} as reference")
    print(f"Alignment mode: {alignment_mode}")
    if edge_metrics:
        print("Edge metrics: ENABLED (structural Sobel magnitude)")
        ref_struct = structural_rep(reference_img)
    else:
        ref_struct = None

    all_metrics: List[AlignmentMetrics] = []

    if alignment_mode == 'translation':
        print("\nComputing translational shifts...")
        shifts = []
        transforms = []
        for i, (img, path) in enumerate(zip(images, image_paths)):
            if i == reference_idx:
                print(f"  {path.name}: Reference image (no shift)")
                shifts.append((0.0, 0.0))
                transforms.append(None)
            else:
                print(f"  {path.name}:")
                sy, sx = compute_shift(reference_img, img, upsample_factor)
                shifts.append((sy, sx))
                transforms.append(('translation', sy, sx))

        print(f"\nApplying translational shifts (interpolation: {warp_interpolation})...")
        aligned_images = []
        masks = []
        for i, (img, transform, path) in enumerate(zip(images, transforms, image_paths)):
            if i == reference_idx:
                aligned_images.append(img)
                masks.append(np.ones_like(img, dtype=np.float32))
                params = {'shift_y': 0.0, 'shift_x': 0.0}
                if edge_metrics:
                    params['edge_rms'] = 0.0
                    params['edge_corr'] = 1.0
                all_metrics.append(AlignmentMetrics(path.name, 'translation', 0.0, 1.0, params))
            else:
                sy, sx = transform[1], transform[2]
                print(f"  {path.name}: Applying shift ({sy:.3f}, {sx:.3f})")
                aligned = apply_shift(img, sy, sx, interpolation=warp_interpolation)
                aligned_images.append(aligned)
                mask = np.ones_like(img, dtype=np.float32)
                masks.append(mask)
                rms = compute_rms_error(reference_img, aligned, mask)
                corr = compute_correlation(reference_img, aligned, mask)
                print(f"    RMS error: {rms:.6f}, Correlation: {corr:.4f}")
                params = {'shift_y': sy, 'shift_x': sx}
                if edge_metrics:
                    aligned_struct = structural_rep(aligned)
                    rms_edge = compute_rms_error(ref_struct, aligned_struct)
                    corr_edge = compute_correlation(ref_struct, aligned_struct)
                    print(f"    Edge RMS: {rms_edge:.6f}, Edge Corr: {corr_edge:.4f}")
                    params['edge_rms'] = rms_edge
                    params['edge_corr'] = corr_edge
                all_metrics.append(AlignmentMetrics(path.name, 'translation', rms, corr, params))
        print("\nComputing valid region...")
        top, bottom, left, right = compute_valid_region(shifts=shifts, image_shape=reference_img.shape)

    elif alignment_mode == 'affine':
        print("\nComputing affine transformations...")
        transforms = []
        ecc_scores = []
        for i, (img, path) in enumerate(zip(images, image_paths)):
            if i == reference_idx:
                print(f"  {path.name}: Reference image (no transformation)")
                transforms.append(None)
                ecc_scores.append(1.0)
            else:
                print(f"  {path.name}:")
                affine_matrix, ecc, stats = compute_affine_transform(reference_img, img,
                                                                    upsample_factor=upsample_factor,
                                                                    improvement_tolerance=affine_tolerance,
                                                                    use_structural=True,
                                                                    ecc_iterations=ecc_iterations)
                transforms.append(('affine', affine_matrix, stats))
                ecc_scores.append(ecc)
        print(f"\nApplying affine transformations (interpolation: {warp_interpolation})...")
        aligned_images = []
        masks = []
        for i, (img, transform, path) in enumerate(zip(images, transforms, image_paths)):
            if i == reference_idx:
                aligned_images.append(img)
                mask = np.ones_like(img, dtype=np.float32)
                masks.append(mask)
                params = {'identity': 1.0}
                if edge_metrics:
                    params['edge_rms'] = 0.0
                    params['edge_corr'] = 1.0
                all_metrics.append(AlignmentMetrics(path.name, 'affine', 0.0, 1.0, params))
            else:
                affine_matrix = transform[1]
                stats = transform[2]
                usage_note = "affine" if stats.get('used_affine') else "translation (affine rejected)"
                print(f"  {path.name}: Applying {usage_note} transformation")
                aligned, mask = apply_affine(img, affine_matrix, interpolation=warp_interpolation)
                aligned_images.append(aligned)
                masks.append(mask)
                rms = compute_rms_error(reference_img, aligned, mask)
                corr = compute_correlation(reference_img, aligned, mask)
                affine_params = decompose_affine_matrix(affine_matrix)
                affine_params['ecc_score'] = ecc_scores[i]
                affine_params.update(stats)
                if edge_metrics:
                    aligned_struct = structural_rep(aligned)
                    rms_edge = compute_rms_error(ref_struct, aligned_struct)
                    corr_edge = compute_correlation(ref_struct, aligned_struct)
                    print(f"    Edge RMS: {rms_edge:.6f}, Edge Corr: {corr_edge:.4f}")
                    affine_params['edge_rms'] = rms_edge
                    affine_params['edge_corr'] = corr_edge
                print(f"    RMS error: {rms:.6f}, Correlation: {corr:.4f}")
                print(f"    Rotation: {affine_params['rotation_deg']:.2f}°, Scale: ({affine_params['scale_x']:.4f}, {affine_params['scale_y']:.4f})")
                all_metrics.append(AlignmentMetrics(path.name, 'affine', rms, corr, affine_params))
        print("\nComputing valid region from masks...")
        top, bottom, left, right = compute_valid_region(masks=masks)

    elif alignment_mode == 'homography':
        print("\nComputing homographies...")
        transforms = []
        match_stats = []
        for i, (img, path) in enumerate(zip(images, image_paths)):
            if i == reference_idx:
                print(f"  {path.name}: Reference image (no transformation)")
                transforms.append(None)
                match_stats.append({})
            else:
                print(f"  {path.name}:")
                H, stats = compute_homography(reference_img, img)
                if H is None:
                    print(f"  Warning: Homography failed for {path.name}, falling back to affine")
                    affine_matrix, ecc, aff_stats = compute_affine_transform(reference_img, img,
                                                                             upsample_factor=upsample_factor,
                                                                             improvement_tolerance=affine_tolerance,
                                                                             use_structural=True,
                                                                             ecc_iterations=ecc_iterations)
                    transforms.append(('affine_fallback', affine_matrix, ecc, aff_stats))
                    match_stats.append({'fallback': 'affine', 'ecc_score': ecc})
                else:
                    transforms.append(('homography', H, stats))
                    match_stats.append(stats)
        print(f"\nApplying perspective transformations (interpolation: {warp_interpolation})...")
        aligned_images = []
        masks = []
        for i, (img, transform, path) in enumerate(zip(images, transforms, image_paths)):
            if i == reference_idx:
                aligned_images.append(img)
                mask = np.ones_like(img, dtype=np.float32)
                masks.append(mask)
                params = {'identity': 1.0}
                if edge_metrics:
                    params['edge_rms'] = 0.0
                    params['edge_corr'] = 1.0
                all_metrics.append(AlignmentMetrics(path.name, 'homography', 0.0, 1.0, params))
            else:
                kind = transform[0]
                if kind == 'homography':
                    H = transform[1]
                    stats = transform[2]
                    print(f"  {path.name}: Applying homography")
                    aligned, mask = apply_homography(img, H, interpolation=warp_interpolation)
                    rms = compute_rms_error(reference_img, aligned, mask)
                    corr = compute_correlation(reference_img, aligned, mask)
                    params = stats.copy()
                    if edge_metrics:
                        aligned_struct = structural_rep(aligned)
                        rms_edge = compute_rms_error(ref_struct, aligned_struct)
                        corr_edge = compute_correlation(ref_struct, aligned_struct)
                        print(f"    Edge RMS: {rms_edge:.6f}, Edge Corr: {corr_edge:.4f}")
                        params['edge_rms'] = rms_edge
                        params['edge_corr'] = corr_edge
                    print(f"    RMS error: {rms:.6f}, Correlation: {corr:.4f}")
                    all_metrics.append(AlignmentMetrics(path.name, 'homography', rms, corr, params))
                else:  # affine fallback
                    affine_matrix = transform[1]
                    ecc = transform[2]
                    stats = transform[3]
                    print(f"  {path.name}: Applying affine (fallback)")
                    aligned, mask = apply_affine(img, affine_matrix, interpolation=warp_interpolation)
                    rms = compute_rms_error(reference_img, aligned, mask)
                    corr = compute_correlation(reference_img, aligned, mask)
                    affine_params = decompose_affine_matrix(affine_matrix)
                    affine_params['ecc_score'] = ecc
                    affine_params.update(stats)
                    if edge_metrics:
                        aligned_struct = structural_rep(aligned)
                        rms_edge = compute_rms_error(ref_struct, aligned_struct)
                        corr_edge = compute_correlation(ref_struct, aligned_struct)
                        print(f"    Edge RMS: {rms_edge:.6f}, Edge Corr: {corr_edge:.4f}")
                        affine_params['edge_rms'] = rms_edge
                        affine_params['edge_corr'] = corr_edge
                    print(f"    RMS error: {rms:.6f}, Correlation: {corr:.4f}")
                    all_metrics.append(AlignmentMetrics(path.name, 'homography (affine fallback)', rms, corr, affine_params))
                aligned_images.append(aligned)
                masks.append(mask)
        print("\nComputing valid region from masks...")
        top, bottom, left, right = compute_valid_region(masks=masks)
    else:
        raise ValueError(f"Unknown alignment mode: {alignment_mode}")

    print(f"  Valid region: top={top}, bottom={bottom}, left={left}, right={right}")
    print(f"  Output size: {bottom-top}x{right-left} pixels")
    if bottom <= top or right <= left:
        raise ValueError("Invalid crop region - shifts may be too large!")

    print("\nCropping to valid region...")
    cropped_images = [img[top:bottom, left:right] for img in aligned_images]

    if debug:
        print("\nShowing alignment visualization...")
        visualize_alignment(cropped_images, [p.name for p in image_paths], enable_layer_toggle=True)

    print("\nSaving aligned images...")
    for img, path, orig_dtype in zip(cropped_images, image_paths, original_dtypes):
        output_path = output_dir / path.name
        if orig_dtype == np.uint16:
            img_save = (img * 65535).astype(np.uint16)
        elif orig_dtype == np.uint8:
            img_save = (img * 255).astype(np.uint8)
        else:
            img_save = img
        # Save without compression
        cv2.imwrite(str(output_path), img_save, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        print(f"  Saved {output_path.name}")

    print(f"\n{'='*60}")
    print("Alignment complete!")
    print(f"{'='*60}")

    print("\n" + "="*60)
    print("ALIGNMENT QUALITY METRICS")
    print("="*60)
    for metric in all_metrics:
        print(f"\n{metric.image_name}:")
        print(f"  Mode: {metric.mode}")
        print(f"  RMS Error: {metric.rms_error:.6f}")
        print(f"  Correlation: {metric.correlation:.4f}")
        tp = metric.transform_params
        if metric.mode == 'translation':
            print(f"  Shift: Y={tp['shift_y']:.3f}, X={tp['shift_x']:.3f} px")
        if 'affine' in metric.mode and 'identity' not in tp:
            if 'rotation_deg' in tp:
                print(f"  Rotation: {tp['rotation_deg']:.2f}°")
                print(f"  Scale: X={tp.get('scale_x', 1):.4f}, Y={tp.get('scale_y', 1):.4f}")
            if 'translation_rms' in tp and 'affine_rms' in tp:
                used = tp.get('used_affine', False)
                print(f"  Baseline Translation RMS: {tp['translation_rms']:.6f}")
                print(f"  Affine RMS: {tp['affine_rms']:.6f} ({'accepted' if used else 'rejected'})")
        if metric.mode.startswith('homography') and 'identity' not in tp:
            if 'total_matches' in tp:
                print(f"  Feature Matches: {tp.get('total_matches', 0)}")
                print(f"  Inliers: {tp.get('inliers', 0)} ({tp.get('inlier_ratio', 0)*100:.1f}%)")
                print(f"  Reprojection Error: {tp.get('reprojection_error', float('inf')):.4f}")
        if 'edge_rms' in tp:
            print(f"  Edge RMS: {tp['edge_rms']:.6f}, Edge Corr: {tp['edge_corr']:.4f}")

    non_ref = [m for m in all_metrics if m.rms_error > 0]
    if non_ref:
        avg_rms = sum(m.rms_error for m in non_ref) / len(non_ref)
        avg_corr = sum(m.correlation for m in non_ref) / len(non_ref)
        print(f"\n{'='*60}")
        print("OVERALL STATISTICS:")
        print(f"  Average RMS Error: {avg_rms:.6f}")
        print(f"  Average Correlation: {avg_corr:.4f}")
        if edge_metrics:
            avg_edge_rms = sum(m.transform_params.get('edge_rms', 0.0) for m in non_ref) / len(non_ref)
            avg_edge_corr = sum(m.transform_params.get('edge_corr', 0.0) for m in non_ref) / len(non_ref)
            print(f"  Average Edge RMS: {avg_edge_rms:.6f}")
            print(f"  Average Edge Corr: {avg_edge_corr:.4f}")
        print(f"  Images processed: {len(image_paths)}")
        print(f"={'='*60}\n")

    # Optional CSV export
    if metrics_out is not None:
        header = ["image_name", "mode", "rms_error", "correlation", "edge_rms", "edge_corr"]
        # include dynamic transform params keys union
        dyn_keys = set()
        for m in all_metrics:
            dyn_keys.update(k for k in m.transform_params.keys() if k not in ("edge_rms", "edge_corr"))
        dyn_keys = sorted(dyn_keys)
        # Build CSV rows
        import csv
        file_exists = metrics_out.exists()
        with open(metrics_out, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header + dyn_keys)
            for m in all_metrics:
                tp = m.transform_params
                row = [m.image_name, m.mode, f"{m.rms_error:.8f}", f"{m.correlation:.6f}",
                       f"{tp.get('edge_rms', '')}", f"{tp.get('edge_corr', '')}"]
                for k in dyn_keys:
                    row.append(str(tp.get(k, '')))
                writer.writerow(row)
        print(f"Metrics appended to CSV: {metrics_out}")

    return AlignmentResult(reference_full=reference_img,
                           crop_region=(top, bottom, left, right),
                           output_dir=output_dir,
                           cropped_images=cropped_images,
                           cropped_names=[p.name for p in image_paths],
                           metrics=all_metrics)


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
    parser.add_argument("--upsample", type=int, default=25,
                       help="Upsampling factor for sub-pixel precision (default: 25=fast, 100=precise, 25 gives ~0.04px accuracy)")
    parser.add_argument("--ecc-iterations", type=int, default=600,
                       help="Maximum ECC iterations for affine refinement (default: 600, higher=slower but more accurate)")
    parser.add_argument("--mi-iterations", type=int, default=150,
                       help="Maximum MI iterations for thermal alignment (default: 150, higher=slower)")
    parser.add_argument("--interpolation", type=str, default="cubic",
                       choices=["linear", "cubic", "lanczos"],
                       help="Interpolation method for image warping (default: cubic)")
    parser.add_argument("--alignment-mode", type=str, default="translation",
                       choices=["translation", "affine", "homography"],
                       help="Alignment mode: translation (lateral shift only), affine (rotation+scale+shear), "
                            "homography (perspective/parallax distortions) (default: translation)")
    parser.add_argument("--affine-tolerance", type=float, default=0.005,
                       help="Minimum RMS improvement required to accept affine refinement (default: 0.005)")
    parser.add_argument("--edge-metrics", action="store_true",
                       help="Compute structural (edge/Sobel magnitude) RMS & correlation metrics")
    parser.add_argument("--metrics-out", type=str, default=None,
                       help="Optional CSV file path to append per-image metrics (created if missing)")
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
    
    # Determine output directory for checking existing files
    if output_dir is None:
        check_output_dir = input_dir / "aligned_output"
    else:
        check_output_dir = output_dir
    
    # Filter out image sets based on what needs processing
    if check_output_dir.exists():
        filtered_image_sets = {}
        already_aligned_multispectral = {}  # Store sets where bands 1-5 exist but may need thermal
        
        for base_name, paths in image_sets.items():
            # Check if all 5 bands exist in output directory
            all_bands_exist = True
            for i in range(1, 6):
                expected_file = check_output_dir / f"{base_name}_{i}.tif"
                if not expected_file.exists():
                    all_bands_exist = False
                    break
            
            if all_bands_exist:
                # Check if thermal (band 6) exists
                thermal_exists = (check_output_dir / f"{base_name}_6.tif").exists()
                
                if thermal_exists:
                    print(f"Skipping {base_name}: fully aligned (found in {check_output_dir.name})")
                else:
                    # Multispectral done but thermal needs processing
                    if args.align_thermal and len(paths) >= 6:
                        print(f"Skipping {base_name} multispectral: already aligned (will process thermal)")
                        already_aligned_multispectral[base_name] = paths
                    else:
                        print(f"Skipping {base_name}: already aligned (found in {check_output_dir.name})")
            else:
                # Needs multispectral alignment
                filtered_image_sets[base_name] = paths
        
        image_sets = filtered_image_sets
        
        if not image_sets and not already_aligned_multispectral:
            print("\nAll image sets are already aligned. Nothing to do.")
            return
    
    # Phase 1: Process all multispectral bands (bands 1-5) first
    print("\n" + "="*60)
    print("PHASE 1: ALIGNING MULTISPECTRAL BANDS (1-5)")
    print("="*60)
    
    results = {}  # Store results for thermal processing
    thermal_sets = {}  # Store thermal image paths for later processing
    
    # Process new multispectral alignments
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
            # Always defer debug until after thermal alignment
            visible_debug = args.debug and not args.align_thermal
            metrics_out_path = Path(args.metrics_out) if args.metrics_out else None
            result = align_images(paths_to_process, reference_idx=reference_idx, 
                        debug=visible_debug, output_dir=output_dir, 
                        upsample_factor=args.upsample,
                        warp_interpolation=args.interpolation,
                        alignment_mode=args.alignment_mode,
                        affine_tolerance=args.affine_tolerance,
                        edge_metrics=args.edge_metrics,
                        metrics_out=metrics_out_path,
                        ecc_iterations=args.ecc_iterations)
            
            # Store result for thermal processing
            if args.align_thermal and len(paths) >= 6:
                results[base_name] = result
                thermal_sets[base_name] = paths[5]
                
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Load pre-existing multispectral alignments for thermal processing
    if 'already_aligned_multispectral' in locals() and already_aligned_multispectral:
        print("\nLoading pre-aligned multispectral bands for thermal processing...")
        for base_name, paths in already_aligned_multispectral.items():
            try:
                paths_to_process = paths[:5]
                reference_idx = args.reference - 1
                if reference_idx < 0 or reference_idx >= len(paths_to_process):
                    reference_idx = 0
                
                # Load the reference image and aligned cropped images
                ref_path = check_output_dir / paths_to_process[reference_idx].name
                reference_full = load_image(ref_path)
                
                cropped_images = []
                cropped_names = []
                for path in paths_to_process:
                    aligned_path = check_output_dir / path.name
                    img = load_image(aligned_path)
                    cropped_images.append(img)
                    cropped_names.append(path.name)
                
                # Estimate crop region from first image (assumes consistent cropping)
                # For simplicity, use full image bounds since we're loading already-cropped images
                h, w = cropped_images[0].shape
                crop_region = (0, h, 0, w)
                
                # Create a minimal AlignmentResult for thermal processing
                result = AlignmentResult(
                    reference_full=reference_full,
                    crop_region=crop_region,
                    output_dir=check_output_dir,
                    cropped_images=cropped_images,
                    cropped_names=cropped_names,
                    metrics=[]
                )
                
                results[base_name] = result
                thermal_sets[base_name] = paths[5]
                print(f"  Loaded {base_name} for thermal alignment")
                
            except Exception as e:
                print(f"Error loading pre-aligned {base_name}: {e}")
                continue
    
    # Phase 2: Process thermal images sequentially, using previous thermal as starting point
    if args.align_thermal and thermal_sets:
        print("\n" + "="*60)
        print("PHASE 2: ALIGNING THERMAL IMAGES (BAND 6)")
        print("="*60)
        print(f"Processing {len(thermal_sets)} thermal image(s) sequentially...")
        print("Each thermal alignment will use the previous result as starting point.\n")
        
        # Track cumulative thermal corrections
        cumulative_ty = 0.0
        cumulative_tx = 0.0
        cumulative_scale = 1.0
        
        for idx, (base_name, thermal_path) in enumerate(thermal_sets.items(), 1):
            result = results[base_name]
            ref_full = result.reference_full
            crop_region = result.crop_region
            out_dir = result.output_dir
            
            # Check if thermal image already exists in output
            thermal_output_path = out_dir / f"{base_name}_6.tif"
            if thermal_output_path.exists():
                print(f"\n[{idx}/{len(thermal_sets)}] Skipping {base_name}: thermal already aligned")
                continue
            
            print(f"\n[{idx}/{len(thermal_sets)}] Processing thermal for: {base_name}")
            if idx > 1:
                print(f"  Starting from previous alignment: shift_y={cumulative_ty:.3f}, shift_x={cumulative_tx:.3f}, scale={cumulative_scale:.5f}")
            
            try:
                # Precedence: fixed params > manual UI > automatic MI
                fixed_params_provided = (args.thermal_shift_y is not None
                                         or args.thermal_shift_x is not None
                                         or args.thermal_scale is not None)
                
                if fixed_params_provided:
                    # Fixed parameters: apply cumulative correction
                    ty = cumulative_ty + (0.0 if args.thermal_shift_y is None else float(args.thermal_shift_y))
                    tx = cumulative_tx + (0.0 if args.thermal_shift_x is None else float(args.thermal_shift_x))
                    scale = cumulative_scale * (1.0 if args.thermal_scale is None else float(args.thermal_scale))
                    
                    thermal_cropped, thermal_dtype = apply_fixed_thermal_correction(
                        ref_full, crop_region, thermal_path, out_dir,
                        shift_y=ty, shift_x=tx, scale=scale,
                        interpolation=args.interpolation,
                    )
                    
                    # Update cumulative values
                    cumulative_ty = ty
                    cumulative_tx = tx
                    cumulative_scale = scale
                    
                    # Combined debug view if requested
                    if args.debug:
                        all_images = list(result.cropped_images) + [thermal_cropped]
                        all_names = list(result.cropped_names) + [thermal_path.name]
                        rescale_map = {}
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
                    # Manual interactive alignment with cumulative starting point
                    print(f"  Opening manual alignment GUI...")
                    trange = None
                    if args.thermal_range_min is not None and args.thermal_range_max is not None:
                        trange = (float(args.thermal_range_min), float(args.thermal_range_max))
                    
                    # Modified function call to accept initial values
                    ty, tx, scale = manual_align_and_save_thermal(
                        ref_full, crop_region, thermal_path, out_dir,
                        result.cropped_images, result.cropped_names,
                        interpolation=args.interpolation,
                        thermal_range_dn=trange,
                        initial_shift_y=cumulative_ty,
                        initial_shift_x=cumulative_tx,
                        initial_scale=cumulative_scale
                    )
                    
                    # Update cumulative values with the new absolute values
                    cumulative_ty = ty
                    cumulative_tx = tx
                    cumulative_scale = scale
                    
                else:
                    # Automatic MI-based alignment
                    thermal_cropped, thermal_dtype = align_and_save_thermal(
                        ref_full, crop_region, thermal_path, out_dir,
                        interpolation=args.interpolation,
                        debug=False,
                        mi_iterations=args.mi_iterations)
                    
                    # For automatic mode, reset cumulative values (each is independent)
                    cumulative_ty = 0.0
                    cumulative_tx = 0.0
                    cumulative_scale = 1.0
                    
                    # Show combined debug view if requested
                    if args.debug:
                        all_images = list(result.cropped_images) + [thermal_cropped]
                        all_names = list(result.cropped_names) + [thermal_path.name]
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
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("THERMAL ALIGNMENT COMPLETE")
        print("="*60)


if __name__ == "__main__":
    main()
