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
