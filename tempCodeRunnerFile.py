ia
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 2000, 1e-6)

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