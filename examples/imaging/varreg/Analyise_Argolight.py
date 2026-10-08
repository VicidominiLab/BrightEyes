
#%% Import
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
import matplotlib.colors as mcolors
import os
import io
import torch
import tifffile
import importlib
import pandas as pd
# import napari as nap

from tqdm import trange
from pathlib import Path
from skimage.registration import phase_cross_correlation
# from microssim import MicroSSIM, micro_structural_similarity
from scipy.ndimage import shift
from scipy.ndimage import gaussian_filter
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from email.mime import image
from matplotlib import path

import brighteyes_ism.analysis.Tools_lib as tools
import brighteyes_ism.simulation.simulator as sim
import brighteyes_ism.simulation.PSF_sim as psf_sim
import brighteyes_ism.simulation.Tubulin_sim as st
import brighteyes_ism.analysis.Graph_lib as gra
import brighteyes_ism.analysis.APR_lib as apr
import brighteyes_ism.dataio.mcs as mcs
from torch.fft import rfftn, irfftn, ifftshift, fftn, ifftn, fftshift

from s2ism import s2ism as s2
import s2ism.psf_estimator as est

from varreg import varreg, loss, utils
from varreg import preprocessing as pp
from varreg import phantom_simulator as ph
from varreg import initializers as init
from varreg import discrepancy_principle as dp
from varreg import whiteness_principle as wp
from varreg.utils import compute_otf_from_psf, autocorrelation2d

from scipy.ndimage import gaussian_filter1d, binary_fill_holes, median_filter
from skimage import filters, measure
from skimage.transform import rotate
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import correlate, correlation_lags
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.stats import t
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.gridspec import GridSpec




def add_axis_arrows(ax, labels=("x", "y"),
                    xy_start=(0.1, 0.1),        # starting point in normalized coords
                    dxdy=((0.2, 0), (0, 0.2)),  # direction vectors in normalized coords
                    color="white", lw=1.5, fontsize=10):
    """
    Draw axis arrows in normalized coordinates [0,1].

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the arrows.
    labels : tuple of str, default=("x", "y")
        Labels for the two arrows.
    xy_start : tuple of float, default=(0.1, 0.1)
        Starting point for the arrows in normalized coordinates (0=left/bottom, 1=right/top).
    dxdy : tuple of 2 tuples, default=((0.2, 0), (0, 0.2))
        Displacement vectors (dx, dy) for each arrow, in normalized coordinates.
    color : str, default="white"
        Color of the arrows and text.
    lw : float, default=1.5
        Line width of the arrows.
    fontsize : int, default=10
        Font size of the labels.
    """
    for (dx, dy), label in zip(dxdy, labels):
        # Draw the arrow
        ax.annotate("",
                    xy=(xy_start[0] + dx, xy_start[1] + dy),
                    xytext=xy_start,
                    xycoords=ax.transAxes,
                    textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw),
                    annotation_clip=False)

        # Draw the label
        ax.text(xy_start[0] + dx*1.15,
                xy_start[1] + dy*1.15,
                label,
                color=color, fontsize=fontsize,
                ha="center", va="center",
                transform=ax.transAxes,
                clip_on=False)

def crop_border(image, border=10):
    """Remove empty pixels at the image borders."""
    return image[border:-border, border:-border]


def estimate_grid_angle(image, smooth_sigma=2):
    """Estimate the orientation of the Argolight pattern."""

    smooth = filters.gaussian(
        image,
        sigma=smooth_sigma,
        preserve_range=True,
    )

    threshold = filters.threshold_otsu(smooth)
    binary = binary_fill_holes(smooth > threshold)

    contours = measure.find_contours(binary.astype(float), 0.5)

    angles = []

    for contour in contours:

        if len(contour) < 50:
            continue

        x = contour[:, 1]
        y = contour[:, 0]

        try:
            slope, _ = np.polyfit(x, y, 1)
        except np.linalg.LinAlgError:
            continue

        angle = np.degrees(np.arctan(slope))

        if np.isfinite(angle):
            angles.append(abs(angle))

    if len(angles) == 0:
        return 45.0

    return np.median(angles)

def measure_rotation_drift(result, col_range, n_sections=12, max_shift_um=0.5):
    """
    Measure the residual geometrical drift along the Argolight lines.

    The target width is divided into vertical sections. The profile of
    each section is compared with the central section by cross-correlation.
    A correct rotation should give no systematic shift along the line.
    """

    roi = np.asarray(result["analysis_roi"])
    mask = np.asarray(result["analysis_mask"])
    x_um = np.asarray(result["x_um"])
    pixel_size_um = np.median(np.diff(x_um))

    left, right = col_range
    roi = roi[:, left:right]
    mask = mask[:, left:right]

    edges = np.linspace(0, roi.shape[1], n_sections + 1, dtype=int)
    profiles = []
    section_centers_px = []

    for i in range(n_sections):
        a, b = edges[i], edges[i + 1]
        strip = np.ma.array(roi[:, a:b], mask=~mask[:, a:b])
        profile = strip.mean(axis=1).filled(np.nan)

        valid = np.isfinite(profile)
        if not np.all(valid):
            indices = np.arange(len(profile))
            profile[~valid] = np.interp(indices[~valid], indices[valid], profile[valid])

        profiles.append(profile)
        section_centers_px.append((a + b - 1) / 2)

    profiles = np.asarray(profiles)
    section_centers_px = np.asarray(section_centers_px)

    # Use the central section as geometrical reference
    reference_index = n_sections // 2
    reference = profiles[reference_index] - np.mean(profiles[reference_index])

    max_shift_px = int(round(max_shift_um / pixel_size_um))
    shifts_px = []

    for profile in profiles:
        current = profile - np.mean(profile)

        corr = correlate(current, reference, mode="full", method="auto")
        lags = correlation_lags(len(current), len(reference), mode="full")

        keep = np.abs(lags) <= max_shift_px
        best_lag = lags[keep][np.argmax(corr[keep])]

        shifts_px.append(best_lag)

    shifts_px = np.asarray(shifts_px, dtype=float)

    # Convert both directions to physical units
    section_positions_um = (
        section_centers_px - section_centers_px[reference_index]
    ) * pixel_size_um

    shifts_um = shifts_px * pixel_size_um

    # Residual linear drift
    slope, intercept = np.polyfit(section_positions_um, shifts_um, 1)
    fitted_shifts_um = intercept + slope * section_positions_um

    return {
        "section_positions_um": section_positions_um,
        "shifts_um": shifts_um,
        "fitted_shifts_um": fitted_shifts_um,
        "slope": slope,
        "intercept": intercept,
        "profiles": profiles,
    }

def refine_rotation_angle(
    image,
    initial_angle_deg,
    pixel_size_um,
    x_slice,
    y_slice,
    threshold_fraction=0.10,
    left_inner_fraction=0.02,
    right_inner_fraction=0.01,
    n_sections=12,
    max_shift_um=0.5,
    coarse_range_deg=0.8,
    coarse_step_deg=0.05,
    fine_range_deg=0.08,
    fine_step_deg=0.005,
):
    """
    Refine the rotation angle by minimizing the residual drift of the
    Argolight structure along the line direction.

    The Raw image should be used here. The resulting angle is then
    frozen and reused for every reconstruction.
    """

    def evaluate_angle(angle_deg):
        result = prepare_rotated_roi(
            image=image,
            angle_deg=angle_deg,
            pixel_size_um=pixel_size_um,
            x_slice=x_slice,
            y_slice=y_slice,
        )

        range_info = find_target_column_range(
            result=result,
            threshold_fraction=threshold_fraction,
            smooth_sigma_px=5,
            left_inner_fraction=left_inner_fraction,
            right_inner_fraction=right_inner_fraction,
        )

        col_range = (range_info["left"], range_info["right"])

        drift = measure_rotation_drift(
            result=result,
            col_range=col_range,
            n_sections=n_sections,
            max_shift_um=max_shift_um,
        )

        return abs(drift["slope"]), drift, result, col_range

    # ------------------------------------------------------------
    # Coarse search
    # ------------------------------------------------------------

    coarse_angles = np.arange(
        initial_angle_deg - coarse_range_deg,
        initial_angle_deg + coarse_range_deg + coarse_step_deg / 2,
        coarse_step_deg,
    )

    coarse_scores = []

    for angle in coarse_angles:
        score, _, _, _ = evaluate_angle(angle)
        coarse_scores.append(score)

    coarse_scores = np.asarray(coarse_scores)
    coarse_best_angle = coarse_angles[np.argmin(coarse_scores)]

    # ------------------------------------------------------------
    # Fine search around the best coarse angle
    # ------------------------------------------------------------

    fine_angles = np.arange(
        coarse_best_angle - fine_range_deg,
        coarse_best_angle + fine_range_deg + fine_step_deg / 2,
        fine_step_deg,
    )

    fine_scores = []

    for angle in fine_angles:
        score, _, _, _ = evaluate_angle(angle)
        fine_scores.append(score)

    fine_scores = np.asarray(fine_scores)
    best_index = np.argmin(fine_scores)
    refined_angle = fine_angles[best_index]

    _, final_drift, final_result, final_col_range = evaluate_angle(refined_angle)

    info = {
        "initial_angle_deg": initial_angle_deg,
        "refined_angle_deg": refined_angle,
        "coarse_angles_deg": coarse_angles,
        "coarse_scores": coarse_scores,
        "fine_angles_deg": fine_angles,
        "fine_scores": fine_scores,
        "final_drift": final_drift,
        "final_result": final_result,
        "final_col_range": final_col_range,
    }

    return refined_angle, info

def rotate_image_and_mask(
    image,
    angle_deg,
):
    """
    Rotate the image and a binary validity mask.

    The mask distinguishes real image pixels from
    artificial padding introduced by rotation.
    """

    rotated = rotate(
        image,
        angle_deg,
        resize=True,
        preserve_range=True,
        order=1,
        mode="constant",
        cval=0,
    )

    valid_mask = rotate(
        np.ones_like(image, dtype=float),
        angle_deg,
        resize=True,
        preserve_range=True,
        order=0,
        mode="constant",
        cval=0,
    ) > 0.5

    return rotated, valid_mask


def prepare_rotated_roi(
    image,
    angle_deg,
    pixel_size_um,
    x_slice=slice(None),
    y_slice=slice(None),
):
    """
    Rotate the image and prepare the ROI used for analysis.

    Rows containing only rotation padding are removed.
    No intensity-based cropping is applied.
    """

    rotated, valid_mask = rotate_image_and_mask(
        image=image,
        angle_deg=angle_deg,
    )

    # User-defined crop after rotation
    roi = rotated[
        y_slice,
        x_slice,
    ]

    roi_mask = valid_mask[
        y_slice,
        x_slice,
    ]

    # Remove only rows containing no real image pixels
    valid_rows = np.where(
        np.any(roi_mask, axis=1)
    )[0]

    if len(valid_rows) == 0:
        raise ValueError(
            "The selected ROI contains no valid image pixels."
        )

    top = valid_rows[0]
    bottom = valid_rows[-1] + 1

    analysis_roi = roi[
        top:bottom,
        :
    ]

    analysis_mask = roi_mask[
        top:bottom,
        :
    ]

    position_um = (
        np.arange(analysis_roi.shape[0])
        * pixel_size_um
    )

    return {
        "rotated": rotated,
        "analysis_roi": analysis_roi,
        "analysis_mask": analysis_mask,
        "x_um": position_um,
    }

def find_target_column_range(
    result,
    threshold_fraction=0.10,
    smooth_sigma_px=5,
    left_inner_fraction=0.02,
    right_inner_fraction=0.01,
):
    """
    Estimate the horizontal range occupied by the Argolight lines.

    The range is detected from the Raw image.

    A small additional inward margin can be applied independently
    on the left and right sides to avoid line-end deformations.

    Parameters
    ----------
    threshold_fraction :
        Relative threshold used to identify the horizontal target.

    smooth_sigma_px :
        Smoothing applied only to the horizontal intensity profile.

    left_inner_fraction :
        Fraction of the detected target width removed from the left.

    right_inner_fraction :
        Fraction of the detected target width removed from the right.
    """

    roi = np.asarray(
        result["analysis_roi"]
    )

    mask = np.asarray(
        result["analysis_mask"]
    )

    masked_roi = np.ma.array(
        roi,
        mask=~mask,
    )

    # Mean intensity as a function of horizontal position
    column_profile = masked_roi.mean(
        axis=0
    ).filled(np.nan)

    valid = np.isfinite(
        column_profile
    )

    if np.sum(valid) < 2:
        raise ValueError(
            "Not enough valid columns to detect the target."
        )

    # Fill only mask-generated NaNs for smoothing
    indices = np.arange(
        len(column_profile)
    )

    filled = column_profile.copy()

    filled[~valid] = np.interp(
        indices[~valid],
        indices[valid],
        column_profile[valid],
    )

    smooth = gaussian_filter1d(
        filled,
        sigma=smooth_sigma_px,
    )

    baseline = np.percentile(
        smooth[valid],
        5,
    )

    high = np.percentile(
        smooth[valid],
        95,
    )

    threshold = (
        baseline
        + threshold_fraction
        * (high - baseline)
    )

    active = np.where(
        valid
        & (smooth > threshold)
    )[0]

    if len(active) == 0:
        raise ValueError(
            "Argolight horizontal range not found."
        )

    # Keep the largest contiguous active region
    blocks = np.split(
        active,
        np.where(
            np.diff(active) > 1
        )[0] + 1,
    )

    block = max(
        blocks,
        key=len,
    )

    detected_left = int(
        block[0]
    )

    detected_right = int(
        block[-1] + 1
    )

    detected_width = (
        detected_right
        - detected_left
    )

    # Additional inward crop
    left_margin = int(
        round(
            left_inner_fraction
            * detected_width
        )
    )

    right_margin = int(
        round(
            right_inner_fraction
            * detected_width
        )
    )

    left = (
        detected_left
        + left_margin
    )

    right = (
        detected_right
        - right_margin
    )

    if right <= left:
        raise ValueError(
            "Inner margins removed the complete target."
        )

    return {
        "left": left,
        "right": right,

        "detected_left": detected_left,
        "detected_right": detected_right,

        "width_px": right - left,

        "column_profile": column_profile,
        "column_profile_smooth": smooth,
        "threshold": threshold,
    }

def estimate_strip_shifts(
    result, pair_centers_um, spacings_nm,
    reference_spacings=(300, 330, 360),
    strip_fraction=None, strip_width_px=None, col_range=None,
    smooth_sigma=0.5, peak_search_um=0.20, valley_search_um=0.12,
    min_peak_ratio=0.5, refine_px=1, confidence_level=0.95,
):
    """Estimate geometrical strip shifts from well-resolved Raw pairs."""

    strip_data = build_contrast_profiles(
        result=result, strip_fraction=strip_fraction,
        strip_width_px=strip_width_px, col_range=col_range,
    )

    profiles = strip_data["profiles"]
    strips = strip_data["strips"]
    masks = strip_data["masks"]

    x_um = np.asarray(result["x_um"])
    n_strips = len(profiles)

    measured_shift = np.full(n_strips, np.nan)
    shifts_per_strip = [[] for _ in range(n_strips)]

    for spacing_nm in reference_spacings:
        pair_index = int(np.argmin(np.abs(np.asarray(spacings_nm) - spacing_nm)))
        center_um = pair_centers_um[pair_index]
        spacing = spacings_nm[pair_index]

        for strip_index, profile in enumerate(profiles):
            _, info = pair_contrast(
                x_um=x_um, profile=profile, center_um=center_um, spacing_nm=spacing,
                strip_pixels=strips[strip_index], strip_mask=masks[strip_index],
                strip_shift_um=0.0, smooth_sigma=smooth_sigma,
                peak_search_um=peak_search_um, valley_search_um=valley_search_um,
                min_peak_ratio=min_peak_ratio, refine_px=refine_px,
                confidence_level=confidence_level,
            )

            if info["valid_pair"] and np.isfinite(info["common_shift_um"]):
                shifts_per_strip[strip_index].append(info["common_shift_um"])

    for strip_index, values in enumerate(shifts_per_strip):
        if len(values) > 0:
            measured_shift[strip_index] = np.median(values)

    valid = np.isfinite(measured_shift)

    if np.sum(valid) < 2:
        raise ValueError("Not enough valid strips to estimate geometrical shifts.")

    strip_index = np.arange(n_strips, dtype=float)
    slope, intercept = np.polyfit(strip_index[valid], measured_shift[valid], deg=1)
    fitted_shift = intercept + slope * strip_index

    strip_shifts_um = measured_shift.copy()
    strip_shifts_um[~valid] = fitted_shift[~valid]

    info = {
        "strip_index": strip_index,
        "measured_shift_um": measured_shift,
        "fitted_shift_um": fitted_shift,
        "final_shift_um": strip_shifts_um,
        "slope_um_per_strip": slope,
        "intercept_um": intercept,
        "reference_spacings": np.asarray(reference_spacings),
    }

    return strip_shifts_um, info

def estimate_pair_geometry(
    x_um,
    profile,
    n_pairs=14,
    smooth_sigma_px=6,
):
    """
    Estimate the approximate regular geometry of the Argolight target.

    A strong smoothing merges the two lines of each pair into
    a single broad peak.

    Returns
    -------
    first_center_um : float
    pair_pitch_um : float
    expected_centers_um : ndarray
    """

    smooth = gaussian_filter1d(
        profile,
        sigma=smooth_sigma_px,
    )

    # Roughly estimate the width occupied by the target
    baseline = np.percentile(smooth, 5)
    high = np.percentile(smooth, 99)

    threshold = baseline + 0.10 * (high - baseline)

    active = np.where(smooth > threshold)[0]

    if len(active) == 0:
        raise ValueError("Argolight target not found.")

    left = x_um[active[0]]
    right = x_um[active[-1]]

    # Approximate pitch from the total target width
    approximate_pitch = (right - left) / n_pairs

    dx = np.median(np.diff(x_um))

    min_distance_px = int(
        0.6 * approximate_pitch / dx
    )

    # Find the broad maxima corresponding to the pairs
    peaks, properties = find_peaks(
        smooth,
        distance=min_distance_px,
        prominence=0.02 * np.ptp(smooth),
    )

    if len(peaks) < n_pairs:
        raise ValueError(
            f"Found only {len(peaks)} pair candidates "
            f"instead of {n_pairs}. "
            "Try increasing smooth_sigma_px."
        )

    # Keep the 14 most prominent maxima
    strongest = np.argsort(
        properties["prominences"]
    )[-n_pairs:]

    measured_centers = np.sort(
        x_um[peaks[strongest]]
    )

    # Fit a regular grid:
    # center(i) = first_center + i * pitch
    i = np.arange(n_pairs)

    pair_pitch_um, first_center_um = np.polyfit(
        i,
        measured_centers,
        1,
    )

    expected_centers_um = (
        first_center_um
        + i * pair_pitch_um
    )

    return (
        first_center_um,
        pair_pitch_um,
        expected_centers_um,
    )

def refine_pair_centers(
    x_um,
    profile,
    expected_centers_um,
    min_offset_um=0.30,
    max_offset_um=0.85,
    smooth_sigma_px=0.5,
    max_center_shift_um=0.12,
):
    """
    Refine each Argolight pair center using the two local minima
    immediately surrounding the pair.

    The central region is excluded so that the internal P1-P2 valley
    of a resolved pair is not mistaken for a boundary.

    Parameters
    ----------
    min_offset_um :
        Minimum distance from the expected center.
        This excludes the internal valley of the line pair.

    max_offset_um :
        Maximum distance from the expected center.
        This avoids using the much more distant minima between cells.
    """

    # Very light smoothing only to avoid single-pixel fluctuations
    smooth = gaussian_filter1d(
        profile,
        sigma=smooth_sigma_px,
    )

    refined_centers = []
    left_minima = []
    right_minima = []

    for expected_center in expected_centers_um:

        # --------------------------------------------------
        # Left local minimum
        # --------------------------------------------------

        left_region = np.where(
            (x_um >= expected_center - max_offset_um)
            & (x_um <= expected_center - min_offset_um)
        )[0]

        left_idx = left_region[
            np.argmin(smooth[left_region])
        ]

        left = x_um[left_idx]

        # --------------------------------------------------
        # Right local minimum
        # --------------------------------------------------

        right_region = np.where(
            (x_um >= expected_center + min_offset_um)
            & (x_um <= expected_center + max_offset_um)
        )[0]

        right_idx = right_region[
            np.argmin(smooth[right_region])
        ]

        right = x_um[right_idx]

        # --------------------------------------------------
        # Center = midpoint between the two local minima
        # --------------------------------------------------

        center = (left + right) / 2

        # Keep the theoretical center if the correction is too large
        if abs(center - expected_center) > max_center_shift_um:
            center = expected_center

        refined_centers.append(center)
        left_minima.append(left)
        right_minima.append(right)

    return (
        np.asarray(refined_centers),
        np.asarray(left_minima),
        np.asarray(right_minima),
    )
def denoise_rotated_parallel_lines(
    result, sigma_along_px=1.0, median_size_along=1,
    blend=0.5, clip_negative=True,
):
    """
    Light directional denoising of an already rotated Argolight image.

    After the common rotation:
        axis 0 = direction across the lines, used for resolution;
        axis 1 = direction along the lines.

    Filtering is applied only along axis 1, so the P1-V-P2 profile
    across the lines is not directly blurred.

    The filtering is performed after rotation, avoiding any additional
    rotate-filter-rotate-back interpolation.

    Parameters
    ----------
    result : dict
        Output of prepare_rotated_roi().

    sigma_along_px : float
        Gaussian sigma along the line direction, in pixels.
        Set to 0 to disable Gaussian filtering.

    median_size_along : int
        Optional median-filter size along the lines.
        Use 1 to disable it.

    blend : float
        0 = original rotated image.
        1 = fully filtered image.
        Intermediate values provide a conservative denoising.

    clip_negative : bool
        Clip negative values to zero.

    Returns
    -------
    clean_result : dict
        Copy of result with the denoised analysis_roi.

        The original ROI is preserved in:
            clean_result["analysis_roi_before_denoise"]
    """

    if sigma_along_px < 0:
        raise ValueError("sigma_along_px must be >= 0.")

    if not 0 <= blend <= 1:
        raise ValueError("blend must be between 0 and 1.")

    if median_size_along < 1 or median_size_along % 2 == 0:
        raise ValueError("median_size_along must be an odd integer >= 1.")

    clean_result = result.copy()

    roi = np.asarray(result["analysis_roi"], dtype=np.float32)
    mask = np.asarray(result["analysis_mask"], dtype=bool)

    clean_result["analysis_roi_before_denoise"] = roi.copy()

    # Work only on valid rotated pixels.
    # Normalized filtering prevents masked rotation borders from
    # contaminating the useful image.
    data = np.where(mask, roi, 0.0).astype(np.float32)
    weight = mask.astype(np.float32)

    filtered = data.copy()

    # Optional removal of isolated spikes along the lines.
    if median_size_along > 1:
        filtered = median_filter(
            filtered,
            size=(1, median_size_along),
            mode="reflect",
        )

    # Gaussian smoothing only along the line direction.
    if sigma_along_px > 0:
        filtered_data = gaussian_filter1d(
            filtered,
            sigma=sigma_along_px,
            axis=1,
            mode="reflect",
        )

        filtered_weight = gaussian_filter1d(
            weight,
            sigma=sigma_along_px,
            axis=1,
            mode="reflect",
        )

        filtered = np.divide(
            filtered_data,
            filtered_weight,
            out=np.zeros_like(filtered_data),
            where=filtered_weight > 1e-6,
        )

    # Conservative blending with the original rotated data.
    clean = (1.0 - blend) * roi + blend * filtered

    # Preserve invalid pixels.
    clean[~mask] = roi[~mask]

    if clip_negative:
        clean = np.maximum(clean, 0)

    clean_result["analysis_roi"] = clean.astype(np.float32)

    clean_result["denoise_info"] = {
        "sigma_along_px": sigma_along_px,
        "median_size_along": median_size_along,
        "blend": blend,
    }

    return clean_result

def denoise_rbn_full_for_display(
    image, angle_deg,
    sigma_along_px=1.0,
    blend=0.7,
):
    """
    Apply the same directional RbN denoising to the full image
    and rotate it back to the original orientation.

    This function is intended only for visualization.
    """

    original_shape = image.shape

    # Rotate to align the Argolight lines
    rotated, valid_mask = rotate_image_and_mask(
        image=image,
        angle_deg=angle_deg,
    )

    data = np.where(valid_mask, rotated, 0.0).astype(np.float32)
    weight = valid_mask.astype(np.float32)

    filtered_data = gaussian_filter1d(
        data,
        sigma=sigma_along_px,
        axis=1,
        mode="reflect",
    )

    filtered_weight = gaussian_filter1d(
        weight,
        sigma=sigma_along_px,
        axis=1,
        mode="reflect",
    )

    filtered = np.divide(
        filtered_data,
        filtered_weight,
        out=np.zeros_like(filtered_data),
        where=filtered_weight > 1e-6,
    )

    clean_rotated = (1.0 - blend) * rotated + blend * filtered
    clean_rotated = np.maximum(clean_rotated, 0)

    # Rotate back to the original orientation
    clean_back = rotate(
        clean_rotated,
        -angle_deg,
        resize=True,
        preserve_range=True,
        order=1,
        mode="constant",
        cval=0,
    )

    # Center-crop back to the original image size
    ny, nx = clean_back.shape
    target_ny, target_nx = original_shape

    y0 = (ny - target_ny) // 2
    x0 = (nx - target_nx) // 2

    clean_back = clean_back[
        y0:y0 + target_ny,
        x0:x0 + target_nx,
    ]

    return clean_back

def compute_line_profile(
    result,
    col_range=None,
):
    """
    Calculate the global line profile.

    No strips are used.

    Mean and standard deviation are calculated directly
    over all valid pixels along the selected line length.
    """

    roi = np.asarray(
        result["analysis_roi"]
    )

    mask = np.asarray(
        result["analysis_mask"]
    )

    if col_range is not None:

        left, right = col_range

        roi = roi[
            :,
            left:right,
        ]

        mask = mask[
            :,
            left:right,
        ]

    masked_roi = np.ma.array(
        roi,
        mask=~mask,
    )

    mean = masked_roi.mean(
        axis=1
    ).filled(np.nan)

    std = masked_roi.std(
        axis=1
    ).filled(np.nan)

    result["mean"] = mean
    result["std"] = std

    result["profile_col_range"] = col_range

    return result


def build_contrast_profiles(
    result, strip_fraction=None, strip_width_px=None, col_range=None,
):
    """
    Build local profiles used for contrast analysis.

    The original strip pixels and masks are also returned so that
    P1-V and P2-V differences can be tested across the strip columns.
    """

    if strip_fraction is None and strip_width_px is None:
        raise ValueError("Specify strip_fraction or strip_width_px.")

    if strip_fraction is not None and strip_width_px is not None:
        raise ValueError("Specify only one between strip_fraction and strip_width_px.")

    roi = np.asarray(result["analysis_roi"])
    mask = np.asarray(result["analysis_mask"])

    if col_range is not None:
        col_left, col_right = col_range
        roi = roi[:, col_left:col_right]
        mask = mask[:, col_left:col_right]

    usable_width_px = roi.shape[1]

    if strip_fraction is not None:
        if not 0 < strip_fraction <= 1:
            raise ValueError("strip_fraction must be between 0 and 1.")
        n_strips = int(round(1.0 / strip_fraction))
    else:
        if strip_width_px <= 0:
            raise ValueError("strip_width_px must be positive.")
        n_strips = int(round(usable_width_px / strip_width_px))

    n_strips = max(1, min(n_strips, usable_width_px))
    edges = np.linspace(0, usable_width_px, n_strips + 1, dtype=int)

    profiles = []
    strip_pixels = []
    strip_masks = []
    strip_ranges = []

    for i in range(n_strips):
        left, right = edges[i], edges[i + 1]

        strip = roi[:, left:right]
        strip_mask = mask[:, left:right]

        if strip.shape[1] == 0:
            continue

        masked_strip = np.ma.array(strip, mask=~strip_mask)
        profile = masked_strip.mean(axis=1).filled(np.nan)

        valid = np.isfinite(profile)

        if np.sum(valid) < 2:
            continue

        # Fill only points missing because of rotation padding.
        if not np.all(valid):
            indices = np.arange(len(profile))
            profile[~valid] = np.interp(indices[~valid], indices[valid], profile[valid])

        profiles.append(profile)
        strip_pixels.append(np.asarray(strip))
        strip_masks.append(np.asarray(strip_mask))
        strip_ranges.append((left, right))

    profiles = np.asarray(profiles)
    widths_px = np.asarray([right - left for left, right in strip_ranges])

    return {
        "profiles": profiles,
        "strips": strip_pixels,
        "masks": strip_masks,
        "ranges": strip_ranges,
        "widths_px": widths_px,
        "n_strips": len(profiles),
        "usable_width_px": usable_width_px,
        "mean_width_px": np.mean(widths_px),
        "mean_fraction": np.mean(widths_px) / usable_width_px,
    }
    
def pair_contrast(
    x_um, profile, center_um, spacing_nm,
    strip_pixels=None, strip_mask=None, strip_shift_um=0.0,
    smooth_sigma=0.5, peak_search_um=0.20, valley_search_um=0.12,
    min_peak_ratio=0.5, refine_px=1, confidence_level=0.95,
):
    """
    Calculate contrast for one Argolight pair on one strip.

    Extrema are detected on the smoothed profile and refined on the
    original mean profile.

    P1-V and P2-V must also be significantly positive across the
    columns of the strip. A one-sided paired t criterion is used.

    confidence_level = 0.95 means that the lower one-sided 95%
    confidence bound for both P1-V and P2-V must be above zero.
    """

    if not 0.5 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0.5 and 1.")

    x_um = np.asarray(x_um)
    profile = np.asarray(profile)

    info = {
        "expected_p1_um": np.nan, "expected_p2_um": np.nan,
        "global_center_um": center_um, "strip_shift_um": strip_shift_um,
        "local_center_prior_um": np.nan,
        "p1_candidate": None, "valley_candidate": None, "p2_candidate": None,
        "p1": None, "valley": None, "p2": None,
        "local_midpoint_um": np.nan,
        "common_shift_um": np.nan, "translation_mismatch_um": np.nan,
        "peak_pair_score": np.nan,
        "valley_search_left_um": np.nan, "valley_search_right_um": np.nan,
        "peak_ratio": np.nan,
        "confidence_level": confidence_level,
        "p1_v_n": 0, "p2_v_n": 0,
        "p1_v_difference": np.nan, "p2_v_difference": np.nan,
        "p1_v_sem": np.nan, "p2_v_sem": np.nan,
        "p1_v_lower_bound": np.nan, "p2_v_lower_bound": np.nan,
        "p1_v_significant": False, "p2_v_significant": False,
        "valid_pair": False, "reason": "Not evaluated",
        "measured_spacing_um": np.nan,
        "spacing_error_um": np.nan,
    }

    if spacing_nm <= 0:
        info["reason"] = "Zero spacing"
        return 0.0, info

    spacing_um = spacing_nm / 1000.0
    local_center_prior = center_um + strip_shift_um
    expected_p1 = local_center_prior - spacing_um / 2
    expected_p2 = local_center_prior + spacing_um / 2

    info["local_center_prior_um"] = local_center_prior
    info["expected_p1_um"] = expected_p1
    info["expected_p2_um"] = expected_p2

    smooth = gaussian_filter1d(profile, sigma=smooth_sigma)

    # ------------------------------------------------------------
    # 1. Possible P1 and P2 candidates
    # ------------------------------------------------------------

    peaks, _ = find_peaks(smooth)

    # P1 must physically lie on the left side of the pair center.
    p1_pool = peaks[
        (np.abs(x_um[peaks] - expected_p1) <= peak_search_um)
        & (x_um[peaks] < local_center_prior)
    ]

    # P2 must physically lie on the right side of the pair center.
    p2_pool = peaks[
        (np.abs(x_um[peaks] - expected_p2) <= peak_search_um)
        & (x_um[peaks] > local_center_prior)
    ]

    if len(p1_pool) == 0:
        info["reason"] = "No P1 candidate in search region"
        return 0.0, info

    if len(p2_pool) == 0:
        info["reason"] = "No P2 candidate in search region"
        return 0.0, info

    # ------------------------------------------------------------
    # 2. Joint P1/P2 selection
    # ------------------------------------------------------------

    pair_candidates = []

    for p1_candidate in p1_pool:
        for p2_candidate in p2_pool:
            if p1_candidate >= p2_candidate:
                continue

            shift1 = x_um[p1_candidate] - expected_p1
            shift2 = x_um[p2_candidate] - expected_p2
            common_shift = (shift1 + shift2) / 2
            translation_mismatch = abs(shift1 - shift2)
            score = abs(common_shift) + translation_mismatch

            pair_candidates.append(
                (score, translation_mismatch, abs(common_shift),
                 p1_candidate, p2_candidate, common_shift)
            )

    if len(pair_candidates) == 0:
        info["reason"] = "No ordered P1-P2 candidate pair"
        return 0.0, info

    best = min(pair_candidates, key=lambda item: (item[0], item[1], item[2]))
    score, mismatch, _, p1_candidate, p2_candidate, common_shift = best

    info["p1_candidate"] = p1_candidate
    info["p2_candidate"] = p2_candidate
    info["common_shift_um"] = common_shift
    info["translation_mismatch_um"] = mismatch
    info["peak_pair_score"] = score

    # ------------------------------------------------------------
    # 3. Refine P1 and P2 on original mean profile
    # ------------------------------------------------------------

    p1_region = np.arange(
        max(0, p1_candidate - refine_px),
        min(len(profile), p1_candidate + refine_px + 1),
    )
    p2_region = np.arange(
        max(0, p2_candidate - refine_px),
        min(len(profile), p2_candidate + refine_px + 1),
    )

    if len(p1_region) == 0:
        info["reason"] = "P1 refinement failed"
        return 0.0, info

    if len(p2_region) == 0:
        info["reason"] = "P2 refinement failed"
        return 0.0, info

    p1 = p1_region[np.argmax(profile[p1_region])]
    p2 = p2_region[np.argmax(profile[p2_region])]

    info["p1"] = p1
    info["p2"] = p2
    
    measured_spacing_um = x_um[p2] - x_um[p1]

    info["measured_spacing_um"] = measured_spacing_um
    info["spacing_error_um"] = measured_spacing_um - spacing_um
        
    if not (x_um[p1] < local_center_prior < x_um[p2]):
        info["reason"] = "Refined P1/P2 do not bracket the pair center"
        return 0.0, info

    if p1 >= p2:
        info["reason"] = "Refined P1 and P2 are not ordered"
        return 0.0, info

    local_midpoint = (x_um[p1] + x_um[p2]) / 2
    info["local_midpoint_um"] = local_midpoint

    # ------------------------------------------------------------
    # 4. Valley candidate
    # ------------------------------------------------------------

    valleys, _ = find_peaks(-smooth)

    valley_search_left = min(local_center_prior, local_midpoint) - valley_search_um
    valley_search_right = max(local_center_prior, local_midpoint) + valley_search_um

    info["valley_search_left_um"] = valley_search_left
    info["valley_search_right_um"] = valley_search_right

    valley_pool = valleys[
        (valleys > p1)
        & (valleys < p2)
        & (x_um[valleys] >= valley_search_left)
        & (x_um[valleys] <= valley_search_right)
    ]

    if len(valley_pool) == 0:
        info["reason"] = "No valley candidate"
        return 0.0, info

    valley_scores = np.sqrt(
        (x_um[valley_pool] - local_center_prior) ** 2
        + (x_um[valley_pool] - local_midpoint) ** 2
    )

    valley_candidate = valley_pool[np.argmin(valley_scores)]
    info["valley_candidate"] = valley_candidate

    # ------------------------------------------------------------
    # 5. Refine valley on original mean profile
    # ------------------------------------------------------------

    valley_region = np.arange(
        max(0, valley_candidate - refine_px),
        min(len(profile), valley_candidate + refine_px + 1),
    )

    valley_region = valley_region[
        (valley_region > p1)
        & (valley_region < p2)
        & (x_um[valley_region] >= valley_search_left)
        & (x_um[valley_region] <= valley_search_right)
    ]

    if len(valley_region) == 0:
        info["reason"] = "Valley refinement failed"
        return 0.0, info

    valley = valley_region[np.argmin(profile[valley_region])]
    info["valley"] = valley

    # ------------------------------------------------------------
    # 6. Mean-profile intensity checks
    # ------------------------------------------------------------

    P1 = profile[p1]
    V = profile[valley]
    P2 = profile[p2]

    if P1 <= 0 or P2 <= 0:
        info["reason"] = "Non-positive peak intensity"
        return 0.0, info

    if V >= min(P1, P2):
        info["reason"] = "Valley not below both peaks"
        return 0.0, info

    peak_ratio = min(P1, P2) / max(P1, P2)
    info["peak_ratio"] = peak_ratio

    if peak_ratio < min_peak_ratio:
        info["reason"] = "Peak ratio too low"
        return 0.0, info

    # ------------------------------------------------------------
    # 7. Paired significance of P1-V and P2-V
    # ------------------------------------------------------------

    if strip_pixels is not None and strip_mask is not None:
        strip_pixels = np.asarray(strip_pixels)
        strip_mask = np.asarray(strip_mask)

        def paired_test(peak_idx, valley_idx):
            valid_cols = strip_mask[peak_idx] & strip_mask[valley_idx]
            peak_values = strip_pixels[peak_idx, valid_cols]
            valley_values = strip_pixels[valley_idx, valid_cols]

            finite = np.isfinite(peak_values) & np.isfinite(valley_values)
            differences = peak_values[finite] - valley_values[finite]
            n = len(differences)

            if n < 2:
                return False, n, np.nan, np.nan, np.nan

            mean_difference = np.mean(differences)
            std_difference = np.std(differences, ddof=1)
            sem_difference = std_difference / np.sqrt(n)

            # One-sided confidence quantile.
            quantile = t.ppf(confidence_level, df=n - 1)

            if sem_difference == 0:
                lower_bound = mean_difference
            else:
                lower_bound = mean_difference - quantile * sem_difference

            significant = lower_bound > 0

            return significant, n, mean_difference, sem_difference, lower_bound

        p1_sig, p1_n, p1_diff, p1_sem, p1_lower = paired_test(p1, valley)
        p2_sig, p2_n, p2_diff, p2_sem, p2_lower = paired_test(p2, valley)

        info["p1_v_n"] = p1_n
        info["p2_v_n"] = p2_n
        info["p1_v_difference"] = p1_diff
        info["p2_v_difference"] = p2_diff
        info["p1_v_sem"] = p1_sem
        info["p2_v_sem"] = p2_sem
        info["p1_v_lower_bound"] = p1_lower
        info["p2_v_lower_bound"] = p2_lower
        info["p1_v_significant"] = p1_sig
        info["p2_v_significant"] = p2_sig

        confidence_pct = 100 * confidence_level

        if not p1_sig:
            info["reason"] = f"P1-V not significant at {confidence_pct:.1f}% confidence"
            return 0.0, info

        if not p2_sig:
            info["reason"] = f"P2-V not significant at {confidence_pct:.1f}% confidence"
            return 0.0, info

    # ------------------------------------------------------------
    # 8. Contrast
    # ------------------------------------------------------------

    peak_mean = (P1 + P2) / 2
    contrast = (peak_mean - V) / peak_mean

    info["valid_pair"] = True
    info["reason"] = "Valid pair"

    return float(contrast), info
    
def compute_contrast(
    result, pair_centers_um, spacings_nm, strip_shifts_um=None,
    strip_fraction=None, strip_width_px=None, col_range=None,
    smooth_sigma=0.5, peak_search_um=0.20, valley_search_um=0.12,
    min_peak_ratio=0.5, refine_px=1, confidence_level=0.95,
    min_valid_fraction=0.20,
):
    """
    Calculate local and global Argolight contrast.

    Local level:
        Each strip is classified independently using pair_contrast().
        A strip is valid only if P1-V-P2 exists and both P1-V and P2-V
        are statistically significant.

    Global level:
        A spacing is considered resolved only if the fraction of valid
        strips is >= min_valid_fraction.

    Contrast:
        Mean and std are always calculated only over locally valid strips.
        The global contrast is set to zero when the spacing does not pass
        the global valid-fraction criterion.
    """

    if not 0 <= min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be between 0 and 1.")

    x = np.asarray(result["x_um"])

    strip_data = build_contrast_profiles(
        result=result,
        strip_fraction=strip_fraction,
        strip_width_px=strip_width_px,
        col_range=col_range,
    )

    profiles = strip_data["profiles"]
    strips = strip_data["strips"]
    masks = strip_data["masks"]

    n_strips = profiles.shape[0]
    n_pairs = len(spacings_nm)

    if strip_shifts_um is None:
        strip_shifts_um = np.zeros(n_strips, dtype=float)
    else:
        strip_shifts_um = np.asarray(strip_shifts_um, dtype=float)

    if len(strip_shifts_um) != n_strips:
        raise ValueError(
            f"strip_shifts_um has length {len(strip_shifts_um)}, "
            f"but {n_strips} strips were generated."
        )

    # Local results
    contrast_all = np.full((n_strips, n_pairs), np.nan, dtype=float)
    valid_pair_all = np.zeros((n_strips, n_pairs), dtype=bool)
    pair_info_all = [[None for _ in range(n_pairs)] for _ in range(n_strips)]

    for strip_index, profile in enumerate(profiles):
        for pair_index, (center, spacing) in enumerate(zip(pair_centers_um, spacings_nm)):

            contrast, info = pair_contrast(
                x_um=x,
                profile=profile,
                center_um=center,
                spacing_nm=spacing,
                strip_pixels=strips[strip_index],
                strip_mask=masks[strip_index],
                strip_shift_um=strip_shifts_um[strip_index],
                smooth_sigma=smooth_sigma,
                peak_search_um=peak_search_um,
                valley_search_um=valley_search_um,
                min_peak_ratio=min_peak_ratio,
                refine_px=refine_px,
                confidence_level=confidence_level,
            )

            pair_info_all[strip_index][pair_index] = info
            valid_pair_all[strip_index, pair_index] = info["valid_pair"]

            if info["valid_pair"]:
                contrast_all[strip_index, pair_index] = contrast

    # ========================================================
    # LOCAL AGGREGATION
    # ========================================================

    n_valid = np.sum(valid_pair_all, axis=0)
    valid_pair_fraction = n_valid / n_strips

    # Mean/std over locally valid strips, independently of the
    # global resolved/not-resolved decision.
    contrast_valid_mean = np.zeros(n_pairs, dtype=float)
    contrast_valid_std = np.zeros(n_pairs, dtype=float)

    for pair_index in range(n_pairs):
        valid = valid_pair_all[:, pair_index]

        if np.any(valid):
            values = contrast_all[valid, pair_index]
            contrast_valid_mean[pair_index] = np.mean(values)
            contrast_valid_std[pair_index] = np.std(values)

    # ========================================================
    # GLOBAL DECISION
    # ========================================================

    globally_resolved = valid_pair_fraction >= min_valid_fraction

    # These are the values used for the final contrast plot.
    contrast_mean = np.where(globally_resolved, contrast_valid_mean, 0.0)
    contrast_std = np.where(globally_resolved, contrast_valid_std, 0.0)

    # ========================================================
    # STORE RESULTS
    # ========================================================

    result["contrast_profiles"] = profiles
    result["contrast_strip_pixels"] = strips
    result["contrast_strip_masks"] = masks
    result["strip_shifts_um"] = strip_shifts_um

    result["contrast_all"] = contrast_all
    result["valid_pair_all"] = valid_pair_all
    result["pair_info_all"] = pair_info_all

    result["contrast_n_valid"] = n_valid
    result["valid_pair_fraction"] = valid_pair_fraction

    # Contrast measured wherever the structure is locally resolved
    result["contrast_valid_mean"] = contrast_valid_mean
    result["contrast_valid_std"] = contrast_valid_std

    # Global resolution decision
    result["globally_resolved"] = globally_resolved
    result["min_valid_fraction"] = min_valid_fraction

    # Final values used in the global contrast plot
    result["contrast_mean"] = contrast_mean
    result["contrast_std"] = contrast_std

    result["strip_ranges"] = strip_data["ranges"]
    result["strip_widths_px"] = strip_data["widths_px"]
    result["n_strips"] = strip_data["n_strips"]
    result["usable_width_px"] = strip_data["usable_width_px"]
    result["mean_strip_width_px"] = strip_data["mean_width_px"]
    result["mean_strip_fraction"] = strip_data["mean_fraction"]

    return result

def extract_longitudinal_profiles(
    results,
    pair_centers_um,
    spacings_nm,
    col_range,
    reference_name="Raw",
    transverse_half_width_px=1,
    smooth_sigma=0.5,
    refine_px=1,
):
    """
    Extract one longitudinal reference-line profile for each Argolight cell.

    The reference line is the leftmost peak located on the left side of
    each pair center, inside the corresponding Argolight cell.

    Reference-line positions are detected only on the Raw image and then
    frozen for all reconstruction methods.

    No distinction between resolved and unresolved pairs is used.

    transverse_half_width_px=1 means averaging three transverse pixels:
    reference-1, reference, reference+1.
    """

    if reference_name not in results:
        raise ValueError(
            f"Reference image '{reference_name}' not found in results."
        )

    if transverse_half_width_px < 0:
        raise ValueError(
            "transverse_half_width_px must be >= 0."
        )

    pair_centers_um = np.asarray(
        pair_centers_um,
        dtype=float,
    )

    spacings_nm = np.asarray(
        spacings_nm,
        dtype=float,
    )

    if len(pair_centers_um) != len(spacings_nm):
        raise ValueError(
            "pair_centers_um and spacings_nm must have the same length."
        )

    # ============================================================
    # REFERENCE TRANSVERSE PROFILE FROM RAW
    # ============================================================

    reference_result = results[reference_name]

    x_um = np.asarray(
        reference_result["x_um"]
    )

    reference_roi = np.asarray(
        reference_result["analysis_roi"]
    )

    reference_mask = np.asarray(
        reference_result["analysis_mask"],
        dtype=bool,
    )

    col_left, col_right = col_range

    roi_used = reference_roi[
        :,
        col_left:col_right,
    ]

    mask_used = reference_mask[
        :,
        col_left:col_right,
    ]

    mean_profile = np.ma.array(
        roi_used,
        mask=~mask_used,
    ).mean(axis=1).filled(np.nan)

    valid = np.isfinite(mean_profile)

    if np.sum(valid) < 2:
        raise ValueError(
            "Not enough valid pixels in the Raw transverse profile."
        )

    if not np.all(valid):
        indices = np.arange(
            len(mean_profile)
        )

        mean_profile[~valid] = np.interp(
            indices[~valid],
            indices[valid],
            mean_profile[valid],
        )

    smooth_profile = gaussian_filter1d(
        mean_profile,
        sigma=smooth_sigma,
    )

    # All transverse peaks in the Raw profile
    peaks, _ = find_peaks(
        smooth_profile
    )

    # ============================================================
    # CELL BOUNDARIES
    # ============================================================

    n_pairs = len(pair_centers_um)

    cell_left = np.empty(
        n_pairs,
        dtype=float,
    )

    cell_right = np.empty(
        n_pairs,
        dtype=float,
    )

    for i in range(n_pairs):

        if i == 0:
            pitch_left = (
                pair_centers_um[1]
                - pair_centers_um[0]
            )

            cell_left[i] = (
                pair_centers_um[i]
                - pitch_left / 2
            )

        else:
            cell_left[i] = 0.5 * (
                pair_centers_um[i - 1]
                + pair_centers_um[i]
            )

        if i == n_pairs - 1:
            pitch_right = (
                pair_centers_um[-1]
                - pair_centers_um[-2]
            )

            cell_right[i] = (
                pair_centers_um[i]
                + pitch_right / 2
            )

        else:
            cell_right[i] = 0.5 * (
                pair_centers_um[i]
                + pair_centers_um[i + 1]
            )

    # ============================================================
    # FIND THE LEFT REFERENCE LINE IN EACH CELL
    # ============================================================

    reference_indices = []
    reference_positions_um = []

    for pair_index, center_um in enumerate(
        pair_centers_um
    ):

        # Peaks inside this cell and on the left side of its center
        candidates = peaks[
            (x_um[peaks] >= cell_left[pair_index])
            & (x_um[peaks] < center_um)
        ]

        if len(candidates) == 0:
            raise ValueError(
                f"No left reference line found for "
                f"{spacings_nm[pair_index]:.0f} nm."
            )

        # First line from the left in the zoom/cell
        candidate = candidates[
            np.argmin(
                x_um[candidates]
            )
        ]

        # Refine on the original Raw transverse profile
        refine_region = np.arange(
            max(
                0,
                candidate - refine_px,
            ),
            min(
                len(mean_profile),
                candidate + refine_px + 1,
            ),
        )

        reference_index = refine_region[
            np.argmax(
                mean_profile[refine_region]
            )
        ]

        reference_indices.append(
            int(reference_index)
        )

        reference_positions_um.append(
            float(x_um[reference_index])
        )

    reference_indices = np.asarray(
        reference_indices,
        dtype=int,
    )

    reference_positions_um = np.asarray(
        reference_positions_um,
        dtype=float,
    )

    # ============================================================
    # EXTRACT SAME PHYSICAL LINE FROM EVERY METHOD
    # ============================================================

    output = {}

    for name, result in results.items():

        roi = np.asarray(
            result["analysis_roi"]
        )

        mask = np.asarray(
            result["analysis_mask"],
            dtype=bool,
        )

        method_x_um = np.asarray(
            result["x_um"]
        )

        if len(method_x_um) != len(x_um):
            raise ValueError(
                f"{name}: transverse geometry differs from Raw."
            )

        pixel_size_um = np.median(
            np.diff(method_x_um)
        )

        y_um = (
            np.arange(
                col_right - col_left
            )
            * pixel_size_um
        )

        def extract_profile(line_index):

            row_left = max(
                0,
                line_index
                - transverse_half_width_px,
            )

            row_right = min(
                roi.shape[0],
                line_index
                + transverse_half_width_px
                + 1,
            )

            data = roi[
                row_left:row_right,
                col_left:col_right,
            ]

            valid_mask = mask[
                row_left:row_right,
                col_left:col_right,
            ]

            return np.ma.array(
                data,
                mask=~valid_mask,
            ).mean(axis=0).filled(np.nan)

        method_output = {}

        for pair_index, spacing_nm in enumerate(
            spacings_nm
        ):

            line_index = (
                reference_indices[pair_index]
            )

            profile = extract_profile(
                line_index
            )

            method_output[
                float(spacing_nm)
            ] = {
                "spacing_nm": float(spacing_nm),

                "y_um": y_um,

                "reference_index": int(
                    line_index
                ),

                "reference_um": float(
                    reference_positions_um[
                        pair_index
                    ]
                ),

                "reference_profile": profile,

                # Kept for compatibility with the plotting function
                "plot_profile": profile.copy(),
            }

        output[name] = method_output

    return output

def compute_longitudinal_metrics(
    longitudinal_profiles,
    spacings_nm,
):
    """
    Calculate the normalized longitudinal standard deviation.

    The same type of reference line is analyzed for every spacing,
    independently of whether the central Argolight pair is resolved.

    Each longitudinal profile is normalized by its median.
    """

    spacings_nm = np.asarray(
        spacings_nm,
        dtype=float,
    )

    output = {}

    for name, method_profiles in longitudinal_profiles.items():

        method_output = {}

        for spacing_nm in spacings_nm:

            data = method_profiles[
                float(spacing_nm)
            ]

            profile = np.asarray(
                data["reference_profile"],
                dtype=float,
            )

            valid = np.isfinite(
                profile
            )

            entry = {
                "spacing_nm": float(
                    spacing_nm
                ),
                "normalized_profile": None,
                "std": np.nan,
            }

            if np.sum(valid) > 0:

                profile_valid = (
                    profile[valid]
                )

                reference = np.median(
                    profile_valid
                )

                if (
                    np.isfinite(reference)
                    and reference > 0
                ):

                    normalized = (
                        profile_valid
                        / reference
                    )

                    entry[
                        "normalized_profile"
                    ] = normalized

                    entry["std"] = np.std(
                        normalized
                    )

            method_output[
                float(spacing_nm)
            ] = entry

        output[name] = method_output

    return output

def compute_longitudinal_hole_metric(
    longitudinal_profiles,
    spacings_nm,
    smooth_sigma_px=2.0,
    baseline_sigma_px=20.0,
):
    """
    Calculate the longitudinal hole RMS on the reference line.

    The same type of lateral reference line is used for every spacing,
    independently of the resolution of the central line pair.
    """

    spacings_nm = np.asarray(
        spacings_nm,
        dtype=float,
    )

    output = {}

    def analyze_profile(profile):

        profile = np.asarray(
            profile,
            dtype=float,
        ).copy()

        valid = np.isfinite(
            profile
        )

        if np.sum(valid) < 2:
            return None

        if not np.all(valid):

            indices = np.arange(
                len(profile)
            )

            profile[~valid] = np.interp(
                indices[~valid],
                indices[valid],
                profile[valid],
            )

        reference = np.median(
            profile
        )

        if (
            not np.isfinite(reference)
            or reference <= 0
        ):
            return None

        normalized = (
            profile / reference
        )

        smooth = gaussian_filter1d(
            normalized,
            sigma=smooth_sigma_px,
            mode="reflect",
        )

        baseline = gaussian_filter1d(
            smooth,
            sigma=baseline_sigma_px,
            mode="reflect",
        )

        deficit = np.maximum(
            (baseline - smooth)
            / baseline,
            0.0,
        )

        hole_rms = np.sqrt(
            np.mean(
                deficit**2
            )
        )

        return {
            "normalized": normalized,
            "smooth": smooth,
            "baseline": baseline,
            "deficit": deficit,
            "hole_rms": hole_rms,
        }

    for name, method_profiles in longitudinal_profiles.items():

        method_output = {}

        for spacing_nm in spacings_nm:

            data = method_profiles[
                float(spacing_nm)
            ]

            analysis = analyze_profile(
                data["reference_profile"]
            )

            entry = {
                "spacing_nm": float(
                    spacing_nm
                ),
                "reference": analysis,
                "hole_rms": np.nan,
            }

            if analysis is not None:
                entry["hole_rms"] = (
                    analysis["hole_rms"]
                )

            method_output[
                float(spacing_nm)
            ] = entry

        output[name] = method_output

    return output

def cmap_from_color(color, min_strength=0.30, name="custom"):
    """
    Create a colormap from a light version of the base color
    to the original base color.
    """
    rgb = np.asarray(to_rgb(color))
    light_rgb = (1 - min_strength) * np.ones(3) + min_strength * rgb

    return LinearSegmentedColormap.from_list(
        name,
        [light_rgb, rgb],
    )

def plot_full_images_groups(display_images, groups, method_labels, colors, pxsize_um,
                            cmap="bone", vmax_dict=None, figsize=(24, 12)):
    """Plot full images for all reconstruction groups, allowing a different number of methods per row."""

    n_groups = len(groups)
    max_methods = max(len(names) for names in groups.values())

    fig, axes = plt.subplots(n_groups, max_methods, figsize=figsize, squeeze=False)

    for row, (group_name, names) in enumerate(groups.items()):

        # Plot all methods available for this group
        for col, name in enumerate(names):
            ax = axes[row, col]
            vmax = None if vmax_dict is None else vmax_dict.get(name)

            gra.ShowImg(display_images[name], pxsize_um, clabel="Counts",
                        fig=fig, ax=ax, cmap=cmap, vmax=vmax)

            ax.set_title(method_labels[name], fontsize=15, color=colors[name])

        # Hide unused axes when a group contains fewer methods
        for col in range(len(names), max_methods):
            axes[row, col].axis("off")

        # Group label on the left side of each row
        axes[row, 0].text(-0.10, 0.5, group_name, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="center",
                          fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig, axes

def plot_longitudinal_profiles(longitudinal_profiles, groups, method_labels, spacings_nm, colors,
                               min_color_strength=0.30, normalize=False,
                               share_y_between_groups=False, figsize=(24, 11)):
    """Plot longitudinal profiles for all methods, with one curve for each Argolight spacing."""

    n_groups = len(groups)
    max_methods = max(len(names) for names in groups.values())
    sharey = True if share_y_between_groups else "row"

    fig, axes = plt.subplots(n_groups, max_methods, figsize=figsize,
                             sharex=True, sharey=sharey, squeeze=False)

    for row, (group_name, names) in enumerate(groups.items()):
        for col, name in enumerate(names):
            ax = axes[row, col]
            base_color = np.array(mcolors.to_rgb(colors[name]))

            for i, spacing_nm in enumerate(spacings_nm):
                info = longitudinal_profiles[name][float(spacing_nm)]
                y_um = np.asarray(info["y_um"])
                profile = np.asarray(info["plot_profile"], dtype=float).copy()

                # Normalize each longitudinal profile by its median intensity
                if normalize:
                    finite = np.isfinite(profile)
                    median = np.median(profile[finite]) if np.any(finite) else np.nan
                    if np.isfinite(median) and median != 0:
                        profile /= median

                # Increase color strength progressively with nominal spacing
                fraction = i / max(len(spacings_nm) - 1, 1)
                strength = min_color_strength + (1 - min_color_strength) * fraction
                curve_color = 1 - strength * (1 - base_color)

                ax.plot(y_um, profile, color=curve_color, lw=1.2,
                        label=f"{spacing_nm:.0f} nm")

            ax.set_title(method_labels[name], color=colors[name], fontweight="bold")
            ax.set_xlabel(r"Position along line ($\mu$m)")
            ax.grid(alpha=0.2)

            if col == 0:
                ylabel = "Normalized intensity" if normalize else "Intensity"
                ax.set_ylabel(f"{group_name}\n{ylabel}")

        # Hide unused axes when a group contains fewer methods
        for col in range(len(names), max_methods):
            axes[row, col].axis("off")

    # Common legend: spacing colors are shown using the first method
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, axes

def plot_longitudinal_std(metric_data, groups, method_labels, spacings_nm, colors,
                          group_names=None, shared_y=True, y_ranges=None,
                          panel_width_mm=51.389, panel_height_mm=31.632,
                          top_fraction=0.30, tick_fontsize=6, display_scale=1.45):
    """
    Plot longitudinal normalized STD with a broken y axis.

    Late reconstructions are shown on the upper axis and all other methods
    on the lower axis. With shared_y=True all groups use common y ranges.
    With shared_y=False group-specific ranges can be supplied through y_ranges.
    """

    selected_groups = groups if group_names is None else {g: groups[g] for g in group_names}
    n_groups = len(selected_groups)
    bottom_fraction = 1.0 - top_fraction
    panel_aspect = panel_height_mm / panel_width_mm

    panel_w_in = panel_width_mm / 25.4
    panel_h_in = panel_height_mm / 25.4

    fig, axes = plt.subplots(
        2, n_groups,
        figsize=(panel_w_in * n_groups * display_scale,
                 panel_h_in * display_scale),
        sharex="col", squeeze=False,
        gridspec_kw={"height_ratios": [top_fraction, bottom_fraction], "hspace": 0.0},
        layout="constrained",
    )

    ax_top, ax_bottom = axes[0], axes[1]

    def get_values(names, late):
        """Collect finite STD values for late or non-late reconstructions."""
        values = []
        for name in names:
            if ("late" in name.lower()) != late:
                continue
            y = np.array([metric_data[name][float(s)]["std"] for s in spacings_nm], dtype=float)
            values.extend(y[np.isfinite(y)])
        return np.asarray(values)

    def auto_lower(values):
        if values.size == 0:
            return (0, 1)
        return (0, 1.10 * np.max(values))

    def auto_upper(values):
        if values.size == 0:
            return (0, 1)
        ymin, ymax = np.min(values), np.max(values)
        span = max(ymax - ymin, 0.1 * ymax, 1e-6)
        return (max(0, ymin - 0.15 * span), ymax + 0.15 * span)

    # Common ranges for direct comparison among groups
    if shared_y:
        all_names = [name for names in selected_groups.values() for name in names]
        common_lower = auto_lower(get_values(all_names, late=False))
        common_upper = auto_upper(get_values(all_names, late=True))

    for col, (group_name, names) in enumerate(selected_groups.items()):
        top, bottom = ax_top[col], ax_bottom[col]
        legend_handles, legend_labels = [], []

        for name in names:
            y = np.array([metric_data[name][float(s)]["std"] for s in spacings_nm], dtype=float)
            finite = np.isfinite(y)
            ax = top if "late" in name.lower() else bottom

            line, = ax.plot(
                np.asarray(spacings_nm)[finite], y[finite], "o-",
                color=colors[name], lw=1.5, ms=3,
            )
            legend_handles.append(line)
            legend_labels.append(method_labels[name])

        # Shared ranges for comparison, otherwise manual or automatic group ranges
        if shared_y:
            lower_ylim, upper_ylim = common_lower, common_upper
        elif y_ranges is not None and group_name in y_ranges:
            lower_ylim = y_ranges[group_name]["lower"]
            upper_ylim = y_ranges[group_name]["upper"]
        else:
            lower_ylim = auto_lower(get_values(names, late=False))
            upper_ylim = auto_upper(get_values(names, late=True))

        bottom.set_ylim(*lower_ylim)
        top.set_ylim(*upper_ylim)

        # Preserve the requested aspect ratio of the total plotting area
        top.set_box_aspect(panel_aspect * top_fraction)
        bottom.set_box_aspect(panel_aspect * bottom_fraction)

        top.set_title(group_name, fontsize=8)
        bottom.set_xlabel("Spacing (nm)", fontsize=7)
        bottom.set_xticks(spacings_nm)
        bottom.tick_params(axis="x", labelsize=tick_fontsize)
        bottom.tick_params(axis="y", labelsize=tick_fontsize)
        top.tick_params(axis="y", labelsize=tick_fontsize)

        top.grid(alpha=0.2)
        bottom.grid(alpha=0.2)

        # Broken-axis appearance
        top.spines["bottom"].set_visible(False)
        bottom.spines["top"].set_visible(False)
        top.tick_params(axis="x", bottom=False, labelbottom=False)

        d = 0.012
        kwargs = dict(color="k", clip_on=False, lw=0.8)
        top.plot((-d, +d), (-d, +d), transform=top.transAxes, **kwargs)
        top.plot((1-d, 1+d), (-d, +d), transform=top.transAxes, **kwargs)
        bottom.plot((-d, +d), (1-d, 1+d), transform=bottom.transAxes, **kwargs)
        bottom.plot((1-d, 1+d), (1-d, 1+d), transform=bottom.transAxes, **kwargs)

        bottom.legend(legend_handles, legend_labels, fontsize=6)

    # One single y label for the complete broken axis
    fig.supylabel("Longitudinal normalized std", fontsize=7)

    return fig, axes
   
def plot_analysis_roi(
    result,
    col_range,
    title="Region used for analysis",
    cmap="bone",
    figsize=(7, 7),
):
    """Plot the rotated ROI and the horizontal region used for the analysis."""

    roi = np.asarray(result["analysis_roi"])
    mask = np.asarray(result["analysis_mask"])
    left, right = col_range

    display = np.ma.array(
        roi,
        mask=~mask,
    )

    fig, ax = plt.subplots(
        figsize=figsize
    )

    ax.imshow(
        display,
        cmap=cmap,
        aspect="auto",
    )

    ax.axvline(
        left,
        color="red",
        ls="--",
        lw=1.5,
        label="Analysis range",
    )

    ax.axvline(
        right,
        color="red",
        ls="--",
        lw=1.5,
    )

    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    return fig, ax

def plot_profiles(results, groups, method_labels, pair_centers_um, spacings_nm,
                  zoom_spacing_nm, colors, figsize=(24, 11)):
    """Plot transverse profiles for all groups, allowing a different number of methods per row."""

    n_groups = len(groups)
    max_methods = max(len(names) for names in groups.values())
    fig, axes = plt.subplots(n_groups, max_methods, figsize=figsize,
                             sharey="row", squeeze=False)

    pair_pitch = np.median(np.diff(pair_centers_um))

    for row, (group_name, names) in enumerate(groups.items()):
        # Select the Argolight pair to show in the inset for this group
        zoom_spacing = zoom_spacing_nm[group_name]
        zoom_idx = int(np.argmin(np.abs(np.asarray(spacings_nm) - zoom_spacing)))
        zoom_center = pair_centers_um[zoom_idx]
        zoom_left = zoom_center - pair_pitch / 2
        zoom_right = zoom_center + pair_pitch / 2

        for col, name in enumerate(names):
            ax = axes[row, col]
            result = results[name]
            x = result["x_um"]
            mean = result["mean"]
            std = result["std"]
            color = colors[name]

            # Mean transverse profile with ±1 standard deviation
            ax.plot(x, mean, color=color, lw=1.5)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.20)
            ax.set_title(method_labels[name], color=color, fontweight="bold")
            ax.set_xlabel(r"Position ($\mu$m)")
            ax.grid(alpha=0.2)

            # Top axis reports the nominal spacing of each Argolight pair
            ax_top = ax.twiny()
            ax_top.set_xlim(ax.get_xlim())
            ax_top.set_xticks(pair_centers_um)
            ax_top.set_xticklabels([f"{s:.0f}" for s in spacings_nm],
                                   rotation=45, fontsize=7)
            ax_top.set_xlabel("Spacing (nm)")

            # Inset around the selected pair
            local = (x >= zoom_left) & (x <= zoom_right)
            axins = inset_axes(ax, width="40%", height="40%", loc="upper right")
            axins.plot(x, mean, color=color, lw=1.5)
            axins.fill_between(x, mean - std, mean + std, color=color, alpha=0.20)
            axins.set_xlim(zoom_left, zoom_right)

            if np.any(local):
                ymin = np.min(mean[local] - std[local])
                ymax = np.max(mean[local] + std[local])
                padding = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
                axins.set_ylim(ymin - padding, ymax + padding)

            axins.set_xticks([])
            axins.set_yticks([])
            axins.set_title(f"{spacings_nm[zoom_idx]:.0f} nm", fontsize=8)
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        # Hide empty panels for groups containing fewer methods
        for col in range(len(names), max_methods):
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(f"{group_name}\nPhoton counts")

    fig.tight_layout()
    return fig, axes

def plot_contrast(results, groups, method_labels, spacings_nm, colors,
                  group_names=None, shared_y=True,
                  panel_width_mm=51.389, panel_height_mm=31.632,
                  y_padding=0.05, tick_fontsize=6, display_scale=1.45):
    """
    Plot contrast curves with fixed panel proportions.

    If shared_y=True, all groups use the same y range for direct comparison.
    If shared_y=False, each group uses its own y range to reduce white space.
    group_names can be used to select only specific groups.
    """

    # Select groups while preserving their original order
    selected_groups = groups if group_names is None else {g: groups[g] for g in group_names}
    n_groups = len(selected_groups)

    # Required aspect ratio of the plotting area, excluding ticks and axis labels
    panel_aspect = panel_height_mm / panel_width_mm
    panel_w_in = panel_width_mm / 25.4
    panel_h_in = panel_height_mm / 25.4

    # External figure size; set_box_aspect() fixes the actual plotting-area proportions
    fig, axes = plt.subplots(
        1, n_groups,
        figsize=(panel_w_in * n_groups * display_scale, panel_h_in * display_scale),
        squeeze=False, layout="constrained",
    )
    axes = axes[0]

    def get_group_max(names):
        """Maximum contrast including the upper standard-deviation band."""
        ymax = 0.0
        for name in names:
            mean = np.asarray(results[name]["contrast_mean"], dtype=float)
            std = np.asarray(results[name]["contrast_std"], dtype=float)
            upper = mean + std
            finite = np.isfinite(upper)
            if np.any(finite):
                ymax = max(ymax, np.nanmax(upper[finite]))
        return ymax

    # Common upper y limit for direct comparison among groups
    if shared_y:
        global_max = max(get_group_max(names) for names in selected_groups.values())
        global_ymax = global_max * (1 + y_padding) if global_max > 0 else 1.0

    for ax, (group_name, names) in zip(axes, selected_groups.items()):
        for name in names:
            mean = np.asarray(results[name]["contrast_mean"], dtype=float)
            std = np.asarray(results[name]["contrast_std"], dtype=float)
            finite = np.isfinite(mean)

            ax.plot(
                np.asarray(spacings_nm)[finite], mean[finite], "o-",
                color=colors[name], lw=1.5, ms=3,
                label=method_labels[name],
            )

            ax.fill_between(
                np.asarray(spacings_nm)[finite],
                (mean - std)[finite],
                (mean + std)[finite],
                color=colors[name], alpha=0.18,
            )

        # Common or independently optimized y scale
        if shared_y:
            ymax = global_ymax
        else:
            local_max = get_group_max(names)
            ymax = local_max * (1 + y_padding) if local_max > 0 else 1.0

        ax.set_ylim(0, ymax)

        # Fixed W/H proportion of the plotting area
        ax.set_box_aspect(panel_aspect)

        ax.set_title(group_name, fontsize=8)
        ax.set_xlabel("Spacing (nm)", fontsize=7)
        ax.set_ylabel("Contrast", fontsize=7)

        ax.set_xticks(spacings_nm)
        ax.tick_params(axis="x", labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelsize=tick_fontsize)

        ax.grid(alpha=0.2)
        ax.legend(fontsize=6)

    return fig, axes

def plot_valid_pair_fraction(
    results, groups, method_labels, spacings_nm, colors,
    min_valid_fraction=None, figsize=(18, 5),
):
    """Plot valid-pair fraction for all groups with a common y-axis."""

    n_groups = len(groups)

    fig, axes = plt.subplots(
        1, n_groups,
        figsize=figsize,
        sharey=True,
        squeeze=False,
    )

    axes = axes.ravel()

    for ax, (group_name, names) in zip(axes, groups.items()):

        for name in names:
            ax.plot(
                spacings_nm,
                results[name]["valid_pair_fraction"],
                "o-",
                color=colors[name],
                lw=2,
                ms=4,
                label=method_labels[name],
            )

        if min_valid_fraction is not None:
            ax.axhline(
                min_valid_fraction,
                color="0.4",
                ls="--",
                lw=1.5,
                label=f"Threshold = {min_valid_fraction:.2f}",
            )

        ax.set_title(group_name)
        ax.set_xlabel("Spacing (nm)")
        ax.set_xticks(spacings_nm)
        ax.grid(alpha=0.2)
        ax.legend()

    axes[0].set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("Valid pair fraction")

    fig.tight_layout()
    return fig, axes

def plot_rotated_images_with_zoom(
    results, pair_centers_um, spacings_nm, col_range, zoom_spacing_nm,
    colors=None, zoom_half_width_um=None, cmap="bone",
    display_percentiles=(1, 99.5), zoom_percentiles=(1, 99.5),
    full_aspect_ratio=3.0, figsize=(14, 10),
):
    """
    Plot the complete Argolight target and a square zoom.

    Full targets and zooms are independently scaled for each method.
    zoom_percentiles controls the display range of each individual zoom.
    """

    names = list(results.keys())
    n_images = len(names)
    pair_centers_um = np.asarray(pair_centers_um, dtype=float)
    spacings_nm = np.asarray(spacings_nm, dtype=float)

    x_um = np.asarray(results[names[0]]["x_um"])
    dx_um = np.median(np.diff(x_um))
    pair_pitch_um = np.median(np.diff(pair_centers_um))

    # Default zoom width = one complete Argolight cell
    if zoom_half_width_um is None:
        zoom_half_width_um = pair_pitch_um / 2
    zoom_width_um = 2 * zoom_half_width_um

    # Complete transverse target range
    target_x_min = pair_centers_um[0] - pair_pitch_um / 2
    target_x_max = pair_centers_um[-1] + pair_pitch_um / 2
    row_indices = np.where((x_um >= target_x_min) & (x_um <= target_x_max))[0]

    if len(row_indices) == 0:
        raise ValueError("Argolight target range not found.")

    row_start, row_stop = row_indices[0], row_indices[-1] + 1
    full_width_um = x_um[row_stop - 1] + dx_um - x_um[row_start]

    # Central portion along the lines used for full-target display
    col_left, col_right = col_range
    available_width_px = col_right - col_left
    available_height_um = available_width_px * dx_um

    full_height_um = full_width_um / full_aspect_ratio
    full_height_um = max(full_height_um, zoom_width_um)
    full_height_um = min(full_height_um, available_height_um)

    full_height_px = int(round(full_height_um / dx_um))
    full_height_px = max(1, min(full_height_px, available_width_px))

    col_center = (col_left + col_right) // 2
    display_col_left = col_center - full_height_px // 2
    display_col_right = display_col_left + full_height_px

    if display_col_left < col_left:
        display_col_left = col_left
        display_col_right = col_left + full_height_px

    if display_col_right > col_right:
        display_col_right = col_right
        display_col_left = col_right - full_height_px

    actual_height_um = (display_col_right - display_col_left) * dx_um

    # Square zoom around the selected Argolight pair
    zoom_idx = int(np.argmin(np.abs(spacings_nm - zoom_spacing_nm)))
    zoom_spacing = spacings_nm[zoom_idx]
    zoom_center_x = pair_centers_um[zoom_idx]

    zoom_x_min = zoom_center_x - zoom_half_width_um
    zoom_x_max = zoom_center_x + zoom_half_width_um
    zoom_center_y = actual_height_um / 2
    zoom_y_min = zoom_center_y - zoom_width_um / 2
    zoom_y_max = zoom_center_y + zoom_width_um / 2

    fig, axes = plt.subplots(
        n_images, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [3.0, 1.0], "wspace": 0.10},
        constrained_layout=True,
    )

    if n_images == 1:
        axes = np.asarray([axes])

    for row, name in enumerate(names):
        roi = np.asarray(results[name]["analysis_roi"])
        target = roi[row_start:row_stop, display_col_left:display_col_right]
        display = target.T

        finite = display[np.isfinite(display)]
        if finite.size == 0:
            raise ValueError(f"{name}: no finite pixels available for display.")

        # Independent scaling for the full target
        full_vmin, full_vmax = np.percentile(finite, display_percentiles)

        # Extract the exact region used in the zoom
        x_display = x_um[row_start:row_stop]
        y_display = np.arange(display.shape[0]) * dx_um
        x_mask = (x_display >= zoom_x_min) & (x_display <= zoom_x_max)
        y_mask = (y_display >= zoom_y_min) & (y_display <= zoom_y_max)

        zoom_data = display[np.ix_(y_mask, x_mask)]
        finite_zoom = zoom_data[np.isfinite(zoom_data)]

        if finite_zoom.size == 0:
            raise ValueError(f"{name}: no finite pixels available in zoom region.")

        # Each method gets its own zoom intensity scale
        zoom_vmin, zoom_vmax = np.percentile(finite_zoom, zoom_percentiles)

        print(
            f"{name:15s} | zoom range = "
            f"{zoom_vmin:.2f} - {zoom_vmax:.2f}"
        )

        extent = [
            x_um[row_start],
            x_um[row_stop - 1] + dx_um,
            0,
            actual_height_um,
        ]

        ax_full, ax_zoom = axes[row, 0], axes[row, 1]

        ax_full.imshow(
            display, cmap=cmap, origin="lower", interpolation="nearest",
            extent=extent, vmin=full_vmin, vmax=full_vmax,
        )
        ax_full.set_aspect("equal", adjustable="box")

        im_zoom = ax_zoom.imshow(
            display, cmap=cmap, origin="lower", interpolation="nearest",
            extent=extent, vmin=zoom_vmin, vmax=zoom_vmax,
        )
        ax_zoom.set_xlim(zoom_x_min, zoom_x_max)
        ax_zoom.set_ylim(zoom_y_min, zoom_y_max)
        ax_zoom.set_aspect("equal", adjustable="box")

        rect_color = colors[name] if colors is not None and name in colors else "tab:red"

        # Rectangle showing the zoomed region
        rect = Rectangle(
            (zoom_x_min, zoom_y_min), zoom_width_um, zoom_width_um,
            fill=False, edgecolor=rect_color, linewidth=1.8,
        )
        ax_full.add_patch(rect)

        # Independent colorbar for each method
        divider = make_axes_locatable(ax_zoom)
        cax = divider.append_axes("right", size="4%", pad=0.06)
        cbar = fig.colorbar(im_zoom, cax=cax)
        cbar.set_label("Intensity")

        ax_full.set_ylabel(name, color=rect_color, fontsize=12, fontweight="bold")
        ax_full.set_yticks([])
        ax_zoom.set_yticks([])

        ax_full.set_xticks(pair_centers_um)
        ax_full.set_xticklabels(
            [f"{spacing:.0f}" for spacing in spacings_nm],
            rotation=45, ha="right",
        )
        ax_zoom.set_xlabel(r"Position ($\mu$m)")

    axes[0, 0].set_title("Full Argolight target")
    axes[0, 1].set_title(f"Zoom: {zoom_spacing:.0f} nm")

    for row in range(n_images - 1):
        axes[row, 0].set_xlabel("")
    axes[-1, 0].set_xlabel("Spacing (nm)")

    return fig, axes

def plot_rbn_sparsity_rl_normalized(
    results, pair_centers_um, spacings_nm, col_range, zoom_spacing_nm,
    rbn_name="RL-RbN", sparsity_name="Sparsity_RL", colors=None,
    zoom_half_width_um=None, cmap="bone", full_aspect_ratio=3.0,
    figsize=(10, 6)
):
    """Compare RbN and Sparsity_RL, normalizing full image and zoom to the corresponding RbN maximum."""

    names = [rbn_name, sparsity_name]
    pair_centers_um = np.asarray(pair_centers_um, dtype=float)
    spacings_nm = np.asarray(spacings_nm, dtype=float)

    x_um = np.asarray(results[rbn_name]["x_um"])
    dx_um = np.median(np.diff(x_um))
    pair_pitch_um = np.median(np.diff(pair_centers_um))

    if zoom_half_width_um is None:
        zoom_half_width_um = pair_pitch_um / 2
    zoom_width_um = 2 * zoom_half_width_um

    target_x_min = pair_centers_um[0] - pair_pitch_um / 2
    target_x_max = pair_centers_um[-1] + pair_pitch_um / 2
    rows = np.where((x_um >= target_x_min) & (x_um <= target_x_max))[0]

    if len(rows) == 0:
        raise ValueError("Argolight target range not found.")

    row_start, row_stop = rows[0], rows[-1] + 1
    full_width_um = x_um[row_stop - 1] + dx_um - x_um[row_start]

    col_left, col_right = col_range
    available_width_px = col_right - col_left
    available_height_um = available_width_px * dx_um

    full_height_um = min(max(full_width_um / full_aspect_ratio, zoom_width_um),
                         available_height_um)
    full_height_px = int(round(full_height_um / dx_um))
    full_height_px = max(1, min(full_height_px, available_width_px))

    col_center = (col_left + col_right) // 2
    display_col_left = col_center - full_height_px // 2
    display_col_right = display_col_left + full_height_px

    if display_col_left < col_left:
        display_col_left, display_col_right = col_left, col_left + full_height_px
    if display_col_right > col_right:
        display_col_right = col_right
        display_col_left = col_right - full_height_px

    actual_height_um = (display_col_right - display_col_left) * dx_um

    zoom_idx = int(np.argmin(np.abs(spacings_nm - zoom_spacing_nm)))
    zoom_spacing = spacings_nm[zoom_idx]
    zoom_center_x = pair_centers_um[zoom_idx]

    zoom_x_min = zoom_center_x - zoom_half_width_um
    zoom_x_max = zoom_center_x + zoom_half_width_um
    zoom_center_y = actual_height_um / 2
    zoom_y_min = zoom_center_y - zoom_width_um / 2
    zoom_y_max = zoom_center_y + zoom_width_um / 2

    x_display = x_um[row_start:row_stop]
    y_display = np.arange(display_col_right - display_col_left) * dx_um
    x_mask = (x_display >= zoom_x_min) & (x_display <= zoom_x_max)
    y_mask = (y_display >= zoom_y_min) & (y_display <= zoom_y_max)

    # RbN reference maxima
    rbn_roi = np.asarray(results[rbn_name]["analysis_roi"])
    rbn_full = rbn_roi[row_start:row_stop, display_col_left:display_col_right].T
    rbn_zoom = rbn_full[np.ix_(y_mask, x_mask)]

    rbn_full_max = np.nanmax(rbn_full)
    rbn_zoom_max = np.nanmax(rbn_zoom)

    print(f"RbN full max : {rbn_full_max:.2f}")
    print(f"RbN zoom max : {rbn_zoom_max:.2f}")

    fig, axes = plt.subplots(
        2, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.10},
        constrained_layout=True
    )

    extent = [x_um[row_start], x_um[row_stop - 1] + dx_um, 0, actual_height_um]

    for row, name in enumerate(names):
        roi = np.asarray(results[name]["analysis_roi"])
        display = roi[row_start:row_stop, display_col_left:display_col_right].T

        full_norm = display / rbn_full_max
        zoom_norm = display / rbn_zoom_max

        ax_full, ax_zoom = axes[row]

        ax_full.imshow(full_norm, cmap=cmap, origin="lower", interpolation="nearest",
                       extent=extent, vmin=0, vmax=1)
        ax_full.set_aspect("equal", adjustable="box")

        im_zoom = ax_zoom.imshow(zoom_norm, cmap=cmap, origin="lower",
                                 interpolation="nearest", extent=extent,
                                 vmin=0, vmax=1)
        ax_zoom.set_xlim(zoom_x_min, zoom_x_max)
        ax_zoom.set_ylim(zoom_y_min, zoom_y_max)
        ax_zoom.set_aspect("equal", adjustable="box")

        color = colors[name] if colors is not None else "tab:red"
        rect = Rectangle((zoom_x_min, zoom_y_min), zoom_width_um, zoom_width_um,
                         fill=False, edgecolor=color, linewidth=1.8)
        ax_full.add_patch(rect)

        divider = make_axes_locatable(ax_zoom)
        cax = divider.append_axes("right", size="4%", pad=0.06)
        cbar = fig.colorbar(im_zoom, cax=cax)
        cbar.set_label("Normalized intensity")

        label = "RL-RbN" if name == rbn_name else "Sparsity RL"
        ax_full.set_ylabel(label, color=color, fontsize=12, fontweight="bold")
        ax_full.set_yticks([])
        ax_zoom.set_yticks([])

        ax_full.set_xticks(pair_centers_um)
        ax_full.set_xticklabels([f"{s:.0f}" for s in spacings_nm],
                                rotation=45, ha="right")
        ax_zoom.set_xlabel(r"Position ($\mu$m)")

    axes[0, 0].set_title("Full Argolight target")
    axes[0, 1].set_title(f"Zoom: {zoom_spacing:.0f} nm")
    axes[0, 0].set_xlabel("")
    axes[1, 0].set_xlabel("Spacing (nm)")

    return fig, axes

def plot_contrast_rbn_sparsity_rl(
    results, spacings_nm, colors,
    rbn_name="RL-RbN", sparsity_name="Sparsity_RL",
    figsize=(4.2, 3.0), ylim=None
):
    """Plot contrast for RbN and Sparsity_RL using lines only."""

    fig, ax = plt.subplots(figsize=figsize)

    for name, label in [(rbn_name, "RL-RbN"), (sparsity_name, "Sparsity RL")]:
        contrast = np.asarray(results[name]["contrast_mean"], dtype=float)
        ax.plot(spacings_nm, contrast, "-", color=colors[name], lw=2, label=label)

    ax.set_xlabel("Spacing (nm)")
    ax.set_ylabel("Contrast")
    ax.set_xticks(spacings_nm)
    ax.set_ylim(*ylim) if ylim is not None else ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig, ax

def plot_std_rbn_sparsity_rl(
    metric_data, spacings_nm, colors,
    rbn_name="RL-RbN", sparsity_name="Sparsity_RL",
    figsize=(4.2, 3.0), ylim=None
):
    """Plot longitudinal normalized STD for RbN and Sparsity_RL using lines only."""

    fig, ax = plt.subplots(figsize=figsize)

    for name, label in [(rbn_name, "RL-RbN"), (sparsity_name, "Sparsity RL")]:
        values = np.array(
            [metric_data[name][float(s)]["std"] for s in spacings_nm],
            dtype=float
        )
        ax.plot(spacings_nm, values, "-", color=colors[name], lw=2, label=label)

    ax.set_xlabel("Spacing (nm)")
    ax.set_ylabel("Longitudinal normalized std")
    ax.set_xticks(spacings_nm)
    ax.set_ylim(*ylim) if ylim is not None else ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()
    return fig, ax

def plot_argolight_metric_schematic(seed=4, figsize=(7.0, 5.2)):
    """Simple simulated schematic for contrast and continuity metrics."""

    rng = np.random.default_rng(seed)

    # Simulated Argolight-like image
    nx, ny = 300, 240
    x, y = np.linspace(0, 1, nx), np.linspace(0, 1, ny)
    line_centers = [0.15, 0.40, 0.53, 0.82]
    sigma_x = 0.025
    image = np.zeros((ny, nx))

    for i, center in enumerate(line_centers):
        noise = rng.normal(size=ny)
        noise = np.convolve(np.pad(noise, 3, mode="reflect"), np.ones(7)/7, mode="valid")
        modulation = 1 + (0.010 if i == 0 else 0.006) * noise
        line = np.exp(-0.5 * ((x - center) / sigma_x)**2)
        image += modulation[:, None] * line[None, :]

    image += 0.001 * rng.normal(size=image.shape)
    image = np.clip(image, 0, None)
    image_display = 1 - image / image.max()

    # Sampling positions
    y_contrast = 0.55
    x_continuity = line_centers[0]
    iy = np.argmin(np.abs(y - y_contrast))
    ix = np.argmin(np.abs(x - x_continuity))

    # Contrast profile across central pair
    mask = (x >= 0.31) & (x <= 0.62)
    x_profile = x[mask]
    contrast_profile = image[iy, mask]
    contrast_profile = np.convolve(np.pad(contrast_profile, 2, mode="reflect"),
                                   np.ones(5)/5, mode="valid")

    middle = (line_centers[1] + line_centers[2]) / 2
    left_mask, right_mask = x_profile < middle, x_profile > middle
    p1 = np.where(left_mask)[0][np.argmax(contrast_profile[left_mask])]
    p2 = np.where(right_mask)[0][np.argmax(contrast_profile[right_mask])]
    valley = p1 + np.argmin(contrast_profile[p1:p2 + 1])

    I1, I2 = contrast_profile[p1], contrast_profile[p2]
    Idip = contrast_profile[valley]
    Ipeak = (I1 + I2) / 2

    # Continuity profile
    continuity = image[:, ix]
    continuity /= np.median(continuity)
    continuity -= np.mean(continuity)
    continuity_plot = np.convolve(np.pad(continuity, 2, mode="reflect"),
                                  np.ones(5)/5, mode="valid")
    sigma = np.std(continuity)

    # Figure layout
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.2],
                  height_ratios=[1, 1], wspace=0.10, hspace=0.18)

    ax_img_c = fig.add_subplot(gs[0, 0])
    ax_contrast = fig.add_subplot(gs[0, 1])
    ax_img_s = fig.add_subplot(gs[1, 0])
    ax_continuity = fig.add_subplot(gs[1, 1])

    # Top-left: contrast zoom
    ax_img_c.imshow(image_display, cmap="gray", origin="lower",
                    extent=[0, 1, 0, 1], interpolation="bilinear")

    contrast_line_pad = 0.035
    ax_img_c.plot(
        [line_centers[1] - contrast_line_pad, line_centers[2] + contrast_line_pad],
        [y_contrast, y_contrast],
        color="tab:red", lw=2
    )

    ax_img_c.set_xlim(0.05, 0.92)
    ax_img_c.set_ylim(0.08, 0.92)
    ax_img_c.set_box_aspect(1)
    ax_img_c.set_xticks([])
    ax_img_c.set_yticks([])

    # Top-right: contrast profile
    ax_contrast.plot(x_profile, contrast_profile, color="tab:red", lw=2.2)
    ax_contrast.scatter([x_profile[p1], x_profile[p2]], [I1, I2],
                        color="black", s=18, zorder=3)
    ax_contrast.scatter(x_profile[valley], Idip, color="black", s=18, zorder=3)

    arrow_x = x_profile[valley]
    ax_contrast.annotate(
        "", xy=(arrow_x, Ipeak), xytext=(arrow_x, Idip),
        arrowprops=dict(arrowstyle="<->", color="tab:blue", lw=1.6)
    )
    ax_contrast.text(arrow_x + 0.012, (Ipeak + Idip)/2, "C",
                     color="tab:blue", fontsize=12, va="center", fontweight="bold")

    ax_contrast.set_xticks([])
    ax_contrast.set_yticks([])
    ax_contrast.set_ylabel("Intensity", fontsize=10)
    ax_contrast.set_title("Contrast", fontsize=11)

    # Bottom-left: continuity zoom
    ax_img_s.imshow(image_display, cmap="gray", origin="lower",
                    extent=[0, 1, 0, 1], interpolation="bilinear")
    ax_img_s.axvline(x_continuity, color="tab:blue", lw=2)

    ax_img_s.set_xlim(0.05, 0.92)
    ax_img_s.set_ylim(0.08, 0.92)
    ax_img_s.set_box_aspect(1)
    ax_img_s.set_xticks([])
    ax_img_s.set_yticks([])

    # Bottom-right: continuity profile
    ax_continuity.plot(y, continuity_plot, color="tab:blue", lw=1.8)

    ylim = max(4*sigma, 1.15*np.max(np.abs(continuity_plot)))
    ax_continuity.set_ylim(-ylim, ylim)

    arrow_x = 0.88
    sigma_visual = 2.5 * sigma

    ax_continuity.annotate(
        "", xy=(arrow_x, sigma_visual), xytext=(arrow_x, -sigma_visual),
        arrowprops=dict(arrowstyle="<->", color="tab:red", lw=2.0)
    )
    ax_continuity.text(arrow_x + 0.025, 0, r"$\sigma$",
                       color="tab:red", fontsize=12, va="center", fontweight="bold")

    ax_continuity.set_xticks([])
    ax_continuity.set_yticks([])
    ax_continuity.set_ylabel("Intensity", fontsize=10)
    ax_continuity.set_title("Continuity", fontsize=11)

    # Arrow-style axes for the two plots on the right
    for ax in [ax_contrast, ax_continuity]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.annotate(
            "", xy=(1.02, 0), xytext=(0, 0),
            xycoords=("axes fraction", "axes fraction"),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
            annotation_clip=False
        )

        ax.annotate(
            "", xy=(0, 1.02), xytext=(0, 0),
            xycoords=("axes fraction", "axes fraction"),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
            annotation_clip=False
        )

    return fig

#%%
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
# mpl.rcParams['axes.titlesize'] = 16
# mpl.rcParams['legend.fontsize'] = 12
plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['figure.dpi'] = 1200
titlesize = 18

#%%
# Load dataset
wd_original = Path.cwd()
parts = list(wd_original.parts)
idx = next((i for i, p in enumerate(parts) if p.lower() == 'notebooks'), None)
wd = Path(*parts[:idx]) if idx is not None and idx > 0 else wd_original

path_argo = os.path.join(wd, 'Data', 'Argolight', 'data-14-12-2023-18-54-47.h5')
path_argo_sparsity = os.path.join(wd, 'Data', 'Argolight', 'Argolight_reconstructed_sparsity.tif')

save_path = os.path.join(wd, 'Data', 'Results')
save_path_server = "/mnt/iitfsvge101/MMS/Data MMS server/Regularization/ISMFLUX/Results/Saved analyzed data/"


#%%
path_data = path_argo
data, meta = mcs.load(path_data)
pxsizex = meta.dx * 1e3
pixel_size_um = meta.dx # Convert nm to μm
meta.Print()
print(data.shape)

data = np.squeeze(data)
border_to_remove = 3
if len(data.shape) == 5:
    data_3D = True
    Nz, Ny, Nx, Nbin, Nch = data.shape
elif len(data.shape) == 4:
    data_3D = False
    Ny, Nx, Nbin, Nch = data.shape

if Nx % 2 == 0:
    data = data[border_to_remove:-border_to_remove + 1, border_to_remove:-border_to_remove + 1, ...]
else:
    data = data[border_to_remove:-border_to_remove, border_to_remove:-border_to_remove, ...]


if data_3D:
    z_slice = Nz // 2
    dset = data[z_slice]
else:
    dset = data

# dset = data.sum(axis=-2)  # taking the last bin
# raw_img = data[...,-1,:].sum(axis=(-1))  # sum over the bins and channels
# raw_close_img = data[..., -1, 12]  # select the close pinhole channel
dset = data.sum(axis=-2)  #
raw_img = data.sum(axis=(-1, -2))  # sum over the bins and channels
raw_close_img = data[...,12].sum(-1)  # select the close pinhole channel


print(f'Dataset shape {dset.shape}, Confocal shape {raw_img.shape}')

colormap = 'bone'


#%%
save_figures = True

#%%
# Plot Data

plt.plot(data.sum(axis=(0, 1, -1)))

fig, ax = plt.subplots(figsize=(5, 5))
gra.ShowImg(raw_img, pxsizex * 1e-3, clabel='Counts', cmap=colormap, fig=fig, ax=ax)
ax.set_title('Confocal image', fontsize=titlesize)
if save_figures:
    fig.savefig(os.path.join(save_path, 'Confocal_image_real.pdf'), bbox_inches="tight", pad_inches=0.1)


#%%
gra.ShowFingerprint(dset)

#%%
results_rl = np.load(os.path.join(save_path_server, 'Argolight_RL_results.npz'))
conf_rl = results_rl['conf_rl']
whiteness_rl = results_rl['whiteness_measure_rl']

results_close_rl = np.load(os.path.join(save_path_server, 'Argolight_RL_close_results.npz'))
conf_close_rl = results_close_rl['conf_rl']
whiteness_close_rl = results_close_rl['whiteness_measure_rl']

results_sparsity_rl = np.load(os.path.join(save_path_server, 'Argolight_RL_results_sparsity.npz'))
conf_sparsity_rl = results_sparsity_rl['conf_rl']
whiteness_sparsity_rl = results_sparsity_rl['whiteness_measure_rl']

s2ism = np.load(os.path.join(save_path_server, 'Argolight_s2ISM_results.npz'))['s2_rec']

#%%
idx_rl = np.argmin(whiteness_rl)
idx_close_rl = np.argmin(whiteness_close_rl)
# idx_close_rl = 7
idx_s2ism = 20
idx_sparsity_rl = np.argmin(whiteness_sparsity_rl)
idx_sparsity_rl = 15

#%%
sparsity = tifffile.imread(path_argo_sparsity)

#%%
results_rl_RbN_56_2000 = np.load(os.path.join(save_path_server, 'Argolight_RL_RbN_result4_K56_2000.npz'))
RL_RbN_rec_56_2000 = results_rl_RbN_56_2000['RL_RbN_rec']
# results_rl_RbN_8_5000 = np.load(os.path.join(save_path_server, 'Argolight_RL_RbN_result28_K8_5000.npz'))
# RL_RbN_rec_8_5000 = results_rl_RbN_8_5000['RL_RbN_rec']

#%%
results_rl_RbN_close_4 = np.load(os.path.join(save_path_server, 'Argolight_RL_RbN_close_result22_K4_4000.npz')) 
RL_RbN_rec_close_4 = results_rl_RbN_close_4['RL_RbN_rec']


#%%
result_s2ism = np.load(os.path.join(save_path_server, 'Argolight_result_s2ISM_RbN5_K8_4000.npz'))
s2ism_RbN = result_s2ism['recon']

#%%
# ============================================================
# 1. INPUT AND ANALYSIS PARAMETERS
# ============================================================

border = 10

RbN_image = RL_RbN_rec_56_2000
RbN_close_image = RL_RbN_rec_close_4
S2ISM_RbN_image = s2ism_RbN

PIXEL_SIZE_UM = pixel_size_um
IMAGE_CMAP = "bone"

images = {
    "Raw": crop_border(raw_img, border),
    "RL": crop_border(conf_rl[idx_rl], border),
    "RL-RbN": crop_border(RbN_image, border),
    "RL late": crop_border(conf_rl[-1], border),
    
    "Sparsity": crop_border(sparsity, border),
    "Sparsity_RL": crop_border(conf_sparsity_rl[idx_sparsity_rl], border),
    
    "Raw close": crop_border(raw_close_img, border),
    "RL close": crop_border(conf_close_rl[idx_close_rl], border),
    "RL-RbN close": crop_border(RbN_close_image, border),
    "RL close late": crop_border(conf_close_rl[-1], border),
    
    "s2ISM": crop_border(s2ism[idx_s2ism,1], border),
    "s2ISM-RbN": crop_border(s2ism_RbN, border),
    "s2ISM late": crop_border(s2ism[-1,1], border),
}

PLOT_GROUPS = {
    "Open confocal": ["Raw", "RL", "RL-RbN", "RL late", "Sparsity", "Sparsity_RL"],
    "Closed confocal": ["Raw close", "RL close", "RL-RbN close", "RL close late"],
    "s2ISM": ["Raw", "s2ISM", "s2ISM-RbN", "s2ISM late"],
}

METHOD_LABELS = {
    "Raw": "Raw",
    "Raw close": "Raw",

    "RL": "RL",
    "RL close": "RL",

    "RL-RbN": "RL-RbN",
    "RL-RbN close": "RL-RbN",

    "RL late": "RL late",
    "RL close late": "RL late",

    "s2ISM": "s2ISM",
    "s2ISM-RbN": "s2ISM-RbN",
    "s2ISM late": "s2ISM late",
    
    "Sparsity": "Sparsity",
    "Sparsity_RL": "Sparsity RL",
}


COLORS = {
    "Raw": "#02B623",
    "Raw close": "#015611",
    "RL": "#026AB4",
    "RL close": "#012C4B",
    "RL-RbN": "#864401",
    "RL-RbN close": "#552B02",
    "RL late": "#A40286",
    "RL close late": "#580248",
    "s2ISM": "#0393F9",
    "s2ISM-RbN": "#FA7E02",
    "s2ISM late": "#FA05CD",
    "Sparsity": "#02D995",
    "Sparsity_RL": "#EA1111",
}

SPACINGS_NM = np.arange(0, 391, 30, dtype=float)

# Crop after rotation
X_SLICE = slice(450, -450)
Y_SLICE = slice(250, -250)

# Target horizontal range
TARGET_THRESHOLD_FRACTION = 0.10
TARGET_LEFT_INNER_FRACTION = 0.04
TARGET_RIGHT_INNER_FRACTION = 0.03

# Manual exclusion of the visibly distorted left line-end region
ANALYSIS_LEFT_MIN_PX = 230

# RbN directional preprocessing
RBN_DENOISE_SIGMA_PX = 2.0
RBN_DENOISE_BLEND = 0.7
RBN_DENOISE_MEDIAN_SIZE = 1

# Contrast analysis
#STRIP_FRACTION = 1/16 #0.05
STRIP_WIDTH_PX = 40
SMOOTH_SIGMA = 0.5
PEAK_SEARCH_UM = 0.20
VALLEY_SEARCH_UM = 0.12
MIN_PEAK_RATIO = 0.5
REFINE_PX = 1
CONFIDENCE_LEVEL = 0.95
MIN_VALID_FRACTION = 0.2

# Longitudinal analysis
LONGITUDINAL_HALF_WIDTH_PX = 0
#LONGITUDINAL_LEFT_MARGIN_UM = 5.0
LONGITUDINAL_RIGHT_MARGIN_UM = 0

# Longitudinal hole metric
LONGITUDINAL_SMOOTH_SIGMA_PX = 2.0
LONGITUDINAL_BASELINE_SIGMA_PX = 20.0


#%%
# ============================================================
# 2. ESTIMATE AND REFINE COMMON ROTATION
# ============================================================

INITIAL_ANGLE = estimate_grid_angle(images["Raw"])

ANGLE_REF, ROTATION_INFO = refine_rotation_angle(
    image=images["Raw"],
    initial_angle_deg=INITIAL_ANGLE,
    pixel_size_um=PIXEL_SIZE_UM,
    x_slice=X_SLICE,
    y_slice=Y_SLICE,
    threshold_fraction=TARGET_THRESHOLD_FRACTION,
    left_inner_fraction=TARGET_LEFT_INNER_FRACTION,
    right_inner_fraction=TARGET_RIGHT_INNER_FRACTION,
    n_sections=12,
    max_shift_um=0.5,
    coarse_range_deg=0.8,
    coarse_step_deg=0.05,
    fine_range_deg=0.08,
    fine_step_deg=0.005,
)

print(f"Initial angle : {INITIAL_ANGLE:.4f} deg")
print(f"Refined angle : {ANGLE_REF:.4f} deg")
print(f"Correction    : {ANGLE_REF - INITIAL_ANGLE:+.4f} deg")

#%%
# ============================================================
# FULL-SIZE IMAGES FOR DISPLAY
# ============================================================

rbn_clean_display = denoise_rbn_full_for_display(
    RbN_image, ANGLE_REF, sigma_along_px=RBN_DENOISE_SIGMA_PX, blend=RBN_DENOISE_BLEND)

rbn_close_clean_display = denoise_rbn_full_for_display(
    RbN_close_image, ANGLE_REF, sigma_along_px=RBN_DENOISE_SIGMA_PX, blend=RBN_DENOISE_BLEND)

s2ism_rbn_clean_display = denoise_rbn_full_for_display(
    image=S2ISM_RbN_image, angle_deg=ANGLE_REF, sigma_along_px=RBN_DENOISE_SIGMA_PX, blend=RBN_DENOISE_BLEND)

DISPLAY_IMAGES = {
    "Raw": crop_border(raw_img, border),
    "RL": crop_border(conf_rl[idx_rl], border),
    "RL-RbN": crop_border(rbn_clean_display, border),
    "RL late": crop_border(conf_rl[-1], border),
    
    "Sparsity": crop_border(sparsity, border),
    "Sparsity_RL": crop_border(conf_sparsity_rl[idx_sparsity_rl], border),

    "Raw close": crop_border(raw_close_img, border),
    "RL close": crop_border(conf_close_rl[idx_close_rl], border),
    "RL-RbN close": crop_border(rbn_close_clean_display, border),
    "RL close late": crop_border(conf_close_rl[-1], border),

    "s2ISM": crop_border(s2ism[idx_s2ism,1], border),
    "s2ISM-RbN": crop_border(s2ism_rbn_clean_display, border),
    "s2ISM late": crop_border(s2ism[-1,1], border),
}

DISPLAY_VMAX = {
    "RL close late": np.percentile(conf_close_rl[-1], 99),
    "s2ISM late": np.percentile(s2ism[-1,1], 99.9),
}


#%%
# ============================================================
# 3. ROTATE ALL IMAGES
# ============================================================

results = {}

for name, image in images.items():
    results[name] = prepare_rotated_roi(
        image=image,
        angle_deg=ANGLE_REF,
        pixel_size_um=PIXEL_SIZE_UM,
        x_slice=X_SLICE,
        y_slice=Y_SLICE,
    )

#%%
############################################
# Directional preprocessing only for RbN ###
############################################

results["RL-RbN"] = denoise_rotated_parallel_lines(
    result=results["RL-RbN"],
    sigma_along_px=RBN_DENOISE_SIGMA_PX,
    median_size_along=RBN_DENOISE_MEDIAN_SIZE,
    blend=RBN_DENOISE_BLEND,
)

results["s2ISM-RbN"] = denoise_rotated_parallel_lines(
    result=results["s2ISM-RbN"],
    sigma_along_px=RBN_DENOISE_SIGMA_PX,
    median_size_along=RBN_DENOISE_MEDIAN_SIZE,
    blend=RBN_DENOISE_BLEND,
)

results["RL-RbN close"] = denoise_rotated_parallel_lines(
    result=results["RL-RbN close"],
    sigma_along_px=RBN_DENOISE_SIGMA_PX,
    median_size_along=RBN_DENOISE_MEDIAN_SIZE,
    blend=RBN_DENOISE_BLEND,
)

#%%
########################################################
###### PLOT FULL-SIZE IMAGES FOR DISPLAY ###############
########################################################

fig, axes = plot_full_images_groups(
    display_images=DISPLAY_IMAGES,
    groups=PLOT_GROUPS,
    method_labels=METHOD_LABELS,
    colors=COLORS,
    pxsize_um=PIXEL_SIZE_UM,
    cmap=IMAGE_CMAP,
    vmax_dict=DISPLAY_VMAX,
    figsize=(20, 12),
)

plt.show()

if save_figures:
    fig.savefig(os.path.join(save_path, "Argolight_images_full.pdf"), bbox_inches="tight", pad_inches=0.1)
    


#%%
# ============================================================
# 4. DEFINE COMMON ANALYSIS REGION
# ============================================================

target_range_info = find_target_column_range(
    result=results["Raw"],
    threshold_fraction=TARGET_THRESHOLD_FRACTION,
    smooth_sigma_px=5,
    left_inner_fraction=TARGET_LEFT_INNER_FRACTION,
    right_inner_fraction=TARGET_RIGHT_INNER_FRACTION,
)

detected_left = target_range_info["left"]
detected_right = target_range_info["right"]

# Exclude the visibly distorted left line-end region
analysis_left = max(
    detected_left,
    ANALYSIS_LEFT_MIN_PX,
)

TARGET_COL_RANGE = (
    analysis_left,
    detected_right,
)

print("Automatically detected range:",
      (detected_left, detected_right))

print("Final analysis range:",
      TARGET_COL_RANGE)

print("Final analysis width:",
      TARGET_COL_RANGE[1] - TARGET_COL_RANGE[0],
      "px")

fig, ax = plot_analysis_roi(
    result=results["Raw"],
    col_range=TARGET_COL_RANGE,
    title="Region used for quantitative analysis",
    cmap=IMAGE_CMAP,
)

plt.show()


#%%
# ============================================================
# 5. GLOBAL LINE PROFILES
# ============================================================

for name in results:
    results[name] = compute_line_profile(
        result=results[name],
        col_range=TARGET_COL_RANGE,
    )
    
#%%
# ============================================================
# 6. PAIR GEOMETRY FROM RAW
# ============================================================

FIRST_CENTER_UM, PAIR_PITCH_UM, EXPECTED_CENTERS_UM = estimate_pair_geometry(
    x_um=results["Raw"]["x_um"],
    profile=results["Raw"]["mean"],
    n_pairs=len(SPACINGS_NM),
    smooth_sigma_px=6,
)

PAIR_CENTERS_UM, LEFT_MINIMA_UM, RIGHT_MINIMA_UM = refine_pair_centers(
    x_um=results["Raw"]["x_um"],
    profile=results["Raw"]["mean"],
    expected_centers_um=EXPECTED_CENTERS_UM,
    min_offset_um=0.30,
    max_offset_um=0.85,
    smooth_sigma_px=0.5,
    max_center_shift_um=0.12,
)

print(f"First pair center: {FIRST_CENTER_UM:.3f} um")
print(f"Pair pitch       : {PAIR_PITCH_UM:.3f} um")


#%%
# ============================================================
# ROTATED IMAGES WITH ZOOM - OPEN CONFOCAL
# ============================================================

IMAGE_ZOOM_SPACING_NM = {
    "Open confocal": 210,
    "Closed confocal": 210,
    "s2ISM": 210,
}

results_open = {
    name: results[name]
    for name in PLOT_GROUPS["Open confocal"]
}

fig, axes = plot_rotated_images_with_zoom(
    results=results_open,
    pair_centers_um=PAIR_CENTERS_UM,
    spacings_nm=SPACINGS_NM,
    col_range=TARGET_COL_RANGE,
    zoom_spacing_nm=IMAGE_ZOOM_SPACING_NM["Open confocal"],
    colors=COLORS,
    zoom_half_width_um=None,
    cmap=IMAGE_CMAP,
    figsize=(14, 19.5),
)

fig.suptitle("Open confocal", fontsize=16, fontweight="bold")
plt.show()

if save_figures:
    fig.savefig(
        os.path.join(save_path, "Argolight_rotated_zoom_open.pdf"),
        bbox_inches="tight",
        pad_inches=0.1,
    )
    
#%%
# ============================================================
# ROTATED IMAGES WITH ZOOM SPARSITY - OPEN CONFOCAL
# ============================================================

fig, axes = plot_rbn_sparsity_rl_normalized(
    results=results,
    pair_centers_um=PAIR_CENTERS_UM,
    spacings_nm=SPACINGS_NM,
    col_range=TARGET_COL_RANGE,
    zoom_spacing_nm=IMAGE_ZOOM_SPACING_NM["Open confocal"],
    rbn_name="RL-RbN",
    sparsity_name="Sparsity_RL",
    colors=COLORS,
    cmap=IMAGE_CMAP,
    figsize=(10, 6)
)

plt.show()

if save_figures:
    fig.savefig(
        os.path.join(save_path, "Argolight_rotated_zoom_open_sparsity.pdf"),
        bbox_inches="tight",
        pad_inches=0.1,
    )

    
#%%
# ============================================================
# ROTATED IMAGES WITH ZOOM - CLOSED CONFOCAL
# ============================================================

results_close = {
    name: results[name]
    for name in PLOT_GROUPS["Closed confocal"]
}

fig, axes = plot_rotated_images_with_zoom(
    results=results_close,
    pair_centers_um=PAIR_CENTERS_UM,
    spacings_nm=SPACINGS_NM,
    col_range=TARGET_COL_RANGE,
    zoom_spacing_nm=IMAGE_ZOOM_SPACING_NM["Closed confocal"],
    colors=COLORS,
    zoom_half_width_um=None,
    cmap=IMAGE_CMAP,
    figsize=(14, 19.5),
)

fig.suptitle("Closed confocal", fontsize=16, fontweight="bold")

plt.show()

if save_figures:
    fig.savefig(
        os.path.join(save_path, "Argolight_rotated_zoom_closed.pdf"),
        bbox_inches="tight",
        pad_inches=0.1,
    )

#%%
# ============================================================
# ROTATED IMAGES WITH ZOOM - s2ISM
# ============================================================

results_s2ism = {
    name: results[name]
    for name in PLOT_GROUPS["s2ISM"]
}

fig, axes = plot_rotated_images_with_zoom(
    results=results_s2ism,
    pair_centers_um=PAIR_CENTERS_UM,
    spacings_nm=SPACINGS_NM,
    col_range=TARGET_COL_RANGE,
    zoom_spacing_nm=IMAGE_ZOOM_SPACING_NM["s2ISM"],
    colors=COLORS,
    zoom_half_width_um=None,
    cmap=IMAGE_CMAP,
    figsize=(14, 19.5),
)

fig.suptitle("s2ISM", fontsize=16,fontweight="bold")

plt.show()

if save_figures:
    fig.savefig(os.path.join(save_path, "Argolight_rotated_zoom_s2ism.pdf"),
        bbox_inches="tight", pad_inches=0.1)

#%%
# ============================================================
# 7. PLOT GLOBAL LINE PROFILES
# ============================================================

PROFILE_ZOOM_SPACING_NM = {
    "Open confocal": 210,
    "Closed confocal": 210,
    "s2ISM": 210,
}

fig, axes = plot_profiles(
    results=results, groups=PLOT_GROUPS, method_labels=METHOD_LABELS,
    pair_centers_um=PAIR_CENTERS_UM, spacings_nm=SPACINGS_NM,
    zoom_spacing_nm=PROFILE_ZOOM_SPACING_NM, colors=COLORS)

plt.show()

if save_figures:
    fig.savefig(os.path.join(save_path, "Argolight_global_profiles_full.pdf"),
                bbox_inches="tight", pad_inches=0.1)
    
    
#%%
# ============================================================
# 8. STRIP GEOMETRY FROM RAW
# ============================================================

STRIP_SHIFTS_UM, STRIP_SHIFT_INFO = estimate_strip_shifts(
    result=results["Raw"],
    pair_centers_um=PAIR_CENTERS_UM,
    spacings_nm=SPACINGS_NM,
    reference_spacings=(300, 330, 360),
    strip_fraction=None,
    strip_width_px=STRIP_WIDTH_PX,
    col_range=TARGET_COL_RANGE,
    smooth_sigma=SMOOTH_SIGMA,
    peak_search_um=PEAK_SEARCH_UM,
    valley_search_um=VALLEY_SEARCH_UM,
    min_peak_ratio=MIN_PEAK_RATIO,
    refine_px=REFINE_PX,
    confidence_level=CONFIDENCE_LEVEL,
)

#%%
# ============================================================
# 9. CONTRAST AND VALID PAIR FRACTION
# ============================================================

for name in results:
    results[name] = compute_contrast(
        result=results[name],
        pair_centers_um=PAIR_CENTERS_UM,
        spacings_nm=SPACINGS_NM,
        strip_shifts_um=STRIP_SHIFTS_UM,
        strip_fraction=None,
        strip_width_px=STRIP_WIDTH_PX,
        col_range=TARGET_COL_RANGE,
        smooth_sigma=SMOOTH_SIGMA,
        peak_search_um=PEAK_SEARCH_UM,
        valley_search_um=VALLEY_SEARCH_UM,
        min_peak_ratio=MIN_PEAK_RATIO,
        refine_px=REFINE_PX,
        confidence_level=CONFIDENCE_LEVEL,
        min_valid_fraction=MIN_VALID_FRACTION,
    )
    

#%%
# ============================================================
# 10. PLOT CONTRAST
# ============================================================

fig, axes = plot_contrast(
    results=results,
    groups=PLOT_GROUPS,
    method_labels=METHOD_LABELS,
    spacings_nm=SPACINGS_NM,
    colors=COLORS,
    shared_y=False,
)

plt.show()

if save_figures:
    fig.savefig(os.path.join(save_path, "Argolight_contrast_full.pdf"), bbox_inches="tight", pad_inches=0.1)
    
#%%
fig, ax = plot_contrast_rbn_sparsity_rl(
    results=results,
    spacings_nm=SPACINGS_NM,
    colors=COLORS,
)
plt.show()

if save_figures:
    fig.savefig(os.path.join(save_path, "Argolight_contrast_sparsity_rbn.pdf"), bbox_inches="tight", pad_inches=0.1)
    

#%%
# ============================================================
# 11. PLOT VALID PAIR FRACTION
# ============================================================

fig, axes = plot_valid_pair_fraction(
    results=results, groups=PLOT_GROUPS, method_labels=METHOD_LABELS,
    spacings_nm=SPACINGS_NM, colors=COLORS,
    min_valid_fraction=MIN_VALID_FRACTION)

plt.show()

if save_figures:
    fig.savefig(os.path.join(save_path, "Argolight_valid_fraction_full.pdf"), bbox_inches="tight", pad_inches=0.1)

#%%
# ============================================================
# 12. LONGITUDINAL ANALYSIS REGION
# ============================================================

right_margin_px = int(round(
    LONGITUDINAL_RIGHT_MARGIN_UM / PIXEL_SIZE_UM
))

LONGITUDINAL_COL_RANGE = (
    TARGET_COL_RANGE[0],
    TARGET_COL_RANGE[1] - right_margin_px,
)

LONGITUDINAL_WIDTH_UM = (
    LONGITUDINAL_COL_RANGE[1]
    - LONGITUDINAL_COL_RANGE[0]
) * PIXEL_SIZE_UM

print(
    "Longitudinal range:",
    LONGITUDINAL_COL_RANGE,
)

print(
    f"Longitudinal analysis width: "
    f"{LONGITUDINAL_WIDTH_UM:.2f} um"
)

#%%
# ============================================================
# 13. LONGITUDINAL LINE PROFILES
# ============================================================

LONGITUDINAL_PROFILES = extract_longitudinal_profiles(
    results=results,
    pair_centers_um=PAIR_CENTERS_UM,
    spacings_nm=SPACINGS_NM,
    col_range=LONGITUDINAL_COL_RANGE,
    reference_name="Raw",
    transverse_half_width_px=LONGITUDINAL_HALF_WIDTH_PX,
    smooth_sigma=SMOOTH_SIGMA,
    refine_px=REFINE_PX,
)

#%%
# ============================================================
# 14. PLOT LONGITUDINAL INTENSITY I(y)
# ============================================================

fig, axes = plot_longitudinal_profiles(
    longitudinal_profiles=LONGITUDINAL_PROFILES, groups=PLOT_GROUPS,
    method_labels=METHOD_LABELS, spacings_nm=SPACINGS_NM, colors=COLORS,
    min_color_strength=0.30, normalize=False, share_y_between_groups=False)

plt.show()

if save_figures:
    fig.savefig(os.path.join(save_path, "Argolight_longitudinal_profiles_full.pdf"), bbox_inches="tight", pad_inches=0.1)
    
    
#%%
# ============================================================
# 15. NORMALIZED LONGITUDINAL INTENSITY
# ============================================================

fig, axes = plot_longitudinal_profiles(
    longitudinal_profiles=LONGITUDINAL_PROFILES, groups=PLOT_GROUPS,
    method_labels=METHOD_LABELS, spacings_nm=SPACINGS_NM, colors=COLORS,
    min_color_strength=0.30, normalize=True, share_y_between_groups=True)

plt.show()

if save_figures:
    fig.savefig(os.path.join(save_path, "Argolight_longitudinal_profiles_normalized_full.pdf"), bbox_inches="tight", pad_inches=0.1)
    
    
#%%
# ============================================================
# 16. LONGITUDINAL NORMALIZED STD
# ============================================================

LONGITUDINAL_METRICS = compute_longitudinal_metrics(
    longitudinal_profiles=LONGITUDINAL_PROFILES,
    spacings_nm=SPACINGS_NM,
)

STD_Y_RANGES = {
    "Open confocal":   {"lower": (0.00, 0.15), "upper": (0.20, 0.50)},
    "Closed confocal": {"lower": (0.08, 0.20), "upper": (0.85, 1.20)},
    "s2ISM":           {"lower": (0.03, 0.18), "upper": (0.35, 0.55)},
}

fig, axes = plot_longitudinal_std(
    metric_data=LONGITUDINAL_METRICS,
    groups=PLOT_GROUPS,
    method_labels=METHOD_LABELS,
    spacings_nm=SPACINGS_NM,
    colors=COLORS,
    shared_y=False,
    y_ranges=STD_Y_RANGES,
)

plt.show()

if save_figures:
    fig.savefig(os.path.join(save_path, 'Argolight_longitudinal_std_full.pdf'), bbox_inches="tight", pad_inches=0.1)

#%%
fig, ax = plot_std_rbn_sparsity_rl(
    metric_data=LONGITUDINAL_METRICS,
    spacings_nm=SPACINGS_NM,
    colors=COLORS,
)
plt.show()

if save_figures:
    fig.savefig(os.path.join(save_path, 'Argolight_longitudinal_std_sparsity_rbn.pdf'), bbox_inches="tight", pad_inches=0.1)


#%%
# ============================================================
# 17. LONGITUDINAL HOLE RMS
# ============================================================

LONGITUDINAL_HOLES = compute_longitudinal_hole_metric(
    longitudinal_profiles=LONGITUDINAL_PROFILES,
    spacings_nm=SPACINGS_NM,
    smooth_sigma_px=LONGITUDINAL_SMOOTH_SIGMA_PX,
    baseline_sigma_px=LONGITUDINAL_BASELINE_SIGMA_PX,
)

#%%
# ============================================================
# LOAD RL NA SWEEP - MINIMUM WHITENESS
# ============================================================

file_path = os.path.join(save_path_server, "Argolight_RL_NA_sweep.npz")
data_na = np.load(file_path)

NA_VALUES = data_na["na_values"]
rl_whiteness_min = data_na["rl_whiteness_min"]
best_iterations = data_na["best_iterations"]

print("NA values:", NA_VALUES)
print("RL minimum-whiteness shape:", rl_whiteness_min.shape)
print("Best iterations:", best_iterations)

# ============================================================
# CONTRAST FOR EACH NA - MINIMUM WHITENESS
# ============================================================

contrast_na = []
contrast_std_na = []

for i, na in enumerate(NA_VALUES):
    print(f"Contrast analysis: NA = {na:.1f} | iter = {best_iterations[i]}")

    # Same rotation and common analysis geometry used for all NA values
    result = prepare_rotated_roi(
        image=crop_border(rl_whiteness_min[i], border),
        angle_deg=ANGLE_REF,
        pixel_size_um=PIXEL_SIZE_UM,
        x_slice=X_SLICE,
        y_slice=Y_SLICE,
    )

    result = compute_line_profile(
        result=result,
        col_range=TARGET_COL_RANGE,
    )

    result = compute_contrast(
        result=result,
        pair_centers_um=PAIR_CENTERS_UM,
        spacings_nm=SPACINGS_NM,
        strip_shifts_um=STRIP_SHIFTS_UM,
        strip_fraction=None,
        strip_width_px=STRIP_WIDTH_PX,
        col_range=TARGET_COL_RANGE,
        smooth_sigma=SMOOTH_SIGMA,
        peak_search_um=PEAK_SEARCH_UM,
        valley_search_um=VALLEY_SEARCH_UM,
        min_peak_ratio=MIN_PEAK_RATIO,
        refine_px=REFINE_PX,
        confidence_level=CONFIDENCE_LEVEL,
        min_valid_fraction=MIN_VALID_FRACTION,
    )

    contrast_na.append(result["contrast_mean"])
    contrast_std_na.append(result["contrast_std"])

contrast_na = np.asarray(contrast_na)
contrast_std_na = np.asarray(contrast_std_na)

print("Contrast shape:", contrast_na.shape)

# ============================================================
# PLOT CONTRAST VS SPACING FOR DIFFERENT NA
# ============================================================

fig, ax = plt.subplots(figsize=(7, 4.5))

cmap = plt.cm.Blues
na_colors = cmap(np.linspace(0.30, 1.00, len(NA_VALUES)))

for i, na in enumerate(NA_VALUES):
    ax.plot(
        SPACINGS_NM,
        contrast_na[i],
        "o-",
        color=na_colors[i],
        lw=1.5,
        ms=3,
        label=f"NA = {na:.1f}",
    )

ax.set_xlabel("Spacing (nm)")
ax.set_ylabel("Contrast")
ax.set_xticks(SPACINGS_NM)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.2)

ax.legend(
    fontsize=7,
    ncol=2,
    title="Numerical aperture",
    title_fontsize=8,
)

fig.tight_layout()
plt.show()
fig.savefig(os.path.join(save_path, "Argolight_contrast_vs_NA.pdf"), bbox_inches="tight", pad_inches=0.1)

#%%
fig = plot_argolight_metric_schematic()
plt.show()
fig.savefig(os.path.join(save_path, "Argolight_metric_schematic.pdf"), bbox_inches="tight", pad_inches=0.1)
