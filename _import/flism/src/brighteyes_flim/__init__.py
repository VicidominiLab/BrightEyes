"""Public package API for BrightEyes FLIM utilities."""

from brighteyes_mcs_file import (
    Alignment,
    H5DataCalibrator,
    H5OutputBuilder,
    add_output_to_h5_file,
    build_h5_output,
    calibrate_h5_file,
    mcs,
    show_h5_structure,
    show_h5_structure_html,
)

from .flism_data import FlimData
from .graph_tools import (
    load_calibration_fit_traces,
    load_calibration_summary,
    normalize_histogram,
    plot_calibration_fit_traces,
    plot_calibration_lifetime_summary,
    plot_calibration_shift_summary,
    plot_channel_skew_correction,
    plot_equalized_lifetime_summary,
    plot_lifetime_histogram,
    plot_lifetime_summary,
    threshold_lifetime_map,
    weighted_lifetime_stats,
)
from .tools_phasor import (
    apply_lifetime_equalizer,
    build_lifetime_equalizer,
    calculate_m_phi_tau_phi_tau_m,
    calculate_tau_m,
    calculate_tau_phi,
    equalized_lifetime_tick_values,
    estimate_lifetime_from_birfi,
    estimate_lifetime_from_circmean,
    estimate_lifetime_from_log,
    linear_shift,
    plot_phasor,
    show_flim_equalized,
    plot_tau,
    plot_universal_circle,
    calculate_phasor,
)
from ._fit_maps import generate_fit_maps


IRF_from_data_deconvolution = Alignment.IRF_from_data_deconvolution
curve_fit_circular = Alignment.curve_fit_circular
fit_data_with_ref_or_irf = Alignment.fit_data_with_ref_or_irf
fit_model_data = Alignment.fit_model_data
hist_for_plot = Alignment.hist_for_plot
model_data = Alignment.model_data
perform_fit_data = Alignment.perform_fit_data
phasor_delay_from_hist = Alignment.phasor_delay_from_hist
rectangular_IRF = Alignment.rectangular_IRF
sum_channel_applying_shifts = Alignment.sum_channel_applying_shifts
centroid = Alignment.centroid
clean_irf = Alignment.clean_irf
clean_irf_stack = Alignment.clean_irf_stack
fit_maps_to_stack = Alignment.fit_maps_to_stack
Alignment.generate_fit_maps = staticmethod(generate_fit_maps)

__all__ = [
    "Alignment",
    "FlimData",
    "H5DataCalibrator",
    "H5OutputBuilder",
    "IRF_from_data_deconvolution",
    "add_output_to_h5_file",
    "apply_lifetime_equalizer",
    "build_h5_output",
    "build_lifetime_equalizer",
    "calculate_m_phi_tau_phi_tau_m",
    "calculate_tau_m",
    "calculate_tau_phi",
    "equalized_lifetime_tick_values",
    "estimate_lifetime_from_birfi",
    "estimate_lifetime_from_circmean",
    "estimate_lifetime_from_log",
    "calibrate_h5_file",
    "centroid",
    "clean_irf",
    "clean_irf_stack",
    "curve_fit_circular",
    "fit_data_with_ref_or_irf",
    "fit_maps_to_stack",
    "fit_model_data",
    "generate_fit_maps",
    "hist_for_plot",
    "linear_shift",
    "load_calibration_fit_traces",
    "load_calibration_summary",
    "mcs",
    "model_data",
    "normalize_histogram",
    "perform_fit_data",
    "phasor_delay_from_hist",
    "plot_calibration_fit_traces",
    "plot_calibration_lifetime_summary",
    "plot_calibration_shift_summary",
    "plot_channel_skew_correction",
    "plot_equalized_lifetime_summary",
    "plot_lifetime_histogram",
    "plot_lifetime_summary",
    "plot_phasor",
    "show_flim_equalized",
    "plot_tau",
    "plot_universal_circle",
    "calculate_phasor",
    "rectangular_IRF",
    "show_h5_structure",
    "show_h5_structure_html",
    "sum_channel_applying_shifts",
    "threshold_lifetime_map",
    "weighted_lifetime_stats",
]
