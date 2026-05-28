BrightEyes-FLIM
---------------

## Installation

Install `brighteyes-flim` from PyPI:

    pip install brighteyes-flim

You can also install the latest development version directly from GitHub:

    pip install git+https://github.com/VicidominiLab/BrightEyes-Flim

In case of local development:
    
    git clone https://github.com/VicidominiLab/BrightEyes-Flim.git
    cd BrightEyes-Flim
    pip install -e .

`brighteyes_flim` re-exports the fitting and calibration helpers from
`brighteyes_mcs_file`, including the optional `model_fn`, `p0`, `bounds`, and
`parameter_names` arguments for custom multi-parameter fit models. This includes
`perform_fit_data`, `fit_data_with_ref_or_irf`, `calibrate_h5_file`, and
`generate_fit_maps`.

The HDF5 helpers understand the BrightEyes MCS 0.0.5 layout, including
`/raw/spad`, `/raw/aux`, and `/calibration/results/<product>/...`, while
keeping reader support for older raw acquisition files with root-level `data`
and `data_channels_extra` datasets.

For plotting calibrated files, use `load_calibration_summary(...)` and
`load_calibration_fit_traces(...)` instead of hard-coding calibration paths.
