# brighteyes

brighteyes is a Python toolbox for image scanning microscopy (ISM), fluorescence lifetime ISM (FLISM) and fluorescence fluctuation spectroscopy (FFS) with array detectors.

```
pip install brighteyes[imaging]
```

Install the extra of each subpackage you need:

| Subpackage | Content | Extra |
|---|---|---|
| `brighteyes.core` | empty for now, reserved for code shared by the other subpackages | none |
| `brighteyes.mcs` | Reading and preparing HDF5 files from the MCS acquisition software | `mcs` |
| `brighteyes.ttm` | Reading data from the time-tagging module | `ttm` |
| `brighteyes.imaging` | ISM, FLISM, s2ISM, flus2ISM, varreg, psf-distiller and birfi; also installs `mcs` | `imaging` |
| `brighteyes.spectroscopy` | FFS; also installs `imaging` and `ttm` | `spectroscopy` |
