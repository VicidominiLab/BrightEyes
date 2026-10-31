# brighteyes

brighteyes is a Python toolbox for image scanning microscopy (ISM), fluorescence lifetime ISM (FLISM) and fluorescence fluctuation spectroscopy (FFS) with array detectors.

```
pip install brighteyes[imaging]
```

Install the extra of each subpackage you need:

| Subpackage | Content | Extra |
|---|---|---|
| `brighteyes.core` | empty for now, reserved for code shared by the other subpackages | none |
| `brighteyes.imaging` | ISM, with FLISM as `brighteyes.imaging.flism` | `imaging` |
| `brighteyes.spectroscopy` | FFS | `spectroscopy` |
