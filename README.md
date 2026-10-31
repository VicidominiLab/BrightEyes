# brighteyes

brighteyes is a Python toolbox for image scanning microscopy (ISM), fluorescence lifetime ISM (FLISM) and fluorescence fluctuation spectroscopy (FFS) with array detectors.

```
pip install brighteyes[imaging]
```

Each subpackage has its own extra:

| Subpackage | Content | Extra |
|---|---|---|
| `brighteyes.core` | empty in phase 1 | none |
| `brighteyes.imaging` | ISM, with FLISM as `brighteyes.imaging.flism` | `imaging` |
| `brighteyes.spectroscopy` | FFS | `spectroscopy` |
